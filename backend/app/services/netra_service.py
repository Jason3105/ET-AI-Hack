"""NETRA — lightweight Groq Vision banknote review.

NETRA deliberately performs no local CV, OCR, TensorFlow, TFLite or YOLO
inference.  The image is sent to Groq's vision model and the response is
normalised into the API contract used by the web client.  This keeps a Render
instance small while retaining a human-review-first result for a high-stakes
use case such as currency verification.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_SUPPORTED_DENOMS = ("₹2000", "₹500", "₹200", "₹100", "₹50", "₹20", "₹10", "unknown")
_KNOWN_COUNTERFEIT_PREFIXES = frozenset({"XY12", "AB78", "MN34", "PQ56"})
_SERIAL_RE = re.compile(r"^[A-Z]{2,3}\d{6,7}$")
_SPECIMEN_RE = re.compile(r"^(?:0|O)AA0{6,7}$", re.I)
_MAX_IMAGE_BYTES = 15 * 1024 * 1024
_scan_store: dict[str, dict[str, Any]] = {}
_scan_order: list[str] = []

_SYSTEM_PROMPT = """You are NETRA, an assistive Indian banknote visual-review system.
Review the supplied image only. You cannot authenticate currency or replace a bank,
RBI, or trained examiner. Do not claim certainty where image quality or visibility
is insufficient. Treat a clear 'SPECIMEN', a specimen serial, or obvious printed
copy as counterfeit/training material; otherwise use SUSPICIOUS rather than
COUNTERFEIT when evidence is inconclusive.

Return ONLY JSON with exactly this shape:
{
  "is_indian_banknote": true,
  "verdict": "AUTHENTIC|SUSPICIOUS|COUNTERFEIT",
  "confidence": 0.0,
  "denomination": "₹500|₹200|₹100|₹50|₹20|₹10|₹2000|unknown",
  "denomination_confidence": 0.0,
  "serial_number": "string or null",
  "features": [
    {"name":"Security Thread","status":"pass|fail|warn","confidence":0.0,"description":"brief evidence"}
  ],
  "detection_reason": "brief plain-language summary"
}
Use 0..1 for confidence values. Include 4-8 visible/checkable security features.
If this is not a single Indian banknote, set is_indian_banknote false and explain.
Never invent a serial number or feature that is not visible."""


def _get_client():
    if not settings.groq_api_key:
        raise ValueError("NETRA requires GROQ_API_KEY. Add it as a Render environment secret.")
    try:
        from groq import Groq
        return Groq(api_key=settings.groq_api_key)
    except ImportError as exc:
        raise RuntimeError("Groq SDK is not installed") from exc


def _image_data_url(image_bytes: bytes) -> str:
    if not image_bytes:
        raise ValueError("Uploaded file is empty")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise ValueError("Image is too large. Upload an image smaller than 15 MB.")
    if image_bytes.startswith(b"\x89PNG"):
        mime = "image/png"
    elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        mime = "image/webp"
    elif image_bytes[:2] == b"BM":
        mime = "image/bmp"
    elif image_bytes[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    else:
        raise ValueError("Unsupported image. Upload a JPEG, PNG, WEBP, or BMP image.")
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Groq returned an invalid NETRA analysis")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Groq returned an invalid NETRA analysis")
    return parsed


def _confidence(value: Any, default: float = 0.5) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 2)
    except (TypeError, ValueError):
        return default


def _normalise_feature(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict) or not str(item.get("name", "")).strip():
        return None
    status = str(item.get("status", "warn")).lower()
    if status not in {"pass", "fail", "warn"}:
        status = "warn"
    confidence = _confidence(item.get("confidence"))
    return {
        "name": str(item["name"])[:80], "status": status, "confidence": round(confidence, 2),
        "detected": status == "pass", "detector": "groq-vision",
        "description": str(item.get("description", "Visual review required."))[:280],
    }


def _serial_result(value: Any, denomination: str) -> dict[str, Any]:
    serial = str(value).upper().replace(" ", "") if value else None
    if serial and not re.fullmatch(r"[A-Z0-9]{4,12}", serial):
        serial = None
    prefix = serial[:4] if serial else ""
    specimen = bool(serial and _SPECIMEN_RE.match(serial))
    return {
        "extracted": serial, "format_valid": bool(serial and _SERIAL_RE.match(serial) and not specimen),
        "is_known_counterfeit_prefix": prefix in _KNOWN_COUNTERFEIT_PREFIXES,
        "is_specimen_pattern": specimen, "denomination_match": True,
        "ocr_detected": bool(serial),
    }


def _persist(result: dict[str, Any]) -> None:
    scan_id = result["scan_id"]
    _scan_store[scan_id] = result
    _scan_order.append(scan_id)
    if len(_scan_order) > 200:
        expired = _scan_order.pop(0)
        _scan_store.pop(expired, None)


def model_status() -> dict[str, Any]:
    return {
        "ready": bool(settings.groq_api_key), "provider": "Groq Vision",
        "model": settings.groq_vision_model,
        "limitations": ["Visual screening only; human or bank verification is required before any action."],
    }


def scan_currency_image(image_bytes: bytes, denomination_hint: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    image_url = _image_data_url(image_bytes)
    hint = denomination_hint if denomination_hint in _SUPPORTED_DENOMS else "none"
    completion = _get_client().chat.completions.create(
        model=settings.groq_vision_model,
        temperature=0.1,
        max_tokens=min(settings.groq_max_tokens, 1200),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Optional user denomination hint: {hint}. Analyse this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    review = _json_object(raw)
    if not bool(review.get("is_indian_banknote", False)):
        raise ValueError("The uploaded image does not appear to be a clear single Indian banknote. Please upload the front of one note in good light.")
    denomination = str(review.get("denomination", "unknown"))
    if denomination not in _SUPPORTED_DENOMS:
        denomination = denomination_hint if denomination_hint in _SUPPORTED_DENOMS else "unknown"
    verdict = str(review.get("verdict", "SUSPICIOUS")).upper()
    if verdict not in {"AUTHENTIC", "SUSPICIOUS", "COUNTERFEIT"}:
        verdict = "SUSPICIOUS"
    confidence = _confidence(review.get("confidence"))
    serial = _serial_result(review.get("serial_number"), denomination)
    if serial["is_specimen_pattern"] or serial["is_known_counterfeit_prefix"]:
        verdict = "COUNTERFEIT"
        confidence = max(confidence, 0.9)
    features = [_normalise_feature(item) for item in review.get("features", [])]
    features = [item for item in features if item is not None][:8]
    if not features:
        features = [{"name": "Visual review", "status": "warn", "confidence": confidence,
                     "detected": False, "detector": "groq-vision",
                     "description": "No individual feature could be reliably verified from this image."}]
    result = {
        "scan_id": str(uuid.uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict, "confidence": round(confidence, 2),
        "overall_score": round((1 - confidence if verdict == "COUNTERFEIT" else confidence) * 100, 1),
        "denomination": denomination,
        "denomination_confidence": _confidence(review.get("denomination_confidence"), default=0.0),
        "features": features, "feature_details": features, "serial_number": serial,
        "detection_reason": str(review.get("detection_reason", "Groq visual review completed."))[:600],
        "processing_time_ms": int((time.perf_counter() - started) * 1000),
        "pipeline_version": "NETRA-GROQ-VISION-1", "banknote_score": 100,
        "image_quality": None, "requires_manual_security_feature_review": True,
        "disclaimer": "This is an AI visual screening result, not proof of authenticity. Verify suspicious notes with a bank or authorised examiner.",
    }
    _persist(result)
    return result


def get_scan_by_id(scan_id: str) -> dict[str, Any] | None:
    return _scan_store.get(scan_id)


def check_serial_number(number: str) -> dict[str, Any]:
    serial = number.strip().upper().replace(" ", "")
    prefix = serial[:4]
    specimen = bool(_SPECIMEN_RE.match(serial))
    flagged = prefix in _KNOWN_COUNTERFEIT_PREFIXES or specimen
    return {"serial_number": serial, "serial": serial, "format_valid": bool(_SERIAL_RE.match(serial)) and not specimen,
            "is_known_counterfeit_prefix": prefix in _KNOWN_COUNTERFEIT_PREFIXES, "is_specimen_pattern": specimen,
            "is_flagged": flagged, "risk_level": "HIGH" if flagged else "LOW",
            "message": "Serial requires manual verification." if not flagged else "This serial matches a flagged demonstration pattern."}


def get_stats() -> dict[str, Any]:
    results = list(_scan_store.values())
    total = len(results)
    counterfeits = sum(item.get("verdict") == "COUNTERFEIT" for item in results)
    authentic = sum(item.get("verdict") == "AUTHENTIC" for item in results)
    return {"total_scans": total, "counterfeits": counterfeits, "authentic": authentic,
            "suspicious": total - counterfeits - authentic,
            "counterfeit_rate": round(counterfeits / total * 100, 2) if total else 0.0,
            "source": "memory", "provider": "Groq Vision"}


def get_scan_history(limit: int = 20) -> list[dict[str, Any]]:
    return [{"id": scan_id, "timestamp": _scan_store[scan_id].get("timestamp", ""),
             "verdict": _scan_store[scan_id].get("verdict"), "confidence": _scan_store[scan_id].get("confidence"),
             "denomination": _scan_store[scan_id].get("denomination")}
            for scan_id in reversed(_scan_order[-limit:]) if scan_id in _scan_store]
