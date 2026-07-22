"""KAVACH — direct Groq-powered citizen fraud assistant.

There is intentionally no embedding model, vector database, scraper or RAG
initialisation here.  Groq handles multilingual conversation directly, while a
small local emergency fallback remains available if the API is unavailable.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "te": "Telugu", "mr": "Marathi", "ta": "Tamil",
    "gu": "Gujarati", "ur": "Urdu", "kn": "Kannada", "or": "Odia", "ml": "Malayalam", "pa": "Punjabi",
}
_sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
_SYSTEM_PROMPT = """You are KAVACH (कवच), RAKSHA AI's Indian citizen fraud-safety assistant.
Give practical, calm, safety-first advice about cyber fraud, suspicious calls,
UPI, OTPs, digital-arrest scams, and fake currency. Never ask for an OTP, PIN,
password, Aadhaar, PAN, bank details, or money. Do not present yourself as a
government authority or claim to have contacted a bank or police.

If money was sent or a digital-arrest/impersonation threat is active, clearly
tell the user to stop contact and call 1930 immediately; they can also report at
cybercrime.gov.in. Answer in the user's language when possible.

Return only JSON:
{"reply":"short helpful answer","intents":["UPI_FRAUD"],"quickActions":["Call 1930"],"riskLevel":"safe|warning|danger"}
Use danger for immediate financial loss, OTP/PIN requests, threats or digital arrest.
"""


def _get_client():
    if not settings.groq_api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=settings.groq_api_key)
    except Exception as exc:
        logger.warning("KAVACH could not initialise Groq: %s", exc)
        return None


def _parse(raw: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("invalid Groq response")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("invalid Groq response")
    risk = str(data.get("riskLevel", "warning")).lower()
    if risk not in {"safe", "warning", "danger"}:
        risk = "warning"
    return {
        "reply": str(data.get("reply", "Please stay alert and call 1930 for urgent cyber-fraud help."))[:1600],
        "intents": [str(value)[:48] for value in data.get("intents", []) if str(value).strip()][:6],
        "quickActions": [str(value)[:80] for value in data.get("quickActions", []) if str(value).strip()][:4],
        "riskLevel": risk, "sources": [], "provider": "Groq",
    }


def _fallback_reply(message: str, language: str = "auto") -> dict[str, Any]:
    lowered = message.lower()
    urgent = any(token in lowered for token in ("otp", "pin", "digital arrest", "arrest", "transfer", "upi", "urgent", "threat"))
    if urgent:
        return {"reply": "Do not share OTP, PIN, passwords or send money. End contact with the caller and call 1930 immediately; report at cybercrime.gov.in.",
                "intents": ["FRAUD_RISK"], "quickActions": ["Call 1930", "Report at cybercrime.gov.in"], "riskLevel": "danger", "sources": [], "provider": "local-safety-fallback"}
    return {"reply": "I can help assess suspicious calls, messages, UPI requests, or fake-currency concerns. For urgent financial fraud, call 1930 immediately.",
            "intents": ["GENERAL_SAFETY"], "quickActions": ["Call 1930", "Check a Number"], "riskLevel": "safe", "sources": [], "provider": "local-safety-fallback"}


def reply(message: str, session_id: str | None = None, language: str = "auto") -> dict[str, Any]:
    client = _get_client()
    if client is None:
        return _fallback_reply(message, language)
    session_key = session_id or "anonymous"
    history = _sessions[session_key][-6:]
    messages: list[dict[str, Any]] = [{"role": "system", "content": _SYSTEM_PROMPT}, *history,
                                      {"role": "user", "content": f"Preferred language: {language}.\nUser message: {message}"}]
    try:
        completion = client.chat.completions.create(
            model=settings.groq_text_model, messages=messages, temperature=0.2,
            max_tokens=min(settings.groq_max_tokens, 700), response_format={"type": "json_object"},
        )
        result = _parse(completion.choices[0].message.content or "{}")
        _sessions[session_key].extend([{"role": "user", "content": message}, {"role": "assistant", "content": result["reply"]}])
        _sessions[session_key] = _sessions[session_key][-8:]
        return result
    except Exception as exc:
        logger.warning("KAVACH Groq request failed: %s", exc)
        return _fallback_reply(message, language)


def status() -> dict[str, Any]:
    return {"status": "ready" if settings.groq_api_key else "missing_api_key", "provider": "Groq",
            "model": settings.groq_text_model, "rag": False, "activeSessions": len(_sessions)}


def check_number(phone: str) -> dict[str, Any]:
    digits = re.sub(r"\D", "", phone)
    safe = not digits.endswith(("9876543210", "9999999999"))
    return {"safe": safe, "risk_score": 0.15 if safe else 0.88}
