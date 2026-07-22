"""NETRA — counterfeit currency detection endpoints.

All routes are mounted under the /api/v1 prefix by main.py, so the full
paths are:

  POST   /api/v1/netra/scan
  POST   /api/v1/netra/scan/batch
  GET    /api/v1/netra/scan/{scan_id}
  GET    /api/v1/netra/serial/{number}
  GET    /api/v1/netra/stats
  POST   /api/v1/netra/report
  GET    /api/v1/netra/history
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.models.schemas import NetraModelEvaluateRequest, NetraModelTrainRequest, NetraReportRequest, fail, ok
from app.services import jaal_service, netra_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/netra", tags=["netra"])


@router.get("/model/status")
def model_status():
    """Return the hosted Groq Vision readiness state."""
    return ok(netra_service.model_status())


@router.post("/model/train")
def train_model(payload: NetraModelTrainRequest):
    """Local model training was removed to keep the Render service lightweight."""
    return fail("NETRA now uses Groq Vision. Local model training is not available in this deployment.")


@router.post("/model/evaluate")
def evaluate_model(payload: NetraModelEvaluateRequest):
    """Local model evaluation was removed with the local model runtime."""
    return fail("NETRA now uses Groq Vision. Local model evaluation is not available in this deployment.")


# ── POST /scan ────────────────────────────────────────────────────────────────


@router.post("/scan")
async def scan_currency(
    file: UploadFile = File(..., description="Currency note image (JPEG/PNG/BMP/WEBP)"),
    denomination: Optional[str] = Query(
        default=None,
        description="Optional denomination hint: ₹500 | ₹200 | ₹100",
    ),
):
    """
    Scan a single currency note image for authenticity.

    Runs the full NETRA multi-stage pipeline (preprocessing → denomination
    detection → YOLOv12 feature analysis → serial extraction → holistic
    scoring) and returns a detailed verdict with per-feature results.
    """
    image_bytes = await file.read()
    if not image_bytes:
        return fail("Uploaded file is empty")

    try:
        # The CV/OCR pipeline is CPU-bound.  Keep it off the ASGI event loop so
        # stats, history, and health requests do not stall during a scan.
        result = await run_in_threadpool(
            netra_service.scan_currency_image, image_bytes, denomination
        )
    except ValueError as exc:
        return fail(str(exc))
    except Exception:
        logger.exception("Unexpected error during NETRA scan (file=%s)", file.filename)
        return fail("Internal scan error — please try again with a valid image")

    if result.get("verdict") == "COUNTERFEIT":
        serial = (result.get("serial_number") or {}).get("extracted") or result.get("scan_id", "unknown-note")
        jaal_service.ingest_module_signal(
            "NETRA", f"Currency serial {serial}",
            "NETRA detected a counterfeit currency note requiring network correlation.",
            entity_type="website", risk_score=float(result.get("confidence", 0.8)),
        )
    return ok(result)


# ── POST /scan/batch ──────────────────────────────────────────────────────────


@router.post("/scan/batch")
async def scan_batch(
    files: list[UploadFile] = File(..., description="One or more currency note images"),
    denomination: Optional[str] = Query(
        default=None,
        description="Optional denomination hint applied to all images",
    ),
):
    """
    Scan multiple currency note images in a single request.

    Returns a list of per-file result objects.  Each item includes either
    ``result`` (on success) or ``error`` (on per-file failure) so a single
    bad image does not fail the whole batch.
    """
    if not files:
        return fail("No files supplied in the batch request")

    results = []
    for upload in files:
        try:
            image_bytes = await upload.read()
            if not image_bytes:
                results.append(
                    {"filename": upload.filename, "result": None, "error": "Empty file"}
                )
                continue
            scan_result = netra_service.scan_currency_image(image_bytes, denomination)
            results.append(
                {"filename": upload.filename, "result": scan_result, "error": None}
            )
        except ValueError as exc:
            results.append(
                {"filename": upload.filename, "result": None, "error": str(exc)}
            )
        except Exception as exc:
            logger.warning("Batch scan failed for %s: %s", upload.filename, exc)
            results.append(
                {
                    "filename": upload.filename,
                    "result": None,
                    "error": "Internal scan error for this file",
                }
            )

    successful = sum(1 for r in results if r["result"] is not None)
    return ok(
        results,
        meta={
            "total": len(results),
            "successful": successful,
            "failed": len(results) - successful,
        },
    )


# ── GET /scan/{scan_id} ───────────────────────────────────────────────────────


@router.get("/scan/{scan_id}")
def get_scan(scan_id: str):
    """
    Retrieve a previous scan result by its UUID.

    Returns the full NETRA result dict stored when the scan was originally
    performed.  Raises HTTP 404 if the scan ID is not found.
    """
    result = netra_service.get_scan_by_id(scan_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scan '{scan_id}' not found. IDs are valid only for scans "
            "performed in the current session (or stored in Supabase).",
        )
    return ok(result)


# ── GET /serial/{number} ──────────────────────────────────────────────────────


@router.get("/serial/{number}")
def check_serial(number: str):
    """
    Check a serial number against known counterfeit patterns.

    Validates the format (2-3 uppercase letters + 6-7 digits) and cross-
    references against the known FICN (Fake Indian Currency Note) prefix
    database.  Returns risk level ``HIGH | MEDIUM | LOW`` and a warning
    message when a counterfeit prefix is matched.
    """
    result = netra_service.check_serial_number(number)
    return ok(result)


# ── GET /stats ────────────────────────────────────────────────────────────────


@router.get("/stats")
def netra_stats():
    """
    Return aggregate detection statistics.

    Counts total scans, verdicts by category, and the overall counterfeit
    rate.  Data is sourced from Supabase when configured, otherwise from
    in-memory counters accumulated since server start.
    """
    return ok(netra_service.get_stats())


# ── POST /report ──────────────────────────────────────────────────────────────


@router.post("/report")
def report_counterfeit(body: NetraReportRequest):
    """
    Report a confirmed counterfeit note for further investigation.

    Requires a valid ``scan_id`` from a previous scan.  Optional fields
    include free-text notes and GPS coordinates to aid geographic analysis.
    """
    scan = netra_service.get_scan_by_id(body.scan_id)
    if scan is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scan '{body.scan_id}' not found — cannot file a report for an unknown scan.",
        )

    report = {
        "scan_id": body.scan_id,
        "notes": body.notes,
        "latitude": body.latitude,
        "longitude": body.longitude,
        "location_description": body.location_description,
        "denomination": scan.get("denomination"),
        "verdict": scan.get("verdict"),
        "confidence": scan.get("confidence"),
        "serial_number": (scan.get("serial_number") or {}).get("extracted"),
        "status": "submitted",
    }

    return ok({"message": "Report submitted successfully", "report": report})


# ── GET /history ──────────────────────────────────────────────────────────────


@router.get("/history")
def scan_history(
    limit: int = Query(
        default=20, ge=1, le=100, description="Number of records to return"
    ),
):
    """
    Return recent scan history (newest first).

    Each item includes the scan ID, timestamp, verdict, confidence, and
    detected denomination.  Full results are available via ``GET /scan/{id}``.
    """
    history = netra_service.get_scan_history(limit)
    return ok(history, meta={"count": len(history), "limit": limit})
