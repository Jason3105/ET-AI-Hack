"""KAVACH citizen fraud-shield endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import KavachChatRequest, KavachNumberCheck, fail, ok
from app.services import kavach_service

router = APIRouter(prefix="/kavach", tags=["kavach"])


@router.post("/chat")
async def chat(payload: KavachChatRequest):
    if not payload.message or not payload.message.strip():
        return fail("Message cannot be empty")
    return ok(kavach_service.reply(payload.message.strip(), payload.sessionId, payload.language))


@router.post("/check/number")
async def check_number(payload: KavachNumberCheck):
    if not payload.phone or not payload.phone.strip():
        return fail("Phone number cannot be empty")
    return ok(kavach_service.check_number(payload.phone))


@router.post("/ingest")
async def ingest_documents():
    return ok({"message": "KAVACH now uses direct Groq analysis; no knowledge-base ingestion is required.", "rag": False})


@router.get("/status")
async def kavach_status():
    return ok(kavach_service.status())
