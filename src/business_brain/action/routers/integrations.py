"""Integrations routes — Telegram, Duplicates, Source Mappings."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["integrations"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class TelegramRegisterRequest(BaseModel):
    chat_id: str


class ConfirmMappingRequest(BaseModel):
    table_a: str
    table_b: str
    column_mappings: list[dict]
    entity_type: str = ""
    authoritative_source: str = ""


# ---------------------------------------------------------------------------
# Telegram Routes
# ---------------------------------------------------------------------------


@router.post("/telegram/register")
async def register_telegram(req: TelegramRegisterRequest) -> dict:
    """Register a Telegram chat ID for receiving alerts."""
    return {"status": "registered", "chat_id": req.chat_id}


@router.get("/telegram/status")
async def telegram_status() -> dict:
    """Check Telegram bot connection status."""
    from business_brain.action.telegram_bot import get_bot_info
    try:
        info = await get_bot_info()
        return {"status": "connected", "bot_username": info.get("username"), "bot_name": info.get("first_name")}
    except Exception as exc:
        return {"status": "disconnected", "error": str(exc)}


# ---------------------------------------------------------------------------
# Format Detection & Source Mappings
# ---------------------------------------------------------------------------


@router.get("/duplicates")
async def detect_duplicates(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Detect potential duplicate data sources."""
    from business_brain.discovery.format_detector import detect_duplicate_sources
    return await detect_duplicate_sources(session)


@router.post("/source-mappings")
async def confirm_mapping_endpoint(
    req: ConfirmMappingRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Confirm a source mapping between two tables."""
    from business_brain.discovery.format_detector import confirm_source_mapping
    mapping = await confirm_source_mapping(
        session, req.table_a, req.table_b, req.column_mappings,
        req.entity_type, req.authoritative_source,
    )
    return {"status": "confirmed", "id": mapping.id}


@router.get("/source-mappings")
async def list_mappings(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all confirmed source mappings."""
    from business_brain.discovery.format_detector import get_source_mappings
    mappings = await get_source_mappings(session)
    return [
        {
            "id": m.id,
            "source_a_table": m.source_a_table,
            "source_b_table": m.source_b_table,
            "column_mappings": m.column_mappings,
            "entity_type": m.entity_type,
            "authoritative_source": m.authoritative_source,
            "confirmed_by_user": m.confirmed_by_user,
        }
        for m in mappings
    ]
