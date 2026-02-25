"""Business Context Management routes."""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["context"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class ContextRequest(BaseModel):
    text: str
    source: str = "api"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/context")
async def submit_context(req: ContextRequest, session: AsyncSession = Depends(get_session)) -> dict:
    """Submit natural-language business context for embedding."""
    from business_brain.ingestion.context_ingestor import ingest_context
    ids = await ingest_context(req.text, session, source=req.source)
    return {"status": "created", "ids": ids, "chunks": len(ids), "source": req.source}


@router.get("/context/entries")
async def list_context_entries(session: AsyncSession = Depends(get_session)) -> dict:
    """List all business context entries grouped by source."""
    from business_brain.memory.vector_store import list_all_contexts
    entries = await list_all_contexts(session, active_only=True)

    grouped: dict[str, list] = {}
    for entry in entries:
        source = entry["source"]
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(entry)

    return {"sources": grouped, "total": len(entries)}


@router.put("/context/entries/{entry_id}")
async def update_context_entry(
    entry_id: int,
    body: dict,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Edit a single context entry's text. Re-embeds the content."""
    from sqlalchemy import select
    from business_brain.db.models import BusinessContext
    from business_brain.ingestion.embeddings import embed_text

    result = await session.execute(
        select(BusinessContext).where(BusinessContext.id == entry_id)
    )
    entry = result.scalar_one_or_none()
    if not entry:
        return {"error": "Context entry not found"}

    new_content = body.get("content", "").strip()
    if not new_content:
        return {"error": "Content cannot be empty"}

    entry.content = new_content
    entry.embedding = embed_text(new_content)
    await session.commit()

    return {"status": "updated", "id": entry_id}


@router.delete("/context/entries/{entry_id}")
async def delete_context_entry(
    entry_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Soft-delete a context entry (mark as inactive)."""
    from datetime import datetime, timezone
    from sqlalchemy import select
    from business_brain.db.models import BusinessContext

    result = await session.execute(
        select(BusinessContext).where(BusinessContext.id == entry_id)
    )
    entry = result.scalar_one_or_none()
    if not entry:
        return {"error": "Context entry not found"}

    entry.active = False
    entry.superseded_at = datetime.now(timezone.utc)
    await session.commit()

    return {"status": "deleted", "id": entry_id}
