"""CRUD operations for chat message history."""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import ChatMessage


async def append(
    session: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ChatMessage:
    """Append a message to a chat session."""
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        metadata_=metadata,
    )
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    return msg


async def get_history(
    session: AsyncSession,
    session_id: str,
    limit: int = 20,
) -> list[ChatMessage]:
    """Retrieve the most recent messages for a session, ordered oldest-first."""
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    messages = list(result.scalars().all())
    messages.reverse()  # oldest first
    return messages


async def clear(session: AsyncSession, session_id: str) -> int:
    """Delete all messages for a session. Returns count deleted."""
    stmt = delete(ChatMessage).where(ChatMessage.session_id == session_id)
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount
