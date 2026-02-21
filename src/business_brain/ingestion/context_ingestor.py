"""Ingest natural-language business context into the vector store.

Improvements:
- Batch commit (one commit per document instead of per chunk)
- Deduplication: checks for existing content before inserting
- Source-aware chunking: preserves source metadata through chunks
- Overlap-based chunking on sentence boundaries
"""

import logging
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import BusinessContext
from business_brain.ingestion.embeddings import embed_text

logger = logging.getLogger(__name__)

# Sentence-ending patterns used to find split points
_SPLIT_RE = re.compile(
    r"(?<=\. )"        # period + space
    r"|(?<=\? )"       # question mark + space
    r"|(?<=! )"        # exclamation + space
    r"|(?<=\n\n)"      # double newline (paragraph break)
    r"|(?<=\n)(?=[-*])"  # newline before bullet (- or *)
    r"|(?<=\n)(?=\d+\.)" # newline before numbered list item
    r"|(?<=\n)"        # single newline
    r"|(?<=: )"        # colon + space (before content)
)


def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks on sentence boundaries.

    Args:
        text: The input text to chunk.
        max_chars: Target maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # Find all possible split positions (sentence boundaries)
    split_positions = [m.start() for m in _SPLIT_RE.finditer(text)]
    if not split_positions:
        # No sentence boundaries — fall back to hard splits
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end].strip())
            start = end - overlap if end < len(text) else end
        return [c for c in chunks if c]

    chunks = []
    start = 0

    while start < len(text):
        if len(text) - start <= max_chars:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Find the last split position within max_chars from start
        end = start + max_chars
        best_split = None
        for pos in split_positions:
            if pos <= start:
                continue
            if pos <= end:
                best_split = pos
            else:
                break

        if best_split is None:
            # No sentence boundary found within range — hard split
            best_split = end

        chunk = text[start:best_split].strip()
        if chunk:
            chunks.append(chunk)

        # Next chunk starts with overlap
        start = max(best_split - overlap, start + 1)

    return chunks


async def _content_exists(session: AsyncSession, content: str, source: str) -> bool:
    """Check if identical content from the same source already exists.

    Returns False on any error so ingestion is never blocked by dedup failures.
    """
    try:
        prefix = content[:200]
        stmt = (
            select(BusinessContext.id)
            .where(
                BusinessContext.source == source,
                BusinessContext.content.startswith(prefix),
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None
    except Exception:
        return False


async def ingest_context(
    text: str,
    session: AsyncSession,
    source: str = "manual",
) -> list[int]:
    """Embed *text* in chunks and store each as a BusinessContext row.

    Args:
        text: Natural-language business context.
        session: Async SQLAlchemy session.
        source: Label for where the context came from.

    Returns:
        List of IDs of the newly created rows.
    """
    chunks = chunk_text(text)
    if not chunks:
        return []

    ids: list[int] = []
    new_entries: list[BusinessContext] = []

    for chunk in chunks:
        # Deduplication: skip if identical content from same source exists
        try:
            if await _content_exists(session, chunk, source):
                logger.debug("Skipping duplicate context chunk (source=%s, len=%d)", source, len(chunk))
                continue
        except Exception:
            # If dedup check fails, proceed with insertion
            logger.debug("Dedup check failed, proceeding with insertion")

        embedding = embed_text(chunk)
        entry = BusinessContext(content=chunk, embedding=embedding, source=source)
        session.add(entry)
        new_entries.append(entry)

    # Batch commit — one commit for all chunks instead of per-chunk
    if new_entries:
        await session.commit()
        for entry in new_entries:
            await session.refresh(entry)
            ids.append(entry.id)
            logger.info("Stored context id=%d source=%s chunk_len=%d", entry.id, source, len(entry.content))

    return ids


async def delete_context_by_source(
    session: AsyncSession,
    source: str,
) -> int:
    """Delete all context entries from a specific source. Useful for re-ingestion.

    Returns count of entries deleted.
    """
    from sqlalchemy import delete

    stmt = delete(BusinessContext).where(BusinessContext.source == source)
    result = await session.execute(stmt)
    await session.commit()
    count = result.rowcount
    if count:
        logger.info("Deleted %d context entries for source=%s", count, source)
    return count


async def reingest_context(
    text: str,
    session: AsyncSession,
    source: str,
) -> list[int]:
    """Delete existing context for a source, then re-ingest fresh.

    Useful when onboarding data or company profile is updated.
    """
    await delete_context_by_source(session, source)
    return await ingest_context(text, session, source=source)
