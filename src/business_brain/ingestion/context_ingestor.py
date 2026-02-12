"""Ingest natural-language business context into the vector store."""

import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import BusinessContext
from business_brain.ingestion.embeddings import embed_text

logger = logging.getLogger(__name__)

# Sentence-ending patterns used to find split points
_SPLIT_RE = re.compile(r"(?<=\. )|(?<=\n\n)|(?<=\n)")


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
    for chunk in chunks:
        embedding = embed_text(chunk)
        entry = BusinessContext(content=chunk, embedding=embedding, source=source)
        session.add(entry)
        await session.commit()
        await session.refresh(entry)
        ids.append(entry.id)
        logger.info("Stored context id=%d source=%s chunk_len=%d", entry.id, source, len(chunk))

    return ids
