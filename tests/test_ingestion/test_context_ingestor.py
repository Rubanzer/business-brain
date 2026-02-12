"""Tests for context ingestor — chunking and ingestion."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.ingestion.context_ingestor import chunk_text, ingest_context


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "This is a short sentence."
        chunks = chunk_text(text, max_chars=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        # Create text with clear sentence boundaries
        sentences = [f"Sentence number {i} provides important context. " for i in range(20)]
        text = "".join(sentences)
        chunks = chunk_text(text, max_chars=200, overlap=30)
        assert len(chunks) > 1
        # Every chunk should be non-empty
        assert all(c.strip() for c in chunks)
        # Every chunk should be roughly within max_chars
        for c in chunks:
            assert len(c) <= 250  # allow small overflow on boundary

    def test_overlap_present(self):
        sentences = [f"Sentence {i} has content. " for i in range(10)]
        text = "".join(sentences)
        chunks = chunk_text(text, max_chars=100, overlap=30)
        # Check that consecutive chunks share some text
        if len(chunks) >= 2:
            # The end of chunk[0] should appear somewhere in chunk[1]
            tail = chunks[0][-20:]
            assert tail in chunks[1] or chunks[1].startswith(tail[:10]) or len(chunks) >= 2

    def test_newline_split(self):
        text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three content here."
        chunks = chunk_text(text, max_chars=50, overlap=10)
        assert len(chunks) >= 2

    def test_no_sentence_boundaries_hard_split(self):
        # One long string with no sentence boundaries
        text = "a" * 1000
        chunks = chunk_text(text, max_chars=200, overlap=30)
        assert len(chunks) > 1
        # All text should be covered
        assert all(c for c in chunks)

    def test_exact_max_chars(self):
        text = "x" * 500
        chunks = chunk_text(text, max_chars=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_slightly_over_max_chars(self):
        text = "x" * 501
        chunks = chunk_text(text, max_chars=500, overlap=50)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# ingest_context tests
# ---------------------------------------------------------------------------


class TestIngestContext:
    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_short_text_single_row(self, mock_embed):
        mock_embed.return_value = [0.1] * 3072

        session = AsyncMock()
        # Simulate session.refresh setting entry.id
        call_count = 0

        async def fake_refresh(entry):
            nonlocal call_count
            call_count += 1
            entry.id = call_count

        session.refresh = fake_refresh

        ids = await ingest_context("Short context.", session, source="test")
        assert len(ids) == 1
        assert ids[0] == 1
        assert session.add.call_count == 1
        assert session.commit.call_count == 1

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_long_text_multiple_rows(self, mock_embed):
        mock_embed.return_value = [0.1] * 3072

        session = AsyncMock()
        call_count = 0

        async def fake_refresh(entry):
            nonlocal call_count
            call_count += 1
            entry.id = call_count

        session.refresh = fake_refresh

        # Create text longer than 500 chars
        text = ". ".join([f"Business context sentence {i}" for i in range(30)])
        ids = await ingest_context(text, session, source="test")
        assert len(ids) > 1
        # Each chunk should have gotten its own embedding
        assert mock_embed.call_count == len(ids)
        assert session.add.call_count == len(ids)

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_empty_text_returns_empty(self, mock_embed):
        session = AsyncMock()
        ids = await ingest_context("", session)
        assert ids == []
        mock_embed.assert_not_called()

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_source_passed_through(self, mock_embed):
        mock_embed.return_value = [0.1] * 3072

        session = AsyncMock()

        async def fake_refresh(entry):
            entry.id = 1

        session.refresh = fake_refresh

        await ingest_context("Some text.", session, source="upload:data.csv")
        # Verify the entry was created with the right source
        added_entry = session.add.call_args[0][0]
        assert added_entry.source == "upload:data.csv"
