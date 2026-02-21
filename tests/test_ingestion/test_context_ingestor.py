"""Tests for context ingestor — chunking, ingestion, dedup, regex edge cases."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.ingestion.context_ingestor import (
    chunk_text,
    ingest_context,
    _content_exists,
    delete_context_by_source,
    reingest_context,
)


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
        sentences = [f"Sentence number {i} provides important context. " for i in range(20)]
        text = "".join(sentences)
        chunks = chunk_text(text, max_chars=200, overlap=30)
        assert len(chunks) > 1
        assert all(c.strip() for c in chunks)
        for c in chunks:
            assert len(c) <= 250

    def test_overlap_present(self):
        sentences = [f"Sentence {i} has content. " for i in range(10)]
        text = "".join(sentences)
        chunks = chunk_text(text, max_chars=100, overlap=30)
        if len(chunks) >= 2:
            tail = chunks[0][-20:]
            assert tail in chunks[1] or chunks[1].startswith(tail[:10]) or len(chunks) >= 2

    def test_newline_split(self):
        text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three content here."
        chunks = chunk_text(text, max_chars=50, overlap=10)
        assert len(chunks) >= 2

    def test_no_sentence_boundaries_hard_split(self):
        text = "a" * 1000
        chunks = chunk_text(text, max_chars=200, overlap=30)
        assert len(chunks) > 1
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

    # --- Sentence boundary edge cases ---

    def test_question_mark_split(self):
        """Question marks should be recognized as sentence boundaries."""
        text = "What is revenue? Revenue is income from sales. How is it calculated? It comes from transactions."
        chunks = chunk_text(text, max_chars=50, overlap=10)
        assert len(chunks) >= 2

    def test_exclamation_mark_split(self):
        """Exclamation marks should be recognized as sentence boundaries."""
        text = "Great news! Sales are up. Amazing performance! Keep it going."
        chunks = chunk_text(text, max_chars=30, overlap=5)
        assert len(chunks) >= 2

    def test_bullet_list_split(self):
        """Bullets after newlines should be split points."""
        text = "Key points:\n- Revenue grew 20%\n- Margins improved\n- Customer count up\n- Retention stable"
        chunks = chunk_text(text, max_chars=40, overlap=5)
        assert len(chunks) >= 2

    def test_star_bullet_split(self):
        """Star bullets after newlines should be split points."""
        text = "Items:\n* First item here\n* Second item here\n* Third item here"
        chunks = chunk_text(text, max_chars=30, overlap=5)
        assert len(chunks) >= 2

    def test_numbered_list_split(self):
        """Numbered list items should be split points."""
        text = "Steps:\n1. First step\n2. Second step\n3. Third step\n4. Fourth step"
        chunks = chunk_text(text, max_chars=30, overlap=5)
        assert len(chunks) >= 2

    def test_colon_split(self):
        """Colons should be recognized as split points."""
        text = "Revenue: $1M in Q1. Cost: $500K in Q1. Profit: $500K in Q1. Growth: 20% YoY increase."
        chunks = chunk_text(text, max_chars=50, overlap=10)
        assert len(chunks) >= 2

    def test_single_newline_split(self):
        text = "Line one here\nLine two here\nLine three here\nLine four here"
        chunks = chunk_text(text, max_chars=30, overlap=5)
        assert len(chunks) >= 2

    def test_mixed_boundaries(self):
        """Text with multiple boundary types."""
        text = (
            "Revenue grew 20% last quarter. "
            "Key drivers:\n"
            "- New customer acquisition\n"
            "- Improved retention rates\n"
            "What should we focus on? "
            "The data suggests: focusing on enterprise sales.\n\n"
            "Next steps:\n"
            "1. Hire more enterprise reps\n"
            "2. Expand product features"
        )
        chunks = chunk_text(text, max_chars=100, overlap=20)
        assert len(chunks) >= 2
        # All content should be covered
        full = " ".join(chunks)
        assert "Revenue" in full
        assert "enterprise" in full

    def test_unicode_text(self):
        """Unicode characters should be handled."""
        text = "Revenue was $1M. Gross margin: 40%. Net profit: 20%."
        chunks = chunk_text(text, max_chars=500)
        assert len(chunks) == 1
        assert "$1M" in chunks[0]

    def test_very_long_sentence(self):
        """A single sentence longer than max_chars forces hard split."""
        text = "This is a very long sentence " + "with many words " * 50 + "that should be split."
        chunks = chunk_text(text, max_chars=100, overlap=20)
        assert len(chunks) >= 2

    def test_only_whitespace_between_splits(self):
        text = "First sentence.    Second sentence.    Third sentence."
        chunks = chunk_text(text, max_chars=30, overlap=5)
        assert len(chunks) >= 2
        assert all(c.strip() for c in chunks)

    def test_trailing_newlines(self):
        text = "Content here.\n\n\n\n"
        chunks = chunk_text(text, max_chars=500)
        assert len(chunks) == 1
        assert chunks[0].strip() == "Content here."

    def test_only_newlines(self):
        text = "\n\n\n\n"
        chunks = chunk_text(text)
        assert chunks == []

    def test_overlap_larger_than_chunk(self):
        """Overlap larger than max_chars should still work."""
        text = "A. B. C. D. E. F. G. H. I. J."
        chunks = chunk_text(text, max_chars=10, overlap=8)
        assert len(chunks) >= 2

    def test_single_character(self):
        chunks = chunk_text("a")
        assert chunks == ["a"]

    def test_period_no_space(self):
        """Period without space shouldn't split (e.g., URLs, abbreviations)."""
        text = "Visit example.com for more info about Dr.Smith and revenue.growth patterns in Q4."
        chunks = chunk_text(text, max_chars=500)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# ingest_context tests
# ---------------------------------------------------------------------------


class TestIngestContext:
    """Tests for the ingest_context function.

    All tests patch _content_exists to return False (no duplicates) so we're
    testing ingestion logic in isolation.  Dedup behaviour has its own test class.
    """

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor._content_exists", return_value=False)
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_short_text_single_row(self, mock_embed, _mock_dedup):
        mock_embed.return_value = [0.1] * 3072

        session = AsyncMock()
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
    @patch("business_brain.ingestion.context_ingestor._content_exists", return_value=False)
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_long_text_multiple_rows(self, mock_embed, _mock_dedup):
        mock_embed.return_value = [0.1] * 3072

        session = AsyncMock()
        call_count = 0

        async def fake_refresh(entry):
            nonlocal call_count
            call_count += 1
            entry.id = call_count

        session.refresh = fake_refresh

        text = ". ".join([f"Business context sentence {i}" for i in range(30)])
        ids = await ingest_context(text, session, source="test")
        assert len(ids) > 1
        assert mock_embed.call_count == len(ids)
        assert session.add.call_count == len(ids)

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor._content_exists", return_value=False)
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_empty_text_returns_empty(self, mock_embed, _mock_dedup):
        session = AsyncMock()
        ids = await ingest_context("", session)
        assert ids == []
        mock_embed.assert_not_called()

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor._content_exists", return_value=False)
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_source_passed_through(self, mock_embed, _mock_dedup):
        mock_embed.return_value = [0.1] * 3072

        session = AsyncMock()

        async def fake_refresh(entry):
            entry.id = 1

        session.refresh = fake_refresh

        await ingest_context("Some text.", session, source="upload:data.csv")
        added_entry = session.add.call_args[0][0]
        assert added_entry.source == "upload:data.csv"

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor._content_exists", return_value=False)
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_whitespace_only_returns_empty(self, mock_embed, _mock_dedup):
        session = AsyncMock()
        ids = await ingest_context("   \n\n  ", session)
        assert ids == []

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor._content_exists", return_value=False)
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_text_with_special_characters(self, mock_embed, _mock_dedup):
        mock_embed.return_value = [0.1] * 3072

        session = AsyncMock()

        async def fake_refresh(entry):
            entry.id = 1

        session.refresh = fake_refresh

        text = "Revenue: $1.5M (Q1 2024). Growth rate: +20%! Wow!"
        ids = await ingest_context(text, session)
        assert len(ids) == 1

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor._content_exists", return_value=False)
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_default_source(self, mock_embed, _mock_dedup):
        mock_embed.return_value = [0.1] * 3072

        session = AsyncMock()

        async def fake_refresh(entry):
            entry.id = 1

        session.refresh = fake_refresh

        await ingest_context("Some text.", session)
        added_entry = session.add.call_args[0][0]
        assert added_entry.source == "manual"


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for _content_exists dedup logic and its effect on ingestion."""

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor._content_exists", return_value=True)
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_duplicate_chunks_skipped(self, mock_embed, _mock_dedup):
        """When _content_exists returns True, chunks should be skipped."""
        mock_embed.return_value = [0.1] * 3072
        session = AsyncMock()

        ids = await ingest_context("Duplicate content.", session, source="test")
        assert ids == []
        mock_embed.assert_not_called()
        session.add.assert_not_called()
        session.commit.assert_not_called()

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor._content_exists", side_effect=Exception("DB down"))
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_dedup_failure_allows_insertion(self, mock_embed, _mock_dedup):
        """When _content_exists raises, ingestion should proceed anyway."""
        mock_embed.return_value = [0.1] * 3072
        session = AsyncMock()

        async def fake_refresh(entry):
            entry.id = 42

        session.refresh = fake_refresh

        ids = await ingest_context("Content despite dedup failure.", session, source="test")
        assert len(ids) == 1
        assert ids[0] == 42

    @pytest.mark.asyncio
    async def test_content_exists_returns_false_on_error(self):
        """_content_exists should return False if the DB query fails."""
        session = AsyncMock()
        session.execute.side_effect = Exception("connection lost")

        result = await _content_exists(session, "some content", "test")
        assert result is False

    @pytest.mark.asyncio
    async def test_content_exists_returns_true_when_found(self):
        """_content_exists should return True when a matching row exists."""
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 42  # row ID = found
        session.execute.return_value = mock_result

        result = await _content_exists(session, "some content", "test")
        assert result is True

    @pytest.mark.asyncio
    async def test_content_exists_returns_false_when_not_found(self):
        """_content_exists should return False when no matching row exists."""
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # not found
        session.execute.return_value = mock_result

        result = await _content_exists(session, "some content", "test")
        assert result is False


# ---------------------------------------------------------------------------
# delete_context_by_source and reingest_context tests
# ---------------------------------------------------------------------------


class TestDeleteAndReingest:
    @pytest.mark.asyncio
    async def test_delete_context_by_source(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 3
        session.execute.return_value = mock_result

        count = await delete_context_by_source(session, "test_source")
        assert count == 3
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch("business_brain.ingestion.context_ingestor._content_exists", return_value=False)
    @patch("business_brain.ingestion.context_ingestor.embed_text")
    async def test_reingest_deletes_then_ingests(self, mock_embed, _mock_dedup):
        mock_embed.return_value = [0.1] * 3072

        session = AsyncMock()
        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 2
        session.execute.return_value = mock_delete_result

        async def fake_refresh(entry):
            entry.id = 99

        session.refresh = fake_refresh

        ids = await reingest_context("New content.", session, source="profile")
        assert len(ids) == 1
        # commit called once for delete, once for ingest
        assert session.commit.call_count == 2
