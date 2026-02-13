"""Tests for chat message store."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.memory.chat_store import append, clear, get_history


class TestChatStore:
    @pytest.mark.asyncio
    async def test_append_message(self):
        session = AsyncMock()

        async def fake_refresh(entry):
            entry.id = 1

        session.refresh = fake_refresh

        msg = await append(session, "sess-1", "user", "Hello", {"key": "val"})
        assert msg.session_id == "sess-1"
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.metadata_ == {"key": "val"}
        session.add.assert_called_once()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_history(self):
        session = AsyncMock()

        msg1 = MagicMock()
        msg1.role = "user"
        msg1.content = "Q1"
        msg1.created_at = 1

        msg2 = MagicMock()
        msg2.role = "assistant"
        msg2.content = "A1"
        msg2.created_at = 2

        # Simulate query returning messages in desc order
        result = MagicMock()
        result.scalars.return_value.all.return_value = [msg2, msg1]
        session.execute = AsyncMock(return_value=result)

        messages = await get_history(session, "sess-1", limit=10)
        # Should be reversed to oldest first
        assert messages[0].content == "Q1"
        assert messages[1].content == "A1"

    @pytest.mark.asyncio
    async def test_clear(self):
        session = AsyncMock()
        result = MagicMock()
        result.rowcount = 5
        session.execute = AsyncMock(return_value=result)

        count = await clear(session, "sess-1")
        assert count == 5
        session.commit.assert_called_once()
