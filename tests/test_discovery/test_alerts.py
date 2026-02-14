"""Tests for the alerts module (Slack + WhatsApp webhook dispatch)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestSendSlack:
    """Test Slack webhook dispatch."""

    @pytest.mark.asyncio
    @patch("business_brain.action.alerts.settings")
    @patch("business_brain.action.alerts.httpx.AsyncClient")
    async def test_success(self, mock_client_cls, mock_settings):
        mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        from business_brain.action.alerts import send_slack
        result = await send_slack("Test message")
        assert result is True
        mock_client.post.assert_called_once_with(
            "https://hooks.slack.com/test",
            json={"text": "Test message"},
        )

    @pytest.mark.asyncio
    @patch("business_brain.action.alerts.settings")
    async def test_no_webhook_configured(self, mock_settings):
        mock_settings.slack_webhook_url = ""
        from business_brain.action.alerts import send_slack
        result = await send_slack("Test")
        assert result is False

    @pytest.mark.asyncio
    @patch("business_brain.action.alerts.settings")
    async def test_none_webhook(self, mock_settings):
        mock_settings.slack_webhook_url = None
        from business_brain.action.alerts import send_slack
        result = await send_slack("Test")
        assert result is False

    @pytest.mark.asyncio
    @patch("business_brain.action.alerts.settings")
    @patch("business_brain.action.alerts.httpx.AsyncClient")
    async def test_failed_response(self, mock_client_cls, mock_settings):
        mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
        mock_resp = MagicMock()
        mock_resp.is_success = False
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        from business_brain.action.alerts import send_slack
        result = await send_slack("Test")
        assert result is False


class TestSendWhatsApp:
    """Test WhatsApp webhook dispatch."""

    @pytest.mark.asyncio
    @patch("business_brain.action.alerts.settings")
    @patch("business_brain.action.alerts.httpx.AsyncClient")
    async def test_success(self, mock_client_cls, mock_settings):
        mock_settings.whatsapp_webhook_url = "https://wa.example.com/hook"
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        from business_brain.action.alerts import send_whatsapp
        result = await send_whatsapp("Hello from bot")
        assert result is True
        mock_client.post.assert_called_once_with(
            "https://wa.example.com/hook",
            json={"body": "Hello from bot"},
        )

    @pytest.mark.asyncio
    @patch("business_brain.action.alerts.settings")
    async def test_no_webhook_configured(self, mock_settings):
        mock_settings.whatsapp_webhook_url = ""
        from business_brain.action.alerts import send_whatsapp
        result = await send_whatsapp("Test")
        assert result is False

    @pytest.mark.asyncio
    @patch("business_brain.action.alerts.settings")
    async def test_none_webhook(self, mock_settings):
        mock_settings.whatsapp_webhook_url = None
        from business_brain.action.alerts import send_whatsapp
        result = await send_whatsapp("Test")
        assert result is False

    @pytest.mark.asyncio
    @patch("business_brain.action.alerts.settings")
    @patch("business_brain.action.alerts.httpx.AsyncClient")
    async def test_sends_body_key(self, mock_client_cls, mock_settings):
        """WhatsApp uses 'body' key, not 'text'."""
        mock_settings.whatsapp_webhook_url = "https://wa.example.com/hook"
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        from business_brain.action.alerts import send_whatsapp
        await send_whatsapp("Msg")
        call_kwargs = mock_client.post.call_args
        assert "body" in call_kwargs.kwargs["json"]
        assert "text" not in call_kwargs.kwargs["json"]
