"""Tests for telegram bot helper functions (no network needed)."""

from unittest.mock import MagicMock, patch

import pytest

from business_brain.action.telegram_bot import (
    _api_url,
    generate_registration_code,
)


class TestRegistrationCode:
    """Test registration code generation."""

    def test_generates_string(self):
        code = generate_registration_code()
        assert isinstance(code, str)

    def test_correct_length(self):
        code = generate_registration_code()
        assert len(code) == 8

    def test_uppercase_hex(self):
        code = generate_registration_code()
        assert code == code.upper()
        assert all(c in "0123456789ABCDEF" for c in code)

    def test_unique_codes(self):
        codes = {generate_registration_code() for _ in range(100)}
        assert len(codes) > 90  # Extremely unlikely to have many collisions


class TestApiUrl:
    """Test Telegram API URL building."""

    @patch("business_brain.action.telegram_bot.settings")
    def test_builds_url_with_token(self, mock_settings):
        mock_settings.telegram_bot_token = "123:ABC"
        url = _api_url("sendMessage")
        assert url == "https://api.telegram.org/bot123:ABC/sendMessage"

    @patch("business_brain.action.telegram_bot.settings")
    def test_builds_url_for_getMe(self, mock_settings):
        mock_settings.telegram_bot_token = "TOKEN"
        url = _api_url("getMe")
        assert url == "https://api.telegram.org/botTOKEN/getMe"

    @patch("business_brain.action.telegram_bot.settings")
    def test_raises_without_token(self, mock_settings):
        mock_settings.telegram_bot_token = ""
        with pytest.raises(ValueError, match="not configured"):
            _api_url("sendMessage")

    @patch("business_brain.action.telegram_bot.settings")
    def test_raises_with_none_token(self, mock_settings):
        mock_settings.telegram_bot_token = None
        with pytest.raises(ValueError, match="not configured"):
            _api_url("sendMessage")

    @patch("business_brain.action.telegram_bot.settings")
    def test_getUpdates_url(self, mock_settings):
        mock_settings.telegram_bot_token = "MY_TOKEN"
        url = _api_url("getUpdates")
        assert "getUpdates" in url
        assert "MY_TOKEN" in url
