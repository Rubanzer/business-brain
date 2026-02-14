"""Telegram bot integration — one-way alert delivery."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.telegram.org/bot{token}"


def _api_url(method: str) -> str:
    """Build the Telegram Bot API URL for a method."""
    if not settings.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN not configured")
    return f"{_BASE_URL.format(token=settings.telegram_bot_token)}/{method}"


async def send_alert(chat_id: str | int, message: str) -> dict[str, Any]:
    """Send an alert message to a Telegram chat or group.

    Args:
        chat_id: Telegram chat ID (user or group).
        message: The alert message text.

    Returns:
        Telegram API response dict.
    """
    url = _api_url("sendMessage")

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        })

    result = resp.json()
    if not result.get("ok"):
        logger.error("Telegram sendMessage failed: %s", result)
        raise ValueError(f"Telegram API error: {result.get('description', 'unknown')}")

    logger.info("Alert sent to Telegram chat %s", chat_id)
    return result


async def send_alert_formatted(
    chat_id: str | int,
    alert_name: str,
    current_value: str,
    threshold: str,
    source: str,
    web_url: str | None = None,
) -> dict[str, Any]:
    """Send a formatted alert message to Telegram.

    Args:
        chat_id: Telegram chat ID.
        alert_name: Name of the alert rule.
        current_value: The current value that triggered the alert.
        threshold: The threshold that was exceeded.
        source: The data source (table.column).
        web_url: Optional link to the web app for details.
    """
    from datetime import datetime, timezone

    time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M IST")

    lines = [
        f"🚨 *ALERT: {alert_name}*",
        "",
        f"Current: {current_value}",
        f"Threshold: {threshold}",
        f"Time: {time_str}",
        f"Source: {source}",
    ]

    if web_url:
        lines.append(f"\n[View details]({web_url})")

    message = "\n".join(lines)
    return await send_alert(chat_id, message)


async def get_bot_info() -> dict[str, Any]:
    """Get the bot's info (username, name) to verify the token works."""
    url = _api_url("getMe")

    async with httpx.AsyncClient() as client:
        resp = await client.get(url)

    result = resp.json()
    if not result.get("ok"):
        raise ValueError(f"Telegram API error: {result.get('description', 'unknown')}")

    return result.get("result", {})


async def get_updates(offset: int = 0) -> list[dict]:
    """Get recent updates (messages sent to the bot).

    Used for the registration flow: user sends /start to the bot,
    we get their chat_id from the updates.
    """
    url = _api_url("getUpdates")

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params={"offset": offset, "limit": 10})

    result = resp.json()
    if not result.get("ok"):
        raise ValueError(f"Telegram API error: {result.get('description', 'unknown')}")

    return result.get("result", [])


def generate_registration_code() -> str:
    """Generate a random registration code for linking Telegram to the web app."""
    import secrets
    return secrets.token_hex(4).upper()  # 8-char hex code
