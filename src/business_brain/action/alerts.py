"""Notification dispatch to Slack / WhatsApp via webhooks."""

import httpx

from config.settings import settings


async def send_slack(message: str) -> bool:
    """Post a message to the configured Slack webhook."""
    if not settings.slack_webhook_url:
        print("[alerts] Slack webhook not configured, skipping.")
        return False
    async with httpx.AsyncClient() as client:
        resp = await client.post(settings.slack_webhook_url, json={"text": message})
        return resp.is_success


async def send_whatsapp(message: str) -> bool:
    """Post a message to the configured WhatsApp webhook."""
    if not settings.whatsapp_webhook_url:
        print("[alerts] WhatsApp webhook not configured, skipping.")
        return False
    async with httpx.AsyncClient() as client:
        resp = await client.post(settings.whatsapp_webhook_url, json={"body": message})
        return resp.is_success
