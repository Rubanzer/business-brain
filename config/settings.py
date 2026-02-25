"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Database
    database_url: str = "postgresql+psycopg://brain:brain@localhost:5432/business_brain"

    # LLM — Fast Tier (Gemini)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    gemini_model: str = "gemini-2.0-flash"

    # LLM — Deep Tier (Claude)
    claude_model: str = "claude-sonnet-4-20250514"
    deep_tier_auto_threshold: float = 0.3  # auto-escalate when confidence < this

    # Watcher
    watch_directory: str = "data/incoming"

    # Alerts
    slack_webhook_url: str = ""
    whatsapp_webhook_url: str = ""

    # Google Sheets
    google_service_account_json: str = ""

    # Telegram
    telegram_bot_token: str = ""


settings = Settings()
