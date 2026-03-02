"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Database
    database_url: str = "postgresql+psycopg://brain:brain@localhost:5432/business_brain"

    # LLM — Provider selection
    llm_provider: str = ""  # "gemini" | "openai" | "anthropic" | "" (auto-detect from keys)
    embedding_provider: str = ""  # "" = same as llm_provider

    # LLM — API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # LLM — Model names
    gemini_model: str = "gemini-2.0-flash"
    openai_model: str = "gpt-4o-mini"
    claude_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "gemini-embedding-001"

    # LLM — Deep Tier (Claude)
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
