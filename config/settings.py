"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Database
    database_url: str = "postgresql+psycopg://brain:brain@localhost:5432/business_brain"

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    embedding_model: str = "text-embedding-004"
    gemini_model: str = "gemini-2.0-flash"

    # Watcher
    watch_directory: str = "data/incoming"

    # Alerts
    slack_webhook_url: str = ""
    whatsapp_webhook_url: str = ""


settings = Settings()
