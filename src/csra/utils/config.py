"""Centralized application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_env: str = "development"
    app_name: str = "CSRA"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    # LLM
    llm_provider: str = "openai"
    llm_model_name: str = "gpt-4o-mini"
    llm_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
