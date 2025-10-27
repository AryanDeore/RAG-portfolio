"""Minimal Pydantic Settings v2 for RAG Portfolio."""

from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Where to read from (in priority order inside Pydantic):
    # 1) init kwargs, 2) OS env vars, 3) .env, 4) files in secrets_dir
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        secrets_dir="/var/run/secrets/app",  # <- mount k8s/Docker secrets here
    )

    # --- App basics ---
    app_name: str = "RAG Portfolio"
    debug: bool = False
    log_level: str = "INFO"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Secrets (override via env/.env or mounted files) ---
    llm_api_key: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    database_url: Optional[str] = None

    # --- Chunker knobs (non-secret, but configurable) ---
    chunk_max_chars_paragraph: int = 700
    chunk_split_long_bullets: bool = False
    chunk_max_chars_bullet: int = 700


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Optional: convenient singleton if you prefer `from configs.settings import settings`
settings = get_settings()
