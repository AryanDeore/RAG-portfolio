# A single source of truth for application configuration using Pydantic Settings v2.
# It reads from OS env vars, .env, and an optional secrets directory, and exposes tunables used by the chunking package.

from functools import lru_cache
from typing import Optional, Annotated
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

# Port type with numeric bounds for future use
Port = Annotated[int, Field(ge=1, le=65535)]

class Settings(BaseSettings):
    # Comment: Controls where settings are pulled from and in what order (init kwargs > env > .env > secrets_dir).
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # secrets_dir="/var/run/secrets/app",
    )

    # App basics
    app_name: str = "RAG Portfolio"
    debug: bool = False
    log_level: str = "INFO"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # Secrets / external
    llm_api_key: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    database_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL"),
        description="SQLAlchemy-style URL"
    )
    qdrant_port: Port = 6333

    # Chunker knobs (consumed directly by chunking.*)
    chunk_max_chars_paragraph: int = 700
    chunk_split_long_bullets: bool = False
    chunk_max_chars_bullet: int = 700

    # JSON catalogs for entity normalization and triggers (consumed by chunking.entities)
    tech_alias_path: str = Field(default="configs/tech_alias.json")
    tech_catalog_path: str = Field(default="configs/tech_catalog.json")


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    # Comment: Returns a cached singleton Settings instance so downstream imports are fast and consistent.
    return Settings()


# Convenience import for code that prefers a singleton
settings = get_settings()
