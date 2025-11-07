"""
Centralized application configuration (Pydantic Settings v2).
Reads from init kwargs > environment > .env and exposes tunables used across the app,
including chunking, embedding/retrieval, and Qdrant connection details.
"""

from functools import lru_cache
from typing import Optional, Annotated, List
from pydantic import Field, AliasChoices, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

# Port type with numeric bounds for future use
Port = Annotated[int, Field(ge=1, le=65535)]


class Settings(BaseSettings):
    """
    Singleton settings object providing typed configuration for the whole project.

    Order of precedence: init kwargs > env vars > .env > (optional) secrets_dir.
    All fields have sane defaults; override via environment when deploying.
    """

    # ---- Pydantic Settings config ----
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # secrets_dir="/var/run/secrets/app",
    )

    # ====================== App / Server ======================
    debug: bool = Field(True, alias="DEBUG")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    server_host: str = Field("0.0.0.0", alias="SERVER_HOST")
    server_port: int = Field(8000, alias="SERVER_PORT")
    cors_allow_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"], description="Allowed CORS origins", alias="CORS_ALLOW_ORIGINS")

    # ====================== Qdrant ======================
    # For Qdrant Cloud: set QDRANT_URL to your cloud URL (e.g., https://xxx.cloud.qdrant.io)
    # For local Qdrant: leave QDRANT_URL unset and it will use QDRANT_HOST:QDRANT_PORT
    qdrant_url: Optional[AnyHttpUrl] = Field(None, alias="QDRANT_URL", description="Qdrant cluster URL (cloud or local)")
    qdrant_api_key: Optional[str] = Field(None, alias="QDRANT_API_KEY", description="Qdrant API key (required for cloud)")
    qdrant_timeout: int = Field(30, alias="QDRANT_TIMEOUT")
    qdrant_host: str = Field("http://127.0.0.1", alias="QDRANT_HOST", description="Local Qdrant host (fallback if QDRANT_URL not set)")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT", description="Local Qdrant port (fallback if QDRANT_URL not set)")

    # Collection names: writer and (optionally) a stable reader alias
    embed_collection: str = Field("rag_portfolio_v1", alias="EMBED_COLLECTION")
    embed_collection_read: str = Field("portfolio_current", alias="EMBED_COLLECTION_READ")
    
    # -------------------- Embedding --------------------------
    embed_model: str = Field("BAAI/bge-small-en-v1.5", alias="EMBED_MODEL")
    embed_dim: int = Field(384, alias="EMBED_DIM")
    embed_metric: str = Field("Cosine", alias="EMBED_METRIC")
    embed_batch_size: int = Field(128, alias="EMBED_BATCH_SIZE")
    embed_id_scheme: str = Field("uuid", alias="EMBED_ID_SCHEME")

    # -------------------- Retrieval --------------------------
    search_top_k: int = Field(5, alias="SEARCH_TOP_K")
    search_filter_project: Optional[str] = Field(None, alias="SEARCH_FILTER_PROJECT")  # e.g., "RAG Portfolio"
    rerank_top_n: int = Field(0, alias="RERANK_TOP_N")  # 0 disables reranking



    # ====================== Chunking ======================
    chunk_max_chars_paragraph: int = Field(default=700, description="Hard cap for paragraph-sized chunks")
    chunk_split_long_bullets: bool = Field(default=False, description="Split long bullets into child chunks")
    chunk_max_chars_bullet: int = Field(default=700, description="Hard cap for bullet-sized chunks")

    # ---- JSON catalogs for entity normalization (chunking.entities) ----
    tech_alias_path: str = Field(default="configs/tech_alias.json", description="Alias map for tech entities")
    tech_catalog_path: str = Field(default="configs/tech_catalog.json", description="Canonical tech catalog JSON")


    # ---- Convenience computed properties ----
    @property
    def qdrant_url_resolved(self) -> str:
        """
        Resolve the Qdrant URL: use QDRANT_URL if set, otherwise compose from host and port.

        Returns:
            str: e.g., "https://xxx.cloud.qdrant.io" or "http://localhost:6333"
        """
        # If QDRANT_URL is set (e.g., from environment), use it directly
        if self.qdrant_url:
            return str(self.qdrant_url)
        # Otherwise, compose from host and port (for local Qdrant)
        if ":" in self.qdrant_host.rsplit("/", 1)[-1]:
            return self.qdrant_host
        return f"{self.qdrant_host}:{self.qdrant_port}"


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """
    Construct and cache a single Settings instance for fast, consistent access.

    Returns:
        Settings: Cached, process-wide configuration object.
    """
    return Settings()


# Convenience import for code that prefers a singleton
settings = get_settings()
