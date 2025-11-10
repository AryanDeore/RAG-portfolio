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
    # Use your Cloud URL here; for local Docker you can set QDRANT_API_KEY=None
    qdrant_url: AnyHttpUrl = Field(..., alias="QDRANT_URL", description="Qdrant cluster URL")
    qdrant_api_key: Optional[str] = Field(None, alias="QDRANT_API_KEY", description="Qdrant API key")
    qdrant_timeout: int = Field(30, alias="QDRANT_TIMEOUT")
    qdrant_host: str = Field("http://127.0.0.1", alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")

    # Collection names: writer and (optionally) a stable reader alias
    embed_collection: str = Field("rag_portfolio_v1", alias="EMBED_COLLECTION")
    embed_collection_read: str = Field("portfolio_current", alias="EMBED_COLLECTION_READ")
    
    # -------------------- Embedding --------------------------
    embed_model: str = Field("BAAI/bge-small-en-v1.5", alias="EMBED_MODEL")
    embed_dim: int = Field(384, alias="EMBED_DIM")
    embed_metric: str = Field("Cosine", alias="EMBED_METRIC")
    embed_batch_size: int = Field(128, alias="EMBED_BATCH_SIZE")
    embed_id_scheme: str = Field("uuid", alias="EMBED_ID_SCHEME")
    embed_cache_dir: Optional[str] = Field(None, alias="EMBED_CACHE_DIR", description="Directory to cache embedding models. If None, uses default FastEmbed cache.")

    # -------------------- Retrieval --------------------------
    search_top_k: int = Field(5, alias="SEARCH_TOP_K")
    search_filter_project: Optional[str] = Field(None, alias="SEARCH_FILTER_PROJECT")  # e.g., "RAG Portfolio"
    rerank_top_n: int = Field(0, alias="RERANK_TOP_N")  # 0 disables reranking

    #====================== Generation ======================
    # -------------------- LLM Defaults------------------------
    llm_provider: str = Field("openai", alias="LLM_PROVIDER", description="Default LLM provider id for generation.")
    llm_model: str = Field("gpt-4.1-nano", alias="LLM_MODEL", description="Default LiteLLM model id for generation.")
    llm_temperature: float = Field(0.2, alias="LLM_TEMPERATURE", description="Default temperature for generation.")

    # ====================== Chunking ======================
    chunk_max_chars_paragraph: int = Field(default=700, description="Hard cap for paragraph-sized chunks")
    chunk_split_long_bullets: bool = Field(default=False, description="Split long bullets into child chunks")
    chunk_max_chars_bullet: int = Field(default=700, description="Hard cap for bullet-sized chunks")

    # ---- JSON catalogs for entity normalization (chunking.entities) ----
    tech_alias_path: str = Field(default="configs/tech_alias.json", description="Alias map for tech entities")
    tech_catalog_path: str = Field(default="configs/tech_catalog.json", description="Canonical tech catalog JSON")


    # ---- Convenience computed properties ----
    @property
    def qdrant_url(self) -> str:
        """
        Compose the full Qdrant base URL from host and port.

        Returns:
            str: e.g., "http://localhost:6333" or "https://xxx.cloud.qdrant.io:6333"
        """
        # If host already includes a port (common in cloud URLs), keep it as-is.
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
