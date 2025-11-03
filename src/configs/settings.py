"""
Centralized application configuration (Pydantic Settings v2).
Reads from init kwargs > environment > .env and exposes tunables used across the app,
including chunking, embedding/retrieval, and Qdrant connection details.
"""

from functools import lru_cache
from typing import Optional, Annotated, List
from pydantic import Field, AliasChoices
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

    # ---- App basics ----
    app_name: str = Field(default="RAG Portfolio", description="Human-friendly app name")
    debug: bool = Field(default=False, description="Enable verbose logging and debug switches")
    log_level: str = Field(default="INFO", description="Root log level")
    cors_allow_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000"], description="Allowed CORS origins"
    )

    # ---- Secrets / external ----
    llm_api_key: Optional[str] = Field(default=None, description="Provider API key for LLM (if used)")
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="API key for Qdrant Cloud (leave None for local/no-auth)",
    )
    database_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL"),
        description="SQLAlchemy-style URL for relational DB (optional)",
    )

    # ---- Qdrant connection ----
    qdrant_host: str = Field(
        default="http://localhost",
        validation_alias=AliasChoices("QDRANT_HOST", "QDRANT_URL_BASE"),
        description="Host (with scheme) for Qdrant service, e.g., http://localhost or https://xxx.aws.cloud.qdrant.io",
    )
    qdrant_port: Port = Field(
        default=6333,
        description="TCP port for Qdrant; ignored by some managed cloud endpoints",
    )

    # ---- Chunker knobs (consumed directly by chunking.*) ----
    chunk_max_chars_paragraph: int = Field(default=700, description="Hard cap for paragraph-sized chunks")
    chunk_split_long_bullets: bool = Field(default=False, description="Split long bullets into child chunks")
    chunk_max_chars_bullet: int = Field(default=700, description="Hard cap for bullet-sized chunks")

    # ---- JSON catalogs for entity normalization (chunking.entities) ----
    tech_alias_path: str = Field(default="configs/tech_alias.json", description="Alias map for tech entities")
    tech_catalog_path: str = Field(default="configs/tech_catalog.json", description="Canonical tech catalog JSON")

    # ---- Embedding / Retrieval (v1 defaults) ----
    embed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        validation_alias=AliasChoices("EMBED_MODEL"),
        description="FastEmbed model name for dense embeddings (English)",
    )
    embed_dim: int = Field(
        default=384,
        validation_alias=AliasChoices("EMBED_DIM"),
        description="Output vector size of the embedding model; must match collection vector size",
    )
    embed_metric: str = Field(
        default="Cosine",
        validation_alias=AliasChoices("EMBED_METRIC"),
        description='Vector similarity metric: "Cosine" | "Dot" | "Euclid"',
    )
    embed_batch_size: int = Field(
        default=128,
        validation_alias=AliasChoices("EMBED_BATCH", "EMBED_BATCH_SIZE"),
        description="Batch size for embedding/upserts",
    )
    embed_cache_dir: str = Field(
        default=".cache_fastembed",
        validation_alias=AliasChoices("EMBED_CACHE_DIR"),
        description="Local cache dir for FastEmbed model files",
    )
    embed_collection: str = Field(
        default="resourcebooks_v1",
        validation_alias=AliasChoices("EMBED_COLLECTION", "QDRANT_COLLECTION"),
        description="Qdrant collection name for storing vectors",
    )
    embed_id_scheme: str = Field(
        default="uuid",
        validation_alias=AliasChoices("EMBED_ID_SCHEME"),
        description="Scheme for generating point IDs: 'uuid' (default) | 'int'",
    )
    retrieval_k: int = Field(
        default=10,
        validation_alias=AliasChoices("RETRIEVAL_K", "TOPK"),
        description="Default number of nearest neighbors to retrieve",
    )

    # ---- Observability (optional) ----
    comet_project_name: str = Field(
        default="rag-embed",
        validation_alias=AliasChoices("COMET_PROJECT_NAME"),
        description="Comet project to log runs under (if COMET_API_KEY is set)",
    )

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
