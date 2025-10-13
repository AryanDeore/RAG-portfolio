"""Pydantic Settings v2 configuration for RAG Portfolio project."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings using Pydantic Settings v2."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application settings
    app_name: str = Field(default="RAG Portfolio", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # API settings
    # api_host: str = Field(default="127.0.0.1", description="API host")
    # api_port: int = Field(default=8000, description="API port")
    # api_prefix: str = Field(default="/api/v1", description="API prefix")
    
    # LLM settings (LiteLLM)
    # llm_provider: str = Field(default="openai", description="LLM provider")
    # llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model")
    # llm_api_key: Optional[str] = Field(default=None, description="LLM API key")
    # llm_base_url: Optional[str] = Field(default=None, description="LLM base URL")
    # llm_temperature: float = Field(default=0.7, description="LLM temperature")
    # llm_max_tokens: int = Field(default=1000, description="Max tokens for LLM")
    
    # Vector database settings (Qdrant)
    # qdrant_host: str = Field(default="localhost", description="Qdrant host")
    # qdrant_port: int = Field(default=6333, description="Qdrant port")
    # qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    # qdrant_collection_name: str = Field(default="rag_portfolio", description="Qdrant collection name")
    # qdrant_vector_size: int = Field(default=1536, description="Vector size for embeddings")
    
    # Comet ML settings
    # comet_api_key: Optional[str] = Field(default=None, description="Comet ML API key")
    # comet_project_name: str = Field(default="rag-portfolio", description="Comet ML project name")
    # comet_workspace: Optional[str] = Field(default=None, description="Comet ML workspace")
    
    # Embedding settings
    # embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    # embedding_dimensions: int = Field(default=1536, description="Embedding dimensions")
    # chunk_size: int = Field(default=1000, description="Text chunk size for processing")
    # chunk_overlap: int = Field(default=200, description="Text chunk overlap")
    
    # Redis settings (for caching)
    # redis_host: str = Field(default="localhost", description="Redis host")
    # redis_port: int = Field(default=6379, description="Redis port")
    # redis_password: Optional[str] = Field(default=None, description="Redis password")
    # redis_db: int = Field(default=0, description="Redis database number")
    
    # Security settings
    # secret_key: str = Field(default="your-secret-key-change-this", description="Secret key for JWT")
    # access_token_expire_minutes: int = Field(default=30, description="Access token expiration in minutes")
    # algorithm: str = Field(default="HS256", description="JWT algorithm")
    
    # Database settings
    # database_url: str = Field(default="sqlite:///./rag_portfolio.db", description="Database URL")
    
    # File storage settings
    # upload_dir: str = Field(default="./uploads", description="Upload directory")
    # max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes (10MB)")
    # allowed_file_types: list[str] = Field(
    #     default=[".pdf", ".txt", ".docx", ".md"], 
    #     description="Allowed file types"
    # )


# Create a global settings instance
settings = Settings()


# Example of how to use the settings
if __name__ == "__main__":
    print("Current settings:")
    print(f"App Name: {settings.app_name}")
