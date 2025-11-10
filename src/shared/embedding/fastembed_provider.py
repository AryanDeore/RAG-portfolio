"""
Thin wrapper around FastEmbed that applies instruction prefixes ('passage:'/'query:') for better retrieval quality.
Optimized for AWS Lambda with singleton pattern to avoid reloading models.
"""

from fastembed import TextEmbedding
from configs.settings import settings
import os
from pathlib import Path

class FastEmbedProvider:
    """
    Singleton provider for FastEmbed models.
    
    Ensures only one instance exists per process, reusing the loaded model
    across multiple requests. This is especially important for AWS Lambda
    where warm containers can reuse the same instance.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to reuse model instance across requests in the same Lambda container."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model only once, even if __init__ is called multiple times."""
        if FastEmbedProvider._initialized:
            return
        
        # Use cache_dir from settings, or None (FastEmbed's default)
        cache_dir = settings.embed_cache_dir
        
        # If explicitly set to a problematic path, validate it
        if cache_dir and cache_dir.startswith("/opt/python"):
            # Only use /opt/python if it exists and is writable
            if not (os.path.exists("/opt/python") and os.access("/opt/python", os.W_OK)):
                # Fall back to None if /opt/python isn't available
                cache_dir = None
        
        # Initialize model with error handling
        try:
            self.model = TextEmbedding(
                model_name=settings.embed_model,
                cache_dir=cache_dir
            )
        except (OSError, PermissionError) as e:
            # If FastEmbed fails to create cache directory, retry with None
            if cache_dir is not None:
                self.model = TextEmbedding(
                    model_name=settings.embed_model,
                    cache_dir=None
                )
            else:
                raise
        
        FastEmbedProvider._initialized = True

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """
        Embed chunk with 'passages' with instruction prefix.

        Args:
            texts (list[str]): Plain chunk strings to embed.

        Returns:
            list[list[float]]: Dense vectors (one per input text), each matching the model dimension.
        """
        return list(self.model.embed([f"passage: {t}" for t in texts]))

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single user query with the recommended 'query:' prefix.

        Args:
            query (str): The user information need expressed as natural language.

        Returns:
            list[float]: Dense vector suitable for ANN search against passage vectors.
        """
        return list(self.model.embed([f"query: {query}"]))[0]
