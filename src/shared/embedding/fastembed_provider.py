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
        
        # Determine cache directory
        cache_dir = settings.embed_cache_dir
        if cache_dir is None:
            # Check if we're running on AWS Lambda
            is_lambda = os.environ.get("LAMBDA_TASK_ROOT") is not None
            
            if is_lambda:
                # On Lambda: try Lambda Layers location first
                lambda_cache = "/opt/python/fastembed_cache"
                # Check if /opt/python exists and we can write to it
                if os.path.exists("/opt/python") and os.access("/opt/python", os.W_OK):
                    # Try to create the cache directory to verify we can write
                    try:
                        Path(lambda_cache).mkdir(parents=True, exist_ok=True)
                        cache_dir = lambda_cache
                    except (OSError, PermissionError):
                        # Can't write to /opt/python, use /tmp instead
                        cache_dir = "/tmp/fastembed_cache"
                else:
                    # /opt/python doesn't exist or isn't writable, use /tmp
                    cache_dir = "/tmp/fastembed_cache"
            else:
                # Local development: use FastEmbed's default cache location
                # This will be ~/.cache/fastembed on most systems
                cache_dir = None  # Let FastEmbed use its default
        
        self.model = TextEmbedding(
            model_name=settings.embed_model,
            cache_dir=cache_dir
        )
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
