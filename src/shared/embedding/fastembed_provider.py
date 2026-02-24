"""
Thin wrapper around FastEmbed that applies instruction prefixes ('passage:'/'query:') for better retrieval quality.
Optimized for AWS Lambda with singleton pattern to avoid reloading models.
"""

from fastembed import TextEmbedding, SparseTextEmbedding
from configs.settings import settings
import os
from pathlib import Path
import opik

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
        
        # Initialize dense model with error handling
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

        # Initialize BM25 sparse model
        try:
            self.sparse_model = SparseTextEmbedding(
                model_name="Qdrant/bm25",
                cache_dir=cache_dir
            )
        except (OSError, PermissionError):
            if cache_dir is not None:
                self.sparse_model = SparseTextEmbedding(
                    model_name="Qdrant/bm25",
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

    def embed_query(self, query: str, parent_span=None) -> list[float]:
        """
        Embed a single user query with the recommended 'query:' prefix.

        Args:
            query (str): The user information need expressed as natural language.
            parent_span: Optional parent span for nested tracing.

        Returns:
            list[float]: Dense vector suitable for ANN search against passage vectors.
        """
        if parent_span:
            span = parent_span.span(
                name="embed_query",
                type="tool",
                input={"query": query},
                metadata={
                    "embedding_model": settings.embed_model,
                    "embedding_dim": settings.embed_dim,
                }
            )
        else:
            from opik import Opik
            opik_client = Opik()
            span = opik_client.span(
                name="embed_query",
                type="tool",
                input={"query": query},
                metadata={
                    "embedding_model": settings.embed_model,
                    "embedding_dim": settings.embed_dim,
                }
            )
        
        embedding = list(self.model.embed([f"query: {query}"]))[0]

        span.end(
            output={
                "embedding_dim": len(embedding),
                "embedding_norm": sum(x * x for x in embedding) ** 0.5,
            }
        )

        return embedding

    def embed_passages_sparse(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """
        Embed passages with BM25 sparse model for keyword matching.

        Args:
            texts (list[str]): Plain chunk strings to embed.

        Returns:
            list[tuple[list[int], list[float]]]: Sparse vectors as (indices, values) tuples.
        """
        results = list(self.sparse_model.embed(texts))
        return [
            (r.indices.tolist(), r.values.tolist())
            for r in results
        ]

    def embed_query_sparse(self, query: str, parent_span=None) -> tuple[list[int], list[float]]:
        """
        Embed a single query with BM25 sparse model.

        Args:
            query (str): The user query.
            parent_span: Optional parent span for nested tracing.

        Returns:
            tuple[list[int], list[float]]: Sparse vector as (indices, values).
        """
        if parent_span:
            span = parent_span.span(
                name="embed_query_sparse",
                type="tool",
                input={"query": query},
                metadata={"sparse_model": "Qdrant/bm25"}
            )
        else:
            from opik import Opik
            span = Opik().span(
                name="embed_query_sparse",
                type="tool",
                input={"query": query},
                metadata={"sparse_model": "Qdrant/bm25"}
            )

        result = list(self.sparse_model.query_embed(query))[0]
        indices = result.indices.tolist()
        values = result.values.tolist()

        span.end(
            output={"num_nonzero": len(indices)}
        )

        return (indices, values)
