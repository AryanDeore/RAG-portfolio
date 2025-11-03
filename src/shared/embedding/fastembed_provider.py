"""
Thin wrapper around FastEmbed that applies instruction prefixes ('passage:'/'query:') for better retrieval quality.
"""

from fastembed import TextEmbedding
from configs.settings import settings

class FastEmbedProvider:
    def __init__(self):
        self.model = TextEmbedding(
            model_name=settings.embed_model,
            cache_dir=settings.embed_cache_dir
        )

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
