"""
Simple KNN retrieval pipeline that embeds a query and searches top-k chunks in Qdrant.
"""

from typing import List, Dict
from .fastembed_provider import FastEmbedProvider
from .qdrant_store import QdrantStore
from configs.settings import settings

def search_chunks(query: str, k: int | None = None) -> List[Dict]:
    """
    Retrieve the top-k nearest chunks for a natural language query.

    Args:
        query (str): User information need expressed in plain English.
        k (int | None): Number of neighbors; defaults to settings.retrieval_k when None.

    Returns:
        List[Dict]: Lightweight hit dicts with id, score, doc_id, chunk_id, title, and snippet text.
    """
    k = k or settings.retrieval_k
    qvec = FastEmbedProvider().embed_query(query)
    hits = QdrantStore().search(qvec, k=k)
    return [
        {
            "id": h.id,
            "score": h.score,
            "doc_id": h.payload.get("doc_id"),
            "chunk_id": h.payload.get("chunk_id"),
            "title": h.payload.get("title"),
            "text": h.payload.get("text"),
        }
        for h in hits
    ]
