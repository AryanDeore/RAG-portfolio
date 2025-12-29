"""
Simple KNN retrieval pipeline that embeds a query and searches top-k chunks in Qdrant.
"""

from typing import List, Dict, Optional, Any
from opik import Opik
from .fastembed_provider import FastEmbedProvider
from .qdrant_store import QdrantStore
from configs.settings import settings

# Create Opik client instance
opik_client = Opik()

def search_chunks(query: str, k: int | None = None, parent_span: Optional[Any] = None) -> List[Dict]:
    """
    Retrieve the top-k nearest chunks for a natural language query.

    Args:
        query (str): User information need expressed in plain English.
        k (int | None): Number of neighbors; defaults to settings.search_top_k when None.
        parent_span: Optional parent span for nested tracing.

    Returns:
        List[Dict]: Lightweight hit dicts with id, score, doc_id, chunk_id, title, and snippet text.
    """
    k = k or settings.search_top_k
    if parent_span:
        span = parent_span.span(
            name="vector_search",
            type="tool",
            input={"query": query, "k": k},
            metadata={
                "collection": settings.embed_collection_read or settings.embed_collection,
            }
        )
    else:
        span = opik_client.span(
            name="vector_search",
            type="tool",
            input={"query": query, "k": k},
            metadata={
                "collection": settings.embed_collection_read or settings.embed_collection,
            }
        )
    
    qvec = FastEmbedProvider().embed_query(query, parent_span=span)
    hits = QdrantStore().search(qvec, k=k)
    
    results = [
        {
            "id": h.id,
            "score": h.score,
            "doc_id": h.payload.get("doc_id"),
            "chunk_id": h.payload.get("chunk_id"),
            "title": h.payload.get("title"),
            "text": h.payload.get("text"),
            "links": h.payload.get("links"),
        }
        for h in hits
    ]
    
    # Log retrieval results
    span.end(
        output={
            "num_results": len(results),
            "top_score": results[0]["score"] if results else None,
            "chunk_ids": [r["id"] for r in results],
            "doc_ids": list(set(r["doc_id"] for r in results if r.get("doc_id"))),
        },
        metadata={
            "retrieval_k": k,
            "actual_results": len(results),
        }
    )
    
    return results