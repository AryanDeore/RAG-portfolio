"""Retrieval functions (KNN, HYDE, hybrid multi-query, reranking) built on Qdrant search."""

from typing import Dict, List, Iterator, Tuple, Optional, Any
from opik import Opik
from litellm import completion
from src.shared.embedding.retrieval import search_chunks
from server.utils import get_model_name

# Create Opik client instance
opik_client = Opik()


def retrieve_knn(question: str, k: int, parent_span: Optional[Any] = None) -> List[Dict]:
    """
    Runs plain KNN retrieval for a question and returns a list of hit dicts.

    Args:
        question (str): User's question to search for.
        k (int): Number of top results to retrieve.
        parent_span: Optional parent span for nested tracing.

    Returns:
        List[Dict]: List of hit dictionaries containing id, score, doc_id, chunk_id, title, and text fields.
    """
    if parent_span:
        span = parent_span.span(
            name="knn_retrieval",
            type="tool",
            input={"question": question, "k": k},
            metadata={"retrieval_method": "knn"}
        )
    else:
        span = opik_client.span(
            name="knn_retrieval",
            type="tool",
            input={"question": question, "k": k},
            metadata={"retrieval_method": "knn"}
        )

    results = search_chunks(question, k=k, parent_span=span)

    span.end(
        output={
            "num_results": len(results),
            "top_score": results[0]["score"] if results else None,
        }
    )

    return results


def hyde_expand(question: str, model: str, parent_span: Optional[Any] = None) -> str:
    """
    Generates a short hypothetical answer (HYDE) with an LLM and returns the synthetic text.

    Args:
        question (str): User's question to generate a hypothetical answer for.
        model (str): LiteLLM model identifier to use for generation.
        parent_span: Optional parent span for nested tracing.

    Returns:
        str: Generated hypothetical answer text.
    """
    if parent_span:
        span = parent_span.span(
            name="hyde_expansion",
            type="llm",
            input={"question": question},
            metadata={"hyde_model": model}
        )
    else:
        span = opik_client.span(
            name="hyde_expansion",
            type="llm",
            input={"question": question},
            metadata={"hyde_model": model}
        )

    sys = {
        "role": "system",
        "content": (
            "Write a concise, factual answer to the user's question if you can; "
            "otherwise produce a short plausible summary of what the answer might cover."
        ),
    }
    user = {"role": "user", "content": question}
    resp = completion(model=get_model_name(model), messages=[sys, user], temperature=0.0, stream=False)
    hypothetical_answer = resp["choices"][0]["message"]["content"].strip()

    usage = resp.get("usage", {})
    span.end(
        output={"hypothetical_answer": hypothetical_answer},
        usage={
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        } if usage else None,
        model=model,
        provider="litellm"
    )

    return hypothetical_answer


def retrieve_with_hyde(question: str, k: int, hyde_model: str, parent_span: Optional[Any] = None) -> List[Dict]:
    """
    Runs dual retrieval with literal question and HYDE text, then merges by best score and returns top-k.

    Args:
        question (str): User's original question.
        k (int): Number of top results to return after merging.
        hyde_model (str): LiteLLM model identifier for HYDE expansion.
        parent_span: Optional parent span for nested tracing.

    Returns:
        List[Dict]: Merged and ranked list of hit dictionaries, limited to top-k results.
    """
    if parent_span:
        span = parent_span.span(
            name="hyde_retrieval",
            type="tool",
            input={"question": question, "k": k, "hyde_model": hyde_model},
            metadata={"retrieval_method": "hyde"}
        )
    else:
        span = opik_client.span(
            name="hyde_retrieval",
            type="tool",
            input={"question": question, "k": k, "hyde_model": hyde_model},
            metadata={"retrieval_method": "hyde"}
        )

    base_hits = search_chunks(question, k=k, parent_span=span)
    pseudo = hyde_expand(question, model=hyde_model, parent_span=span)
    hyde_hits = search_chunks(pseudo, k=k, parent_span=span)

    merged: Dict[str, Dict] = {}
    for h in base_hits + hyde_hits:
        hid = str(h.get("id"))
        prev = merged.get(hid)
        if prev is None or (h.get("score", 0) > prev.get("score", 0)):
            merged[hid] = h
    ranked = sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    results = ranked[:k]

    span.end(
        output={
            "num_results": len(results),
            "base_hits_count": len(base_hits),
            "hyde_hits_count": len(hyde_hits),
            "merged_unique_count": len(merged),
            "top_score": results[0]["score"] if results else None,
        }
    )

    return results


def retrieve_hybrid_multi(
    sub_queries: List[str],
    k: int,
    parent_span: Optional[Any] = None,
) -> List[Dict]:
    """
    Run hybrid retrieval for each sub-query, merge by point ID (best score wins), return top-k.

    Args:
        sub_queries: Expanded sub-query strings from decompose_and_expand.
        k: Number of top results to return after merging.
        parent_span: Optional Opik parent span for tracing.

    Returns:
        List[Dict]: Deduplicated, score-sorted hit dicts, limited to top-k.
    """
    if parent_span:
        span = parent_span.span(
            name="hybrid_multi_retrieval",
            type="tool",
            input={"sub_queries": sub_queries, "k": k},
            metadata={"retrieval_method": "hybrid_multi", "num_sub_queries": len(sub_queries)},
        )
    else:
        span = opik_client.span(
            name="hybrid_multi_retrieval",
            type="tool",
            input={"sub_queries": sub_queries, "k": k},
            metadata={"retrieval_method": "hybrid_multi", "num_sub_queries": len(sub_queries)},
        )

    # Scale final return count so each sub-query intent gets fair representation.
    # join_context caps characters anyway, so returning more chunks is safe.
    final_k = k * len(sub_queries)

    merged: Dict[str, Dict] = {}
    for sq in sub_queries:
        hits = search_chunks(sq, k=k, parent_span=span)
        for h in hits:
            hid = str(h.get("id"))
            prev = merged.get(hid)
            if prev is None or (h.get("score", 0) > prev.get("score", 0)):
                merged[hid] = h

    ranked = sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    results = ranked[:final_k]

    span.end(
        output={
            "num_results": len(results),
            "total_unique_chunks": len(merged),
            "sub_queries": sub_queries,
            "top_score": results[0]["score"] if results else None,
        },
    )

    return results


def cheap_rerank(question: str, hits: List[Dict], top_n: int, parent_span: Optional[Any] = None) -> List[Dict]:
    """
    Reranks hits using a lightweight lexical overlap heuristic and returns top_n items.

    Args:
        question (str): User's question to compare against hit text.
        hits (List[Dict]): List of hit dictionaries to rerank.
        top_n (int): Number of top results to return after reranking.
        parent_span: Optional parent span for nested tracing.

    Returns:
        List[Dict]: Reranked list of hit dictionaries, limited to top_n results.
    """
    if parent_span:
        span = parent_span.span(
            name="cheap_rerank",
            type="tool",
            input={"question": question, "input_hits": len(hits), "top_n": top_n},
            metadata={"rerank_type": "cheap"}
        )
    else:
        span = opik_client.span(
            name="cheap_rerank",
            type="tool",
            input={"question": question, "input_hits": len(hits), "top_n": top_n},
            metadata={"rerank_type": "cheap"}
        )

    q = (question or "").lower()

    def score(h: Dict) -> float:
        t = (h.get("text") or "").lower()
        overlap = sum(1 for w in q.split() if w and w in t)
        return 0.7 * float(h.get("score", 0.0)) + 0.3 * overlap

    ranked = sorted(hits, key=score, reverse=True)
    results = ranked[: top_n or len(ranked)]

    span.end(
        output={
            "input_hits": len(hits),
            "output_hits": len(results),
            "top_score": results[0]["score"] if results else None,
        }
    )

    return results


def llm_rerank(question: str, hits: List[Dict], top_n: int, model: str, parent_span: Optional[Any] = None) -> List[Dict]:
    """
    Reranks hits with an LLM numeric relevance score 0–10 and returns top_n items.

    Args:
        question (str): User's question to evaluate relevance against.
        hits (List[Dict]): List of hit dictionaries to rerank.
        top_n (int): Number of top results to return after reranking.
        model (str): LiteLLM model identifier for relevance scoring.
        parent_span: Optional parent span for nested tracing.

    Returns:
        List[Dict]: LLM-reranked list of hit dictionaries, limited to top_n results.
    """
    if parent_span:
        span = parent_span.span(
            name="llm_rerank",
            type="tool",
            input={"question": question, "input_hits": len(hits), "top_n": top_n},
            metadata={"rerank_type": "llm", "rerank_model": model}
        )
    else:
        span = opik_client.span(
            name="llm_rerank",
            type="tool",
            input={"question": question, "input_hits": len(hits), "top_n": top_n},
            metadata={"rerank_type": "llm", "rerank_model": model}
        )

    scored: List[Tuple[float, Dict]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for h in hits:
        prompt = [
            {
                "role": "system",
                "content": "Rate the passage's relevance to the question from 0 to 10; respond with only the number.",
            },
            {"role": "user", "content": f"Question:\n{question}\n\nPassage:\n{h.get('text','')}"},
        ]
        try:
            r = completion(model=get_model_name(model), messages=prompt, temperature=0.0, stream=False)
            s = float(r["choices"][0]["message"]["content"].strip())
            usage = r.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
        except Exception:
            s = 0.0
        scored.append((s, h))

    ranked = [h for s, h in sorted(scored, key=lambda x: x[0], reverse=True)]
    results = ranked[: top_n or len(ranked)]

    span.end(
        output={
            "input_hits": len(hits),
            "output_hits": len(results),
            "top_score": results[0]["score"] if results else None,
            "top_rerank_score": scored[0][0] if scored else None,
        },
        usage={
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        model=model,
        provider="litellm"
    )

    return results
