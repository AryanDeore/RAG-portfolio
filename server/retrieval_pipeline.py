"""Retrieval functions (KNN, HYDE, reranking) built on your existing Qdrant search."""

from typing import Dict, List, Iterator, Tuple
from litellm import completion
from src.shared.embedding.retrieval import search_chunks


def retrieve_knn(question: str, k: int) -> List[Dict]:
    """
    Runs plain KNN retrieval for a question and returns a list of hit dicts.
    
    Args:
        question (str): User's question to search for.
        k (int): Number of top results to retrieve.
    
    Returns:
        List[Dict]: List of hit dictionaries containing id, score, doc_id, chunk_id, title, and text fields.
    """
    return search_chunks(question, k=k)


def hyde_expand(question: str, model: str) -> str:
    """
    Generates a short hypothetical answer (HYDE) with an LLM and returns the synthetic text.
    
    Args:
        question (str): User's question to generate a hypothetical answer for.
        model (str): LiteLLM model identifier to use for generation.
    
    Returns:
        str: Generated hypothetical answer text.
    """
    sys = {
        "role": "system",
        "content": (
            "Write a concise, factual answer to the user's question if you can; "
            "otherwise produce a short plausible summary of what the answer might cover."
        ),
    }
    user = {"role": "user", "content": question}
    resp = completion(model=model, messages=[sys, user], temperature=0.0, stream=False)
    return resp["choices"][0]["message"]["content"].strip()


def retrieve_with_hyde(question: str, k: int, hyde_model: str) -> List[Dict]:
    """
    Runs dual retrieval with literal question and HYDE text, then merges by best score and returns top-k.
    
    Args:
        question (str): User's original question.
        k (int): Number of top results to return after merging.
        hyde_model (str): LiteLLM model identifier for HYDE expansion.
    
    Returns:
        List[Dict]: Merged and ranked list of hit dictionaries, limited to top-k results.
    """
    base_hits = search_chunks(question, k=k)
    pseudo = hyde_expand(question, model=hyde_model)
    hyde_hits = search_chunks(pseudo, k=k)

    merged: Dict[str, Dict] = {}
    for h in base_hits + hyde_hits:
        hid = str(h.get("id"))
        prev = merged.get(hid)
        if prev is None or (h.get("score", 0) > prev.get("score", 0)):
            merged[hid] = h
    ranked = sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked[:k]


def cheap_rerank(question: str, hits: List[Dict], top_n: int) -> List[Dict]:
    """
    Reranks hits using a lightweight lexical overlap heuristic and returns top_n items.
    
    Args:
        question (str): User's question to compare against hit text.
        hits (List[Dict]): List of hit dictionaries to rerank.
        top_n (int): Number of top results to return after reranking.
    
    Returns:
        List[Dict]: Reranked list of hit dictionaries, limited to top_n results.
    """
    q = (question or "").lower()

    def score(h: Dict) -> float:
        t = (h.get("text") or "").lower()
        overlap = sum(1 for w in q.split() if w and w in t)
        return 0.7 * float(h.get("score", 0.0)) + 0.3 * overlap

    ranked = sorted(hits, key=score, reverse=True)
    return ranked[: top_n or len(ranked)]


def llm_rerank(question: str, hits: List[Dict], top_n: int, model: str) -> List[Dict]:
    """
    Reranks hits with an LLM numeric relevance score 0–10 and returns top_n items.
    
    Args:
        question (str): User's question to evaluate relevance against.
        hits (List[Dict]): List of hit dictionaries to rerank.
        top_n (int): Number of top results to return after reranking.
        model (str): LiteLLM model identifier for relevance scoring.
    
    Returns:
        List[Dict]: LLM-reranked list of hit dictionaries, limited to top_n results.
    """
    scored: List[Tuple[float, Dict]] = []
    for h in hits:
        prompt = [
            {
                "role": "system",
                "content": "Rate the passage's relevance to the question from 0 to 10; respond with only the number.",
            },
            {"role": "user", "content": f"Question:\n{question}\n\nPassage:\n{h.get('text','')}"},
        ]
        try:
            r = completion(model=model, messages=prompt, temperature=0.0, stream=False)
            s = float(r["choices"][0]["message"]["content"].strip())
        except Exception:
            s = 0.0
        scored.append((s, h))
    ranked = [h for s, h in sorted(scored, key=lambda x: x[0], reverse=True)]
    return ranked[: top_n or len(ranked)]
