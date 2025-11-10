"""FastAPI routes exposing non-stream and streaming chat endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Iterator, List, Dict
from litellm import completion
from server.schemas import ChatRequest, ChatResponse
from server.utils import join_context, build_messages
from server.retrieval_pipeline import (
    retrieve_knn,
    retrieve_with_hyde,
    cheap_rerank,
    llm_rerank,
)

router = APIRouter()


def _choose_retrieval(req: ChatRequest) -> List[Dict]:
    """
    Chooses KNN or HYDE retrieval based on request flags and returns a list of hit dicts.
    
    Args:
        req (ChatRequest): Chat request containing retrieval configuration and question.
    
    Returns:
        List[Dict]: List of hit dictionaries containing id, score, doc_id, chunk_id, title, and text fields.
    """
    if req.use_hyde:
        return retrieve_with_hyde(req.question, k=req.k, hyde_model=req.model)
    return retrieve_knn(req.question, k=req.k)


def _maybe_rerank(req: ChatRequest, hits: List[Dict]) -> List[Dict]:
    """
    Applies reranking if requested and returns the possibly reranked hit list.
    
    Args:
        req (ChatRequest): Chat request containing reranking configuration.
        hits (List[Dict]): List of hit dictionaries to potentially rerank.
    
    Returns:
        List[Dict]: Reranked or original list of hit dictionaries.
    """
    if req.rerank == "cheap":
        return cheap_rerank(req.question, hits, top_n=req.rerank_top_n or req.k)
    if req.rerank == "llm":
        return llm_rerank(req.question, hits, top_n=req.rerank_top_n or req.k, model=req.model)
    return hits


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> JSONResponse:
    """
    Handles non-streaming chat by retrieving context and returning a complete answer.
    
    Args:
        req (ChatRequest): Chat request containing question, history, retrieval options, and model parameters.
    
    Returns:
        JSONResponse: Response containing ChatResponse with the complete generated answer text.
    """
    try:
        hits = _choose_retrieval(req)
        hits = _maybe_rerank(req, hits)
        context = join_context(hits)
        history = [m.model_dump() for m in req.history]
        messages = build_messages(req.question, history, context)

        resp = completion(
            model=req.model,
            messages=messages,
            temperature=req.temperature,
            stream=False,
        )
        text = resp["choices"][0]["message"]["content"]
        return JSONResponse(ChatResponse(answer=text).model_dump())
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Generation failed: {e}") from e


@router.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    Handles streaming chat by yielding incremental tokens as plain text.
    
    Args:
        req (ChatRequest): Chat request containing question, history, retrieval options, and model parameters.
    
    Returns:
        StreamingResponse: Streaming response that yields UTF-8 encoded text chunks as they are generated.
    """
    hits = _choose_retrieval(req)
    hits = _maybe_rerank(req, hits)
    context = join_context(hits)
    history = [m.model_dump() for m in req.history]
    messages = build_messages(req.question, history, context)

    def gen() -> Iterator[bytes]:
        """
        Streams LLM deltas as UTF-8 encoded bytes for chunked transfer.
        
        Returns:
            Iterator[bytes]: Generator yielding UTF-8 encoded text chunks from the LLM stream.
        """
        try:
            for chunk in completion(
                model=req.model,
                messages=messages,
                temperature=req.temperature,
                stream=True,
            ):
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta.encode("utf-8")
        except Exception as e:
            yield f"\n\n[stream_error] {e}".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
