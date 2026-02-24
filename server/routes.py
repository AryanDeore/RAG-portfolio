"""FastAPI routes exposing non-stream and streaming chat endpoints."""

import logging
import os
from typing import Iterator, List, Dict

from opik import Opik
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from litellm import completion
from litellm.exceptions import (
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    APIError,
)
from server.schemas import ChatRequest, ChatResponse
from server.utils import join_context, build_messages, get_model_name
from server.query_processing import moderate_query, decompose_and_expand
from server.retrieval_pipeline import (
    retrieve_hybrid_multi,
    cheap_rerank,
    llm_rerank,
)

MODERATION_BLOCKED_MSG = (
    "Your message was flagged for inappropriate content. "
    "Please keep questions respectful and on-topic."
)


def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key from request header."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        return True
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# Set up logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Create Opik client instance
opik_client = Opik()


def _maybe_rerank(req: ChatRequest, hits: List[Dict], parent_span=None) -> List[Dict]:
    """
    Applies reranking if requested and returns the possibly reranked hit list.

    Args:
        req (ChatRequest): Chat request containing reranking configuration.
        hits (List[Dict]): List of hit dictionaries to potentially rerank.
        parent_span: Optional parent span for nested tracing.

    Returns:
        List[Dict]: Reranked or original list of hit dictionaries.
    """
    if req.rerank == "cheap":
        return cheap_rerank(req.question, hits, top_n=req.rerank_top_n or req.k, parent_span=parent_span)
    if req.rerank == "llm":
        return llm_rerank(req.question, hits, top_n=req.rerank_top_n or req.k, model=req.model, parent_span=parent_span)
    return hits


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: bool = Depends(verify_api_key)) -> JSONResponse:
    """
    Handles non-streaming chat with moderation, query expansion, hybrid retrieval, and generation.

    Args:
        req (ChatRequest): Chat request containing question, history, retrieval options, and model parameters.

    Returns:
        JSONResponse: Response containing ChatResponse with the complete generated answer text.
    """
    trace = opik_client.trace(
        name="rag_query",
        input={
            "question": req.question,
            "model": req.model,
            "temperature": req.temperature,
            "use_hyde": req.use_hyde,
            "rerank": req.rerank,
            "k": req.k,
            "rerank_top_n": req.rerank_top_n,
            "stream": False,
            "history_length": len(req.history),
        },
        tags=["rag", "chat", "non-streaming", "hybrid"],
        metadata={"endpoint": "/chat"},
    )

    try:
        # 1. Moderation gate
        flagged = moderate_query(req.question, parent_span=trace)
        if flagged:
            trace.end(
                output={"answer": MODERATION_BLOCKED_MSG, "moderation_blocked": True},
            )
            return JSONResponse(ChatResponse(answer=MODERATION_BLOCKED_MSG).model_dump())

        # 2. Decompose & expand
        sub_queries = decompose_and_expand(req.question, model=req.model, parent_span=trace)

        # 3. Log preprocessing summary
        preprocess_span = trace.span(
            name="query_preprocessing_summary",
            type="general",
            input={"original_query": req.question},
        )
        preprocess_span.end(
            output={
                "moderation_flagged": False,
                "sub_queries": sub_queries,
                "num_sub_queries": len(sub_queries),
                "was_decomposed": len(sub_queries) > 1,
            },
        )

        # 4. Hybrid multi-query retrieval
        retrieval_span = trace.span(
            name="retrieve_chunks",
            type="tool",
            input={"sub_queries": sub_queries, "k": req.k},
        )
        hits = retrieve_hybrid_multi(sub_queries, k=req.k, parent_span=retrieval_span)
        hits = _maybe_rerank(req, hits, parent_span=retrieval_span)
        retrieval_span.end(
            output={
                "num_chunks": len(hits),
                "chunk_ids": [h.get("id") for h in hits],
                "chunk_scores": [h.get("score") for h in hits],
                "doc_ids": list(set(h.get("doc_id") for h in hits if h.get("doc_id"))),
            },
            metadata={"retrieval_method": "hybrid_multi", "rerank_method": req.rerank},
        )

        # 5. Context building (uses original question, not sub-queries)
        #    Scale context cap so multi-intent queries have room for all intents.
        cap = 1800 * len(sub_queries)
        context_span = trace.span(
            name="build_context",
            type="general",
            input={"num_hits": len(hits), "cap_chars": cap},
        )
        context = join_context(hits, cap_chars=cap)
        history = [m.model_dump() for m in req.history]
        messages = build_messages(req.question, history, context)
        context_span.end(
            output={"context_length": len(context), "num_messages": len(messages)},
        )

        # 6. LLM generation
        llm_span = trace.span(
            name="llm_completion",
            type="llm",
            input={"model": req.model, "temperature": req.temperature, "num_messages": len(messages)},
            metadata={"model": req.model},
        )
        resp = completion(
            model=get_model_name(req.model),
            messages=messages,
            temperature=req.temperature,
            stream=False,
        )
        text = resp["choices"][0]["message"]["content"]

        usage = resp.get("usage", {})
        llm_span.end(
            output={
                "answer_length": len(text),
                "answer": text[:500] if len(text) > 500 else text,
            },
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            } if usage else None,
            model=req.model,
            provider="litellm",
        )

        trace.end(
            output={
                "answer": text[:500] if len(text) > 500 else text,
                "answer_length": len(text),
                "num_chunks_used": len(hits),
            },
        )

        return JSONResponse(ChatResponse(answer=text).model_dump())

    except RateLimitError as e:
        logger.error(f"OpenRouter rate limit exceeded for model {req.model}: {e}")
        trace.end(output={"error": str(e)}, metadata={"error_type": "RateLimitError"})
        raise HTTPException(status_code=429, detail="API rate limit exceeded. Please try again later.") from e
    except AuthenticationError as e:
        logger.error(f"Authentication failed for model {req.model}: {e}")
        trace.end(output={"error": str(e)}, metadata={"error_type": "AuthenticationError"})
        raise HTTPException(status_code=401, detail="API authentication failed. Please check your OpenRouter API key.") from e
    except InvalidRequestError as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower() or "insufficient credits" in error_msg.lower():
            logger.error(f"Insufficient OpenRouter credits for model {req.model}: {e}")
            trace.end(output={"error": str(e)}, metadata={"error_type": "InsufficientQuota"})
            raise HTTPException(status_code=402, detail="Insufficient API credits. Please add credits to your OpenRouter account.") from e
        logger.error(f"Invalid OpenRouter request for model {req.model}: {e}")
        trace.end(output={"error": error_msg}, metadata={"error_type": "InvalidRequestError"})
        raise HTTPException(status_code=400, detail=f"Invalid request: {error_msg}") from e
    except APIError as e:
        logger.error(f"OpenRouter API error for model {req.model}: {e}")
        trace.end(output={"error": str(e)}, metadata={"error_type": "APIError"})
        raise HTTPException(status_code=503, detail=f"API service error: {str(e)}") from e
    except Exception as e:
        logger.exception(f"Unexpected error during chat generation: {e}")
        trace.end(output={"error": str(e)}, metadata={"error_type": type(e).__name__})
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}") from e


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, _: bool = Depends(verify_api_key)) -> StreamingResponse:
    """
    Handles streaming chat with moderation, query expansion, hybrid retrieval, and generation.

    Args:
        req (ChatRequest): Chat request containing question, history, retrieval options, and model parameters.

    Returns:
        StreamingResponse: Streaming response that yields UTF-8 encoded text chunks as they are generated.
    """
    trace = opik_client.trace(
        name="rag_query",
        input={
            "question": req.question,
            "model": req.model,
            "temperature": req.temperature,
            "use_hyde": req.use_hyde,
            "rerank": req.rerank,
            "k": req.k,
            "rerank_top_n": req.rerank_top_n,
            "stream": True,
            "history_length": len(req.history),
        },
        tags=["rag", "chat", "streaming", "hybrid"],
        metadata={"endpoint": "/chat/stream"},
    )

    def gen() -> Iterator[bytes]:
        try:
            # 1. Moderation gate
            flagged = moderate_query(req.question, parent_span=trace)
            if flagged:
                trace.end(
                    output={"answer": MODERATION_BLOCKED_MSG, "moderation_blocked": True},
                )
                yield MODERATION_BLOCKED_MSG.encode("utf-8")
                return

            # 2. Decompose & expand
            sub_queries = decompose_and_expand(req.question, model=req.model, parent_span=trace)

            # 3. Log preprocessing summary
            preprocess_span = trace.span(
                name="query_preprocessing_summary",
                type="general",
                input={"original_query": req.question},
            )
            preprocess_span.end(
                output={
                    "moderation_flagged": False,
                    "sub_queries": sub_queries,
                    "num_sub_queries": len(sub_queries),
                    "was_decomposed": len(sub_queries) > 1,
                },
            )

            # Scale context cap for multi-intent queries
            cap = 1800 * len(sub_queries)

            # 4. Hybrid multi-query retrieval
            retrieval_span = trace.span(
                name="retrieve_chunks",
                type="tool",
                input={"sub_queries": sub_queries, "k": req.k},
            )
            hits = retrieve_hybrid_multi(sub_queries, k=req.k, parent_span=retrieval_span)
            hits = _maybe_rerank(req, hits, parent_span=retrieval_span)
            retrieval_span.end(
                output={
                    "num_chunks": len(hits),
                    "chunk_ids": [h.get("id") for h in hits],
                    "chunk_scores": [h.get("score") for h in hits],
                    "doc_ids": list(set(h.get("doc_id") for h in hits if h.get("doc_id"))),
                },
                metadata={"retrieval_method": "hybrid_multi", "rerank_method": req.rerank},
            )

            # 5. Context building
            context_span = trace.span(
                name="build_context",
                type="general",
                input={"num_hits": len(hits), "cap_chars": cap},
            )
            context = join_context(hits, cap_chars=cap)
            history = [m.model_dump() for m in req.history]
            messages = build_messages(req.question, history, context)
            context_span.end(
                output={"context_length": len(context), "num_messages": len(messages)},
            )

            # 6. Streaming LLM generation
            llm_span = trace.span(
                name="llm_completion_stream",
                type="llm",
                input={"model": req.model, "temperature": req.temperature, "num_messages": len(messages)},
                metadata={"model": req.model},
            )
            chunks_yielded = 0
            total_length = 0
            answer_parts = []

            for chunk in completion(
                model=get_model_name(req.model),
                messages=messages,
                temperature=req.temperature,
                stream=True,
            ):
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    chunks_yielded += 1
                    total_length += len(delta)
                    answer_parts.append(delta)
                    yield delta.encode("utf-8")

            full_answer = "".join(answer_parts)
            llm_span.end(
                output={
                    "chunks_yielded": chunks_yielded,
                    "total_length": total_length,
                    "answer": full_answer[:500] if len(full_answer) > 500 else full_answer,
                },
                model=req.model,
                provider="litellm",
            )

            trace.end(
                output={
                    "answer": full_answer[:500] if len(full_answer) > 500 else full_answer,
                    "answer_length": total_length,
                    "chunks_yielded": chunks_yielded,
                    "num_chunks_used": len(hits),
                },
            )

        except RateLimitError as e:
            logger.error(f"OpenRouter rate limit exceeded for model {req.model}: {e}")
            trace.end(output={"error": str(e)}, metadata={"error_type": "RateLimitError"})
            yield f"\n\n[ERROR] Rate limit exceeded. Please try again later.".encode("utf-8")
        except AuthenticationError as e:
            logger.error(f"OpenRouter authentication failed for model {req.model}: {e}")
            trace.end(output={"error": str(e)}, metadata={"error_type": "AuthenticationError"})
            yield f"\n\n[ERROR] API authentication failed. Please check your OpenRouter API key.".encode("utf-8")
        except InvalidRequestError as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg.lower() or "insufficient credits" in error_msg.lower():
                logger.error(f"Insufficient OpenRouter credits for model {req.model}: {e}")
                trace.end(output={"error": str(e)}, metadata={"error_type": "InsufficientQuota"})
                yield f"\n\n[ERROR] Insufficient API credits. Please add credits to your OpenRouter account.".encode("utf-8")
            else:
                logger.error(f"Invalid request for model {req.model}: {e}")
                trace.end(output={"error": error_msg}, metadata={"error_type": "InvalidRequestError"})
                yield f"\n\n[ERROR] Invalid request: {error_msg}".encode("utf-8")
        except APIError as e:
            logger.error(f"OpenRouter API error for model {req.model}: {e}")
            trace.end(output={"error": str(e)}, metadata={"error_type": "APIError"})
            yield f"\n\n[ERROR] API service error: {str(e)}".encode("utf-8")
        except Exception as e:
            logger.exception(f"Unexpected error during streaming chat: {e}")
            trace.end(output={"error": str(e)}, metadata={"error_type": type(e).__name__})
            yield f"\n\n[ERROR] Generation failed: {str(e)}".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
