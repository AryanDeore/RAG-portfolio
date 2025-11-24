"""FastAPI routes exposing non-stream and streaming chat endpoints."""

import logging
from opik import Opik
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Iterator, List, Dict
from litellm import completion
from litellm.exceptions import (
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    APIError,
)
from server.schemas import ChatRequest, ChatResponse
from server.utils import join_context, build_messages, get_provider_from_model, get_model_name_for_opik
from server.retrieval_pipeline import (
    retrieve_knn,
    retrieve_with_hyde,
    cheap_rerank,
    llm_rerank,
)

# Set up logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Create Opik client instance
opik_client = Opik()


def _get_model_name(model: str) -> str:
    """
    Convert model name to OpenRouter format if OPENROUTER_API_KEY is set.
    
    Args:
        model: Model identifier (e.g., 'openai/gpt-4o-mini')
    
    Returns:
        Model identifier in OpenRouter format (e.g., 'openrouter/openai/gpt-4o-mini')
        or original model name if OpenRouter is not configured.
    """
    import os
    if os.getenv("OPENROUTER_API_KEY") and model.startswith("openai/"):
        # Convert openai/model to openrouter/openai/model
        return f"openrouter/{model}"
    return model


def _choose_retrieval(req: ChatRequest, parent_span=None) -> List[Dict]:
    """
    Chooses KNN or HYDE retrieval based on request flags and returns a list of hit dicts.
    
    Args:
        req (ChatRequest): Chat request containing retrieval configuration and question.
        parent_span: Optional parent span for nested tracing.
    
    Returns:
        List[Dict]: List of hit dictionaries containing id, score, doc_id, chunk_id, title, and text fields.
    """
    if req.use_hyde:
        return retrieve_with_hyde(req.question, k=req.k, hyde_model=req.model, parent_span=parent_span)
    return retrieve_knn(req.question, k=req.k, parent_span=parent_span)


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
async def chat(req: ChatRequest) -> JSONResponse:
    """
    Handles non-streaming chat by retrieving context and returning a complete answer.
    
    Args:
        req (ChatRequest): Chat request containing question, history, retrieval options, and model parameters.
    
    Returns:
        JSONResponse: Response containing ChatResponse with the complete generated answer text.
    """
    # Create trace
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
        tags=["rag", "chat", "non-streaming"],
        metadata={
            "endpoint": "/chat",
        }
    )
    
    try:
        # Span for retrieval
        retrieval_span = trace.span(
            name="retrieve_chunks",
            type="tool",
            input={"question": req.question, "k": req.k, "use_hyde": req.use_hyde}
        )
        hits = _choose_retrieval(req, parent_span=retrieval_span)
        hits = _maybe_rerank(req, hits, parent_span=retrieval_span)
        retrieval_span.end(
            output={
                "num_chunks": len(hits),
                "chunk_ids": [h.get("id") for h in hits],
                "chunk_scores": [h.get("score") for h in hits],
                "doc_ids": list(set(h.get("doc_id") for h in hits if h.get("doc_id"))),
            },
            metadata={
                "retrieval_method": "hyde" if req.use_hyde else "knn",
                "rerank_method": req.rerank,
            }
        )

        # Span for context building
        context_span = trace.span(
            name="build_context",
            type="general",
            input={"num_hits": len(hits)}
        )
        context = join_context(hits)
        history = [m.model_dump() for m in req.history]
        messages = build_messages(req.question, history, context)
        context_span.end(
            output={
                "context_length": len(context),
                "num_messages": len(messages),
            }
        )

        # Span for LLM completion
        llm_span = trace.span(
            name="llm_completion",
            type="llm",
            input={
                "model": req.model,
                "temperature": req.temperature,
                "num_messages": len(messages),
            },
            metadata={"model": req.model}
        )
        resp = completion(
            model=_get_model_name(req.model),
            messages=messages,
            temperature=req.temperature,
            stream=False,
        )
        text = resp["choices"][0]["message"]["content"]
        
        # Extract usage
        usage = resp.get("usage", {})
        provider = get_provider_from_model(req.model)
        model_name = get_model_name_for_opik(req.model)
        
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
            model=model_name,
            provider=provider,
        )

        # End trace with output
        trace.end(
            output={
                "answer": text[:500] if len(text) > 500 else text,  # Truncate for logging
                "answer_length": len(text),
                "num_chunks_used": len(hits),
            }
        )
        
        return JSONResponse(ChatResponse(answer=text).model_dump())
        
    except RateLimitError as e:
        logger.error(f"OpenRouter rate limit exceeded for model {req.model}: {e}")
        trace.end(
            output={"error": str(e)},
            metadata={"error_type": "RateLimitError"}
        )
        raise HTTPException(
            status_code=429,
            detail="API rate limit exceeded. Please try again later."
        ) from e
    except AuthenticationError as e:
        logger.error(f"Authentication failed for model {req.model}: {e}")
        trace.end(
            output={"error": str(e)},
            metadata={"error_type": "AuthenticationError"}
        )
        raise HTTPException(
            status_code=401,
            detail="API authentication failed. Please check your OpenRouter API key."
        ) from e
    except InvalidRequestError as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower() or "insufficient credits" in error_msg.lower():
            logger.error(f"Insufficient OpenRouter credits for model {req.model}: {e}")
            trace.end(
                output={"error": str(e)},
                metadata={"error_type": "InsufficientQuota"}
            )
            raise HTTPException(
                status_code=402,
                detail="Insufficient API credits. Please add credits to your OpenRouter account."
            ) from e
        logger.error(f"Invalid OpenRouter request for model {req.model}: {e}")
        trace.end(
            output={"error": error_msg},
            metadata={"error_type": "InvalidRequestError"}
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {error_msg}"
        ) from e
    except APIError as e:
        logger.error(f"OpenRouter API error for model {req.model}: {e}")
        trace.end(
            output={"error": str(e)},
            metadata={"error_type": "APIError"}
        )
        raise HTTPException(
            status_code=503,
            detail=f"API service error: {str(e)}"
        ) from e
    except Exception as e:
        logger.exception(f"Unexpected error during chat generation: {e}")
        trace.end(
            output={"error": str(e)},
            metadata={"error_type": type(e).__name__}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        ) from e


@router.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    Handles streaming chat by yielding incremental tokens as plain text.
    
    Args:
        req (ChatRequest): Chat request containing question, history, retrieval options, and model parameters.
    
    Returns:
        StreamingResponse: Streaming response that yields UTF-8 encoded text chunks as they are generated.
    """
    # Create trace
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
        tags=["rag", "chat", "streaming"],
        metadata={
            "endpoint": "/chat/stream",
        }
    )
    
    def gen() -> Iterator[bytes]:
        """
        Streams LLM deltas as UTF-8 encoded bytes for chunked transfer.
        
        Returns:
            Iterator[bytes]: Generator yielding UTF-8 encoded text chunks from the LLM stream.
        """
        try:
            # Span for retrieval
            retrieval_span = trace.span(
                name="retrieve_chunks",
                type="tool",
                input={"question": req.question, "k": req.k, "use_hyde": req.use_hyde}
            )
            hits = _choose_retrieval(req, parent_span=retrieval_span)
            hits = _maybe_rerank(req, hits, parent_span=retrieval_span)
            retrieval_span.end(
                output={
                    "num_chunks": len(hits),
                    "chunk_ids": [h.get("id") for h in hits],
                    "chunk_scores": [h.get("score") for h in hits],
                    "doc_ids": list(set(h.get("doc_id") for h in hits if h.get("doc_id"))),
                },
                metadata={
                    "retrieval_method": "hyde" if req.use_hyde else "knn",
                    "rerank_method": req.rerank,
                }
            )

            # Span for context building
            context_span = trace.span(
                name="build_context",
                type="general",
                input={"num_hits": len(hits)}
            )
            context = join_context(hits)
            history = [m.model_dump() for m in req.history]
            messages = build_messages(req.question, history, context)
            context_span.end(
                output={
                    "context_length": len(context),
                    "num_messages": len(messages),
                }
            )

            # Span for streaming LLM completion
            llm_span = trace.span(
                name="llm_completion_stream",
                type="llm",
                input={
                    "model": req.model,
                    "temperature": req.temperature,
                    "num_messages": len(messages),
                },
                metadata={"model": req.model}
            )
            chunks_yielded = 0
            total_length = 0
            answer_parts = []

            for chunk in completion(
                model=_get_model_name(req.model),
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
            
            # Update span with streaming results
            full_answer = "".join(answer_parts)
            llm_span.end(
                output={
                    "chunks_yielded": chunks_yielded,
                    "total_length": total_length,
                    "answer": full_answer[:500] if len(full_answer) > 500 else full_answer,  # Truncate
                },
                model=req.model,
                provider="litellm"
            )
            
            # Update trace output
            trace.end(
                output={
                    "answer": full_answer[:500] if len(full_answer) > 500 else full_answer,
                    "answer_length": total_length,
                    "chunks_yielded": chunks_yielded,
                    "num_chunks_used": len(hits),
                }
            )
            
        except RateLimitError as e:
            logger.error(f"OpenRouter rate limit exceeded for model {req.model}: {e}")
            trace.end(
                output={"error": str(e)},
                metadata={"error_type": "RateLimitError"}
            )
            yield f"\n\n[ERROR] Rate limit exceeded. Please try again later.".encode("utf-8")
        except AuthenticationError as e:
            logger.error(f"OpenRouter authentication failed for model {req.model}: {e}")
            trace.end(
                output={"error": str(e)},
                metadata={"error_type": "AuthenticationError"}
            )
            yield f"\n\n[ERROR] API authentication failed. Please check your OpenRouter API key.".encode("utf-8")
        except InvalidRequestError as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg.lower() or "insufficient credits" in error_msg.lower():
                logger.error(f"Insufficient OpenRouter credits for model {req.model}: {e}")
                trace.end(
                    output={"error": str(e)},
                    metadata={"error_type": "InsufficientQuota"}
                )
                yield f"\n\n[ERROR] Insufficient API credits. Please add credits to your OpenRouter account.".encode("utf-8")
            else:
                logger.error(f"Invalid request for model {req.model}: {e}")
                trace.end(
                    output={"error": error_msg},
                    metadata={"error_type": "InvalidRequestError"}
                )
                yield f"\n\n[ERROR] Invalid request: {error_msg}".encode("utf-8")
        except APIError as e:
            logger.error(f"OpenRouter API error for model {req.model}: {e}")
            trace.end(
                output={"error": str(e)},
                metadata={"error_type": "APIError"}
            )
            yield f"\n\n[ERROR] API service error: {str(e)}".encode("utf-8")
        except Exception as e:
            logger.exception(f"Unexpected error during streaming chat: {e}")
            trace.end(
                output={"error": str(e)},
                metadata={"error_type": type(e).__name__}
            )
            yield f"\n\n[ERROR] Generation failed: {str(e)}".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")