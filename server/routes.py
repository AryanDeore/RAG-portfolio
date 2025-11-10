"""FastAPI routes exposing non-stream and streaming chat endpoints."""

import logging
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
from server.utils import join_context, build_messages
from server.retrieval_pipeline import (
    retrieve_knn,
    retrieve_with_hyde,
    cheap_rerank,
    llm_rerank,
)

# Set up logger
logger = logging.getLogger(__name__)

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
    except RateLimitError as e:
        logger.error(f"OpenRouter rate limit exceeded for model {req.model}: {e}")
        raise HTTPException(
            status_code=429,
            detail="API rate limit exceeded. Please try again later."
        ) from e
    except AuthenticationError as e:
        logger.error(f"Authentication failed for model {req.model}: {e}")
        raise HTTPException(
            status_code=401,
            detail="API authentication failed. Please check your OpenRouter API key."
        ) from e
    except InvalidRequestError as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower() or "insufficient credits" in error_msg.lower():
            logger.error(f"Insufficient OpenRouter credits for model {req.model}: {e}")
            raise HTTPException(
                status_code=402,
                detail="Insufficient API credits. Please add credits to your OpenRouter account."
            ) from e
        logger.error(f"Invalid OpenRouter request for model {req.model}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {error_msg}"
        ) from e
    except APIError as e:
        logger.error(f"OpenRouter API error for model {req.model}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"API service error: {str(e)}"
        ) from e
    except Exception as e:
        logger.exception(f"Unexpected error during chat generation: {e}")
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
    def gen() -> Iterator[bytes]:
        """
        Streams LLM deltas as UTF-8 encoded bytes for chunked transfer.
        
        Returns:
            Iterator[bytes]: Generator yielding UTF-8 encoded text chunks from the LLM stream.
        """
        try:
            hits = _choose_retrieval(req)
            hits = _maybe_rerank(req, hits)
            context = join_context(hits)
            history = [m.model_dump() for m in req.history]
            messages = build_messages(req.question, history, context)

            for chunk in completion(
                model=req.model,
                messages=messages,
                temperature=req.temperature,
                stream=True,
            ):
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta.encode("utf-8")
        except RateLimitError as e:
            logger.error(f"OpenRouter rate limit exceeded for model {req.model}: {e}")
            yield f"\n\n[ERROR] Rate limit exceeded. Please try again later.".encode("utf-8")
        except AuthenticationError as e:
            logger.error(f"OpenRouter authentication failed for model {req.model}: {e}")
            yield f"\n\n[ERROR] API authentication failed. Please check your OpenRouter API key.".encode("utf-8")
        except InvalidRequestError as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg.lower() or "insufficient credits" in error_msg.lower():
                logger.error(f"Insufficient OpenRouter credits for model {req.model}: {e}")
                yield f"\n\n[ERROR] Insufficient API credits. Please add credits to your OpenRouter account.".encode("utf-8")
            else:
                logger.error(f"Invalid request for model {req.model}: {e}")
                yield f"\n\n[ERROR] Invalid request: {error_msg}".encode("utf-8")
        except APIError as e:
            logger.error(f"OpenRouter API error for model {req.model}: {e}")
            yield f"\n\n[ERROR] API service error: {str(e)}".encode("utf-8")
        except Exception as e:
            logger.exception(f"Unexpected error during streaming chat: {e}")
            yield f"\n\n[ERROR] Generation failed: {str(e)}".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
