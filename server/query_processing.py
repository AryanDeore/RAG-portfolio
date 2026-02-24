"""Query preprocessing: moderation gate and query decomposition/expansion."""

import json
import logging
import os
from typing import List, Optional, Any

from litellm import completion
from server.utils import _load_prompt, get_model_name

logger = logging.getLogger(__name__)


def moderate_query(query: str, parent_span: Optional[Any] = None) -> bool:
    """
    Check query against OpenAI moderation endpoint.

    Requires OPENAI_API_KEY to be set. If not set, skips moderation (fail-open).

    Args:
        query: User query string.
        parent_span: Optional Opik parent span for tracing.

    Returns:
        True if flagged (should be blocked), False if safe.
    """
    span = None
    if parent_span:
        span = parent_span.span(
            name="moderation_check",
            type="tool",
            input={"query": query},
        )

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_MODERATION_API_KEY")
    if not api_key:
        logger.info("No OpenAI API key set — skipping moderation check")
        if span:
            span.end(output={"flagged": False, "skipped": True, "reason": "no OpenAI API key"})
        return False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=query,
        )
        flagged = response.results[0].flagged

        if span:
            span.end(
                output={
                    "flagged": flagged,
                    "categories": {
                        k: v for k, v in response.results[0].categories.model_dump().items() if v
                    } if flagged else {},
                },
            )
        return flagged

    except Exception as e:
        logger.warning("Moderation check failed (fail-open): %s", e)
        if span:
            span.end(output={"flagged": False, "error": str(e)})
        return False


def decompose_and_expand(
    query: str,
    model: str,
    parent_span: Optional[Any] = None,
) -> List[str]:
    """
    Decompose multi-intent queries and expand each sub-query with synonyms.

    Args:
        query: Original user query.
        model: LiteLLM model identifier for the expansion LLM call.
        parent_span: Optional Opik parent span for tracing.

    Returns:
        List of expanded sub-query strings. Falls back to [query] on error.
    """
    span = None
    if parent_span:
        span = parent_span.span(
            name="decompose_expand",
            type="llm",
            input={"original_query": query},
            metadata={"model": model},
        )

    try:
        system_prompt = _load_prompt("expansion_decomposition.txt")
        resp = completion(
            model=get_model_name(model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            stream=False,
        )

        content = resp["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)
        sub_queries = parsed.get("sub_queries", [query])

        if not sub_queries:
            sub_queries = [query]

        usage = resp.get("usage", {})
        was_decomposed = len(sub_queries) > 1

        if span:
            span.end(
                output={
                    "sub_queries": sub_queries,
                    "num_sub_queries": len(sub_queries),
                    "was_decomposed": was_decomposed,
                },
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                } if usage else None,
                model=model,
                provider="litellm",
            )

        return sub_queries

    except Exception as e:
        logger.warning("Decompose/expand failed (falling back to original query): %s", e)
        if span:
            span.end(
                output={
                    "sub_queries": [query],
                    "num_sub_queries": 1,
                    "was_decomposed": False,
                    "error": str(e),
                },
            )
        return [query]
