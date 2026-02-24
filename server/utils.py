"""Small utilities for context building and message assembly."""

import os
from pathlib import Path
from typing import Dict, List

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "system_prompts"


def _load_prompt(name: str) -> str:
    """Read a prompt file from the system_prompts/ directory."""
    return (_PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def get_model_name(model: str) -> str:
    """
    Convert model name to OpenRouter format if OPENROUTER_API_KEY is set.

    Args:
        model: Model identifier (e.g., 'openai/gpt-4o-mini' or 'gpt-4.1-nano')

    Returns:
        Model identifier in OpenRouter format or original model name.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        return model

    if model.startswith("openai/"):
        return f"openrouter/{model}"

    if model.startswith("gpt-"):
        return f"openrouter/openai/{model}"

    if model.startswith("openrouter/"):
        return model

    return model


def join_context(hits: List[Dict], cap_chars: int = 1800) -> str:
    """
    Builds a capped context string from retrieval hits; returns a string under cap_chars.

    Args:
        hits (List[Dict]): List of retrieval hit dictionaries containing 'text' field.
        cap_chars (int): Maximum character limit for the output context string. Defaults to 1800.

    Returns:
        str: Formatted context string with numbered segments, or "(no relevant context found)" if empty.
    """
    out, used = [], 0
    for i, h in enumerate(hits, 1):
        t = (h.get("text") or "").strip()
        if not t:
            continue

        # Append links if available
        links_text = ""
        links = h.get("links")
        if links:
            link_parts = []
            if links.get("live"):
                link_parts.append(f"Live: {links['live']}")
            if links.get("github"):
                link_parts.append(f"GitHub: {links['github']}")
            if link_parts:
                links_text = f" [{', '.join(link_parts)}]"

        seg = f"[{i}] {t}{links_text}\n"
        if used + len(seg) > cap_chars:
            break
        out.append(seg)
        used += len(seg)
    return "".join(out) if out else "(no relevant context found)"


def build_messages(question: str, history: List[Dict], context: str) -> List[Dict]:
    """
    Assembles OpenAI-compatible chat messages list from system, context, history, and user question.

    Args:
        question (str): User's question to be added as the final message.
        history (List[Dict]): List of previous chat messages with 'role' and 'content' fields.
        context (str): Retrieved context information to include in system message.

    Returns:
        List[Dict]: OpenAI-compatible messages list with system prompts, context, history, and user question.
    """
    msgs: List[Dict] = [
        {"role": "system", "content": _load_prompt("generation.txt")},
        {"role": "system", "content": f"CONTEXT:\n{context}"},
    ]
    msgs.extend(history[-8:])  # keep recent turns small
    msgs.append({"role": "user", "content": question})
    return msgs
