"""Small utilities for context building and message assembly."""

from typing import Dict, List


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
        seg = f"[{i}] {t}\n"
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
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Use the provided CONTEXT when helpful. "
                "If the answer is not present or unclear, say you don't know. Be concise."
            ),
        },
        {"role": "system", "content": f"CONTEXT:\n{context}"},
    ]
    msgs.extend(history[-8:])  # keep recent turns small
    msgs.append({"role": "user", "content": question})
    return msgs
