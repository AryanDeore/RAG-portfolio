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
                "You are \"Aryan's Portfolio Assistant.\"\n\n"
                "Scope:\n"
                "- Answer ONLY using Aryan's portfolio data from the CONTEXT.\n"
                "- Assume \"I\", \"me\", \"my\", and any unnamed \"the project/role/company\" refer to Aryan.\n"
                "- If the user mentions a different person, treat it as a comparison request (Aryan vs X).\n\n"
                "Behavior:\n"
                "- Be concise and factual. Prefer enumerated lists.\n"
                "- If the query is vague but clearly portfolio-scoped, default to brief summaries (then offer drill-downs).\n"
                "- When listing items, count them first: 'I have X projects:' then list them.\n"
                "- If you cannot find the requested information in CONTEXT, explicitly state that.\n\n"
                "Assistant style guide:\n"
                "- Start with a one-line answer.\n"
                "- Then provide a tight bullet list or mini-cards.\n"
                "- End with 2–3 smart follow-ups (drill-down options).\n\n"
                "Data context:\n"
                "- Source of truth: contents.json (bio, projects, experience, skills, education).\n"
                "- The CONTEXT section below contains the ONLY data you can use.\n"
                "- When uncertain, state what you can't find and propose close matches from the CONTEXT."
            ),
        },
        {"role": "system", "content": f"CONTEXT:\n{context}"},
    ]
    msgs.extend(history[-8:])  # keep recent turns small
    msgs.append({"role": "user", "content": question})
    return msgs


def get_provider_from_model(model: str) -> str:
    """
    Extract provider name from model identifier for Opik cost tracking.
    Handles formats like:
    - "openrouter/openai/gpt-4o-mini" -> "openai"
    - "openrouter/anthropic/claude-3-opus" -> "anthropic"
    - "openai/gpt-4o-mini" -> "openai"
    - "anthropic/claude-3-opus" -> "anthropic"
    """
    # Handle OpenRouter format: "openrouter/{provider}/{model}"
    if model.startswith("openrouter/"):
        parts = model.split("/")
        if len(parts) >= 2:
            provider = parts[1].lower()
            # Normalize to Opik's expected provider names
            provider_map = {
                "google": "google_ai",  # Opik uses "google_ai" not "google"
                "google-ai": "google_ai",
                "google_ai": "google_ai",
            }
            return provider_map.get(provider, provider)
    
    # Handle direct format: "{provider}/{model}"
    if "/" in model:
        provider = model.split("/")[0].lower()
        # Normalize provider names
        provider_map = {
            "google": "google_ai",
        }
        return provider_map.get(provider, provider)
    
    # Try to infer from model name
    model_lower = model.lower()
    if "gpt" in model_lower or "openai" in model_lower:
        return "openai"
    elif "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "google_ai"
    
    # Default fallback - use "openai" instead of "unknown"
    return "openai"


def get_model_name_for_opik(model: str) -> str:
    """
    Extract clean model name from model identifier for Opik cost tracking.
    Handles formats like:
    - "openrouter/openai/gpt-4o-mini" -> "gpt-4o-mini"
    - "openrouter/anthropic/claude-3-opus" -> "claude-3-opus"
    - "openai/gpt-4o-mini" -> "gpt-4o-mini"
    """
    # Handle OpenRouter format: "openrouter/{provider}/{model}"
    if model.startswith("openrouter/"):
        parts = model.split("/")
        if len(parts) >= 3:
            return parts[2]  # Returns "gpt-4o-mini", "claude-3-opus", etc.
        elif len(parts) >= 2:
            return parts[1]  # Fallback
    
    # Handle direct format: "{provider}/{model}"
    if "/" in model:
        return model.split("/")[-1]  # Returns model name
    
    # If no separator, return as-is
    return model
