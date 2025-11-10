"""Pydantic request/response models for the chat API."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class Msg(BaseModel):
    """Represents a single chat message with role and content."""
    role: Role = Field(..., description="Message role: 'user' or 'assistant'.")
    content: str = Field(..., description="Message text content.")


class ChatRequest(BaseModel):
    """Input payload for chat endpoints with retrieval and generation options."""
    question: str = Field(..., description="User's question to answer.")
    history: List[Msg] = Field(default_factory=list, description="Recent chat history as a list of messages.")
    k: int = Field(5, description="Number of nearest chunks to retrieve.")
    model: str = Field("openai/gpt-4o-mini", description="LiteLLM model identifier, e.g. 'openai/gpt-4o-mini'.")
    temperature: float = Field(0.2, description="Sampling temperature for generation.")
    stream: bool = Field(False, description="Whether to stream the response.")
    use_hyde: bool = Field(False, description="Whether to enable HYDE query expansion retrieval.")
    rerank: Literal["none", "cheap", "llm"] = Field("none", description="Reranker choice: none, cheap, or llm.")
    rerank_top_n: Optional[int] = Field(None, description="Top-N after reranking; defaults to k if unset.")


class ChatResponse(BaseModel):
    """Non-streaming chat response with the finalized answer text."""
    answer: str = Field(..., description="LLM-generated final answer text.")
