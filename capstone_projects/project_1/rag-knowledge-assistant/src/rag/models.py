"""Domain schemas shared across the pipeline and the API.

Pydantic models give us validation at the edges (API in/out) and self-documenting
internal contracts. Keeping them in one place avoids the dict-passing soup that
makes RAG pipelines hard to reason about.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document before chunking."""

    doc_id: str
    title: str
    source: str            # e.g. "hr-policy/leave.md"
    category: str          # hr | it | security | engineering
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit. `chunk_id` is stable so re-ingesting is idempotent."""

    chunk_id: str
    doc_id: str
    title: str
    source: str
    category: str
    text: str
    position: int          # ordinal within the parent document
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    # Where the candidate came from, for debuggability of the hybrid path.
    retriever: str = "unknown"  # dense | sparse | fused | reranked


class Citation(BaseModel):
    chunk_id: str
    title: str
    source: str
    snippet: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_n: int | None = None
    category: str | None = None   # optional metadata filter
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
    cached: bool
    request_id: str
    timings_ms: dict[str, float]
