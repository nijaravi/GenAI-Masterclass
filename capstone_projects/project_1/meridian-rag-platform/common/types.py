"""Data shapes for the API and monitoring.

Internally the pipeline now passes LangChain `Document` objects around (that's
what the loaders, splitters, vector store, and retrievers all speak). These
Pydantic models are just the API contract (request/response) and the monitoring
trace — the boundary types that aren't LangChain's job.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------- API request / response (frontend <-> backend) ----------
class Citation(BaseModel):
    title: str
    source: str
    snippet: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    category: Optional[str] = None   # optional filter: hr | it | security | engineering
    use_cache: bool = True


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
    cached: bool
    request_id: str
    blocked: bool = False
    guardrail_notes: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


# ---------- monitoring ----------
class Trace(BaseModel):
    """One row per request — the record monitoring, cost, and eval all read."""
    request_id: str
    user_id: str
    question: str
    answer: str
    model: str
    cached: bool
    blocked: bool
    guardrail_notes: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    relevance_score: Optional[int] = None
    faithfulness_score: Optional[int] = None
    created_at: str = ""
    context_used: str = ""
