"""Configuration for the whole platform.

Everything tunable lives here and is read from environment variables (or a
`.env` file). No other module reads os.environ directly — they all import
`settings` from here. This is the single source of truth.

Style note: this is deliberately a plain settings object. Read it top to bottom;
there's nothing clever to chase.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- App ----
    app_name: str = "meridian-rag-platform"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"

    # ---- LLM providers (OpenAI + Groq only) ----
    # If BOTH keys are empty the platform automatically uses a local stub model
    # so everything still runs end to end (no real answers, but the flow works).
    openai_api_key: str = Field(default="", repr=False)
    groq_api_key: str = Field(default="", repr=False)

    # Model routing: a "fast" model for routine questions, a "smart" one for
    # long / complex questions. Groq serves the cheap high-throughput path.
    fast_model: str = "gpt-4o-mini"
    smart_model: str = "gpt-4o"
    groq_model: str = "llama-3.1-8b-instant"
    prefer_groq_for_fast: bool = False
    temperature: float = 0.1

    # ---- Embeddings ----
    # "openai" -> text-embedding-3-small (needs a key).
    # "stub"   -> a deterministic local embedder (no key, for demo/tests).
    embedding_provider: Literal["openai", "stub"] = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536
    stub_embedding_dim: int = 256

    # ---- Vector store (ChromaDB, persistent on disk) ----
    chroma_path: str = "./.chroma"
    chroma_collection: str = "meridian_docs"

    # ---- Chunking ----
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # ---- Retrieval / hybrid search ----
    dense_top_k: int = 20      # candidates from vector (dense) search
    sparse_top_k: int = 20     # candidates from BM25 (sparse) search
    rrf_k: int = 60            # Reciprocal Rank Fusion constant
    fused_top_k: int = 12      # candidates kept after fusion -> sent to reranker
    rerank_top_n: int = 5      # final passages sent to the LLM
    use_reranker: bool = True

    # ---- Semantic cache ----
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.95
    cache_max_entries: int = 1000

    # ---- Generation ----
    max_context_chars: int = 12_000

    # ---- Gateway ----
    # Simple per-user rate limit: N requests per window_seconds.
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60

    # ---- Monitoring ----
    # SQLite file that stores one row per request (the "trace").
    trace_db_path: str = "./traces.db"

    # ---- Guardrail thresholds ----
    # Minimum fraction of answer words that must appear in the retrieved context
    # for the answer to count as "grounded" (cheap hallucination check).
    groundedness_min_overlap: float = 0.20


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
