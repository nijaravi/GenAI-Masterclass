"""Centralised, validated configuration.

Everything tunable lives here and is driven by environment variables (or a
`.env` file) so the same image runs unchanged in dev, CI, and prod — only the
env differs. This is the single source of truth: no module reads os.environ
directly, they all import `settings`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- App ----
    app_name: str = "meridian-rag-assistant"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True  # structured JSON logs in prod; pretty in dev

    # ---- LLM providers (resume: GPT-4o / GPT-4o-mini + Llama via Groq) ----
    openai_api_key: str = Field(default="", repr=False)
    groq_api_key: str = Field(default="", repr=False)

    # Model routing: "fast" handles routine queries, "smart" handles complex ones.
    fast_model: str = "gpt-4o-mini"
    smart_model: str = "gpt-4o"
    groq_model: str = "llama-3.1-8b-instant"
    # Route to Groq for cheap/high-throughput paths when a key is present.
    prefer_groq_for_fast: bool = False

    # ---- Embeddings ----
    # "openai" uses text-embedding-3-small; "local" uses sentence-transformers.
    embedding_provider: Literal["openai", "local"] = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 1536  # 1536 for OpenAI small; 384 for MiniLM

    # ---- Vector store (resume: ChromaDB / pgvector) ----
    vector_store: Literal["chroma", "pgvector"] = "chroma"
    chroma_path: str = "./.chroma"
    chroma_collection: str = "meridian_docs"
    # pgvector
    pg_dsn: str = "postgresql://rag:rag@localhost:5432/rag"
    pg_table: str = "doc_chunks"

    # ---- Chunking ----
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64
    tokenizer_encoding: str = "cl100k_base"

    # ---- Retrieval / hybrid search ----
    dense_top_k: int = 20         # candidates from vector search
    sparse_top_k: int = 20        # candidates from BM25
    rrf_k: int = 60               # Reciprocal Rank Fusion constant
    fused_top_k: int = 12         # candidates passed to reranker
    rerank_top_n: int = 5         # final chunks sent to the LLM
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- Semantic cache ----
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.95
    cache_max_entries: int = 1000

    # ---- Generation ----
    max_context_chars: int = 12_000  # guard against context overflow
    temperature: float = 0.1


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
