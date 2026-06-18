"""Vector store interface.

The retrieval layer depends only on this Protocol, never on a concrete store.
That is what makes "ChromaDB / pgvector" a config flag rather than a rewrite:
both implementations satisfy the same contract.
"""
from __future__ import annotations

from typing import Protocol

from ..models import Chunk, ScoredChunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Idempotent insert keyed on chunk_id."""

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        """Return top_k nearest chunks by cosine similarity (higher = closer)."""

    def count(self) -> int: ...
