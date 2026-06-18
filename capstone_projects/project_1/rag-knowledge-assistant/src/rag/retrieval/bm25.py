"""BM25 sparse retriever.

Dense (embedding) search is great at *semantics* but blind to exact tokens it
never saw in training — error codes, SKUs, policy numbers, acronyms. BM25 is a
lexical bag-of-words scorer that nails those. Running both and fusing them is
why hybrid search beats either alone.

This in-memory index is rebuilt from the store on startup. For very large
corpora you would push sparse retrieval into the engine itself (Elasticsearch /
OpenSearch / Postgres full-text), but the fusion logic downstream is identical.
"""
from __future__ import annotations

import re

from ..logging_config import get_logger
from ..models import Chunk, ScoredChunk

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self._corpus_tokens = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._corpus_tokens)
        logger.info("bm25 index built", extra={"extra": {"docs": len(chunks)}})

    def search(
        self, query: str, top_k: int, category: str | None = None
    ) -> list[ScoredChunk]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        out: list[ScoredChunk] = []
        for i in ranked:
            chunk = self.chunks[i]
            if category and chunk.category != category:
                continue
            if scores[i] <= 0:
                continue
            out.append(
                ScoredChunk(chunk=chunk, score=float(scores[i]), retriever="sparse")
            )
            if len(out) >= top_k:
                break
        return out
