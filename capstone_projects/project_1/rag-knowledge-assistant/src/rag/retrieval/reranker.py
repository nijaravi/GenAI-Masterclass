"""Cross-encoder reranking.

Retrieval (dense/sparse) scores the query and each chunk *independently* — fast,
but it never lets the model look at the pair together. A cross-encoder does:
it takes (query, chunk) jointly and outputs a relevance score, which is far more
accurate at separating "mentions the keyword" from "actually answers this".

The cost is latency, so we apply it only to the ~12 fused candidates, not the
whole corpus. This retrieve-wide-then-rerank-narrow pattern is the standard way
to get both recall and precision. We fall back to identity (pass-through) if the
model can't load, so the service degrades instead of crashing.
"""
from __future__ import annotations

from ..config import settings
from ..logging_config import get_logger
from ..models import ScoredChunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self._model = None
        self._model_name = model_name or settings.reranker_model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info("reranker loaded", extra={"extra": {"model": self._model_name}})
        except Exception as e:  # pragma: no cover
            logger.warning(
                "reranker unavailable, passing through",
                extra={"extra": {"error": str(e)}},
            )

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        if self._model is None:
            return candidates[:top_n]

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        scored = [
            ScoredChunk(chunk=c.chunk, score=float(s), retriever="reranked")
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]
