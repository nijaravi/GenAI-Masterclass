"""Semantic cache.

An exact-string cache misses "what's the leave policy?" vs "how much annual
leave do I get?" — same intent, different words. A semantic cache embeds the
query and serves a stored answer when cosine similarity to a past query clears
a high threshold (0.95 by default — deliberately strict, since a wrong cache
hit returns a confidently wrong answer).

This is a meaningful cost/latency lever: cached hits skip retrieval, rerank, and
the LLM call entirely. Here it's an in-process LRU; in production you'd back it
with Redis (e.g. RedisVL) so it's shared across replicas.
"""
from __future__ import annotations

import time
from collections import OrderedDict

from ..config import settings
from ..logging_config import get_logger

logger = get_logger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


class SemanticCache:
    def __init__(self) -> None:
        # chunk_id -> (embedding, payload, ts)
        self._store: "OrderedDict[str, tuple[list[float], dict, float]]" = (
            OrderedDict()
        )
        self.threshold = settings.cache_similarity_threshold
        self.max_entries = settings.cache_max_entries

    def get(self, query_embedding: list[float]) -> dict | None:
        best_key, best_sim = None, -1.0
        for key, (emb, _payload, _ts) in self._store.items():
            sim = _cosine(query_embedding, emb)
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_key is not None and best_sim >= self.threshold:
            self._store.move_to_end(best_key)  # LRU touch
            logger.info("cache hit", extra={"extra": {"similarity": round(best_sim, 4)}})
            return self._store[best_key][1]
        return None

    def put(self, query: str, query_embedding: list[float], payload: dict) -> None:
        self._store[query] = (query_embedding, payload, time.time())
        self._store.move_to_end(query)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)  # evict oldest
