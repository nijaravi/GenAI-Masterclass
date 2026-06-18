"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

The problem with naively combining dense + sparse: their scores live on totally
different scales (cosine in [0,1] vs unbounded BM25). Normalising them is
fragile. RRF sidesteps this entirely by fusing on *rank*, not score:

    rrf_score(d) = sum over retrievers of  1 / (k + rank_r(d))

A document ranked highly by either retriever floats up; one ranked highly by
both wins. `k` (default 60, from the original Cormack et al. paper) damps the
influence of very deep ranks. It's simple, parameter-light, and consistently
strong — which is exactly why it's the default fusion method in production
hybrid search.
"""
from __future__ import annotations

from ..config import settings
from ..logging_config import get_logger
from ..models import ScoredChunk

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]],
    k: int = settings.rrf_k,
    top_k: int = settings.fused_top_k,
) -> list[ScoredChunk]:
    fused: dict[str, float] = {}
    chunk_by_id: dict[str, ScoredChunk] = {}

    for results in result_lists:
        for rank, sc in enumerate(results):
            cid = sc.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
            # Keep one representative ScoredChunk per id for return.
            chunk_by_id.setdefault(cid, sc)

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    out: list[ScoredChunk] = []
    for cid, score in ranked[:top_k]:
        rep = chunk_by_id[cid]
        out.append(
            ScoredChunk(chunk=rep.chunk, score=score, retriever="fused")
        )
    logger.info(
        "rrf fused",
        extra={"extra": {"inputs": [len(r) for r in result_lists], "out": len(out)}},
    )
    return out
