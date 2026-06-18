"""End-to-end query pipeline.

This is the orchestrator the API calls. The flow, with timings captured at each
stage so latency regressions are observable in logs/metrics:

    query
      -> [semantic cache lookup]                (skip everything on hit)
      -> embed query
      -> dense search (vector store)  +  sparse search (BM25)   [parallel-able]
      -> RRF fusion
      -> cross-encoder rerank  -> top N
      -> build grounded prompt -> LLM (routed) -> answer + citations
      -> cache put

Every stage is swappable because each dependency is injected in __init__, which
also makes the whole thing trivial to unit-test with fakes.
"""
from __future__ import annotations

import time

from .cache.semantic_cache import SemanticCache
from .config import settings
from .generation.llm import LLMClient
from .generation.prompts import (
    SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
)
from .ingestion.pipeline import build_store
from .logging_config import get_logger, new_request_id, request_id_var
from .models import Citation, QueryResponse, ScoredChunk
from .retrieval.bm25 import BM25Retriever
from .retrieval.hybrid import reciprocal_rank_fusion
from .retrieval.reranker import CrossEncoderReranker

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(
        self,
        store=None,
        embedder=None,
        bm25: BM25Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm: LLMClient | None = None,
        cache: SemanticCache | None = None,
    ) -> None:
        from .embeddings.embedder import build_embedder

        self.store = store or build_store()
        self.embedder = embedder or build_embedder()
        self.bm25 = bm25  # built at startup from the corpus (see api.dependencies)
        self.reranker = reranker or (
            CrossEncoderReranker() if settings.use_reranker else None
        )
        self.llm = llm or LLMClient()
        self.cache = cache or (SemanticCache() if settings.cache_enabled else None)

    def query(
        self,
        query: str,
        top_n: int | None = None,
        category: str | None = None,
        use_cache: bool = True,
    ) -> QueryResponse:
        rid = new_request_id()
        timings: dict[str, float] = {}
        top_n = top_n or settings.rerank_top_n

        t0 = time.perf_counter()
        q_emb = self.embedder.embed_query(query)
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        # --- Semantic cache ---
        if self.cache and use_cache:
            hit = self.cache.get(q_emb)
            if hit is not None:
                return QueryResponse(
                    **hit, cached=True, request_id=rid,
                    timings_ms={**timings, "total_ms": timings["embed_ms"]},
                )

        # --- Hybrid retrieval ---
        t1 = time.perf_counter()
        dense = self.store.search(q_emb, settings.dense_top_k, category)
        sparse = (
            self.bm25.search(query, settings.sparse_top_k, category)
            if self.bm25 else []
        )
        fused = reciprocal_rank_fusion([dense, sparse])
        timings["retrieve_ms"] = (time.perf_counter() - t1) * 1000

        # --- Rerank ---
        t2 = time.perf_counter()
        final: list[ScoredChunk] = (
            self.reranker.rerank(query, fused, top_n)
            if self.reranker else fused[:top_n]
        )
        timings["rerank_ms"] = (time.perf_counter() - t2) * 1000

        if not final:
            answer = "I don't have that information in the available documents."
            resp = QueryResponse(
                answer=answer, citations=[], model="none", cached=False,
                request_id=rid, timings_ms={**timings, "total_ms": _sum(timings)},
            )
            return resp

        # --- Generation ---
        t3 = time.perf_counter()
        context = build_context_block(final, settings.max_context_chars)
        user_prompt = build_user_prompt(query, context)
        llm_out = self.llm.generate(SYSTEM_PROMPT, user_prompt, query)
        timings["generate_ms"] = (time.perf_counter() - t3) * 1000
        timings["total_ms"] = _sum(timings)

        citations = [
            Citation(
                chunk_id=sc.chunk.chunk_id,
                title=sc.chunk.title,
                source=sc.chunk.source,
                snippet=sc.chunk.text[:240],
            )
            for sc in final
        ]
        payload = dict(
            answer=llm_out.text, citations=citations, model=llm_out.model
        )
        if self.cache and use_cache:
            self.cache.put(query, q_emb, payload)

        logger.info(
            "query served",
            extra={"extra": {"timings_ms": timings, "candidates": len(final)}},
        )
        return QueryResponse(
            **payload, cached=False, request_id=rid, timings_ms=timings
        )


def _sum(t: dict[str, float]) -> float:
    return round(sum(v for k, v in t.items() if k != "total_ms"), 2)
