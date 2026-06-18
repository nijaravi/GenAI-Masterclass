"""Wiring for the API.

We build heavy objects (embedder, reranker model, BM25 index) ONCE at startup
and reuse them across requests. Rebuilding a CrossEncoder per request would add
seconds of latency — a classic RAG-in-prod mistake. The BM25 index is rebuilt
from whatever is in the vector store so dense and sparse stay in sync.
"""
from __future__ import annotations

from ..config import settings
from ..ingestion.loader import load_documents
from ..ingestion.chunking import chunk_document
from ..logging_config import get_logger
from ..pipeline import RAGPipeline
from ..retrieval.bm25 import BM25Retriever

logger = get_logger(__name__)

_pipeline: RAGPipeline | None = None


def _build_bm25() -> BM25Retriever:
    # Rebuild the lexical index from source docs (kept in sync with ingestion).
    docs = load_documents("data/raw")
    chunks = [c for d in docs for c in chunk_document(d)]
    return BM25Retriever(chunks)


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("initialising RAG pipeline")
        _pipeline = RAGPipeline(bm25=_build_bm25())
    return _pipeline
