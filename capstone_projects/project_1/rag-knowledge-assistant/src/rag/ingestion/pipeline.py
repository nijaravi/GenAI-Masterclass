"""Ingestion orchestration: load -> chunk -> embed -> upsert.

Kept separate from query-time code because it has a totally different runtime
profile (batch, write-heavy, run on a schedule) and you don't want it imported
into the hot request path.
"""
from __future__ import annotations

from ..config import settings
from ..embeddings.embedder import build_embedder
from ..logging_config import get_logger
from .chunking import chunk_document
from .loader import load_documents

logger = get_logger(__name__)


def build_store():
    if settings.vector_store == "pgvector":
        from ..stores.pgvector_store import PgVectorStore

        return PgVectorStore()
    from ..stores.chroma_store import ChromaStore

    return ChromaStore()


def run_ingestion(data_dir: str) -> int:
    docs = load_documents(data_dir)
    chunks = [c for d in docs for c in chunk_document(d)]
    if not chunks:
        logger.warning("no chunks produced; nothing to ingest")
        return 0

    embedder = build_embedder()
    embeddings = embedder.embed_documents([c.text for c in chunks])

    store = build_store()
    store.upsert(chunks, embeddings)
    logger.info(
        "ingestion complete",
        extra={"extra": {"docs": len(docs), "chunks": len(chunks),
                          "store_count": store.count()}},
    )
    return len(chunks)
