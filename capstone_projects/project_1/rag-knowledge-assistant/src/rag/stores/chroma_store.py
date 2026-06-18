"""ChromaDB store — the zero-ops default for dev and small/medium corpora.

Chroma persists to local disk and needs no separate server, which makes it the
right call for prototyping and single-node deployments. We pass embeddings in
explicitly (we own the embedder) rather than letting Chroma embed for us, so the
exact same vectors flow to both Chroma and pgvector.
"""
from __future__ import annotations

from ..config import settings
from ..logging_config import get_logger
from ..models import Chunk, ScoredChunk

logger = get_logger(__name__)


class ChromaStore:
    def __init__(self) -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        # cosine space matches normalized embeddings used elsewhere.
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "source": c.source,
                    "category": c.category,
                    "position": c.position,
                }
                for c in chunks
            ],
        )
        logger.info("chroma upsert", extra={"extra": {"n": len(chunks)}})

    def search(
        self, query_embedding, top_k, category=None
    ) -> list[ScoredChunk]:
        where = {"category": category} if category else None
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[ScoredChunk] = []
        ids = res["ids"][0]
        for i, cid in enumerate(ids):
            md = res["metadatas"][0][i]
            # Chroma returns cosine *distance*; convert to similarity.
            distance = res["distances"][0][i]
            out.append(
                ScoredChunk(
                    chunk=Chunk(
                        chunk_id=cid,
                        doc_id=md["doc_id"],
                        title=md["title"],
                        source=md["source"],
                        category=md["category"],
                        text=res["documents"][0][i],
                        position=int(md["position"]),
                    ),
                    score=1.0 - distance,
                    retriever="dense",
                )
            )
        return out

    def count(self) -> int:
        return self.collection.count()
