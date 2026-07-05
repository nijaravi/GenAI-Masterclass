"""
Section 8: a namespace-isolated vector store shared by two audiences
(customer product data, developer reference docs) so retrieval never
crosses between them.

This build uses Chroma, running fully local/in-memory with its bundled
ONNX MiniLM embeddings - no API key, no network call once the (~80MB,
one-time) embedding model is cached. In production this is Pinecone
(Section 8 specifies it) - the RAG node only ever calls get_vector_store()
and .query()/.upsert(), so swapping the class body here for a Pinecone
client later doesn't touch any other file.
"""
from __future__ import annotations

import chromadb

from app.state import RetrievedChunk


class VectorStore:
    def __init__(self):
        self._client = chromadb.EphemeralClient()
        self._collections: dict[str, "chromadb.Collection"] = {}

    def _collection(self, namespace: str):
        if namespace not in self._collections:
            self._collections[namespace] = self._client.get_or_create_collection(name=namespace)
        return self._collections[namespace]

    def upsert(self, namespace: str, docs: list[dict]) -> None:
        col = self._collection(namespace)
        col.upsert(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[{"source": d["source"]} for d in docs],
        )

    def query(self, namespace: str, query_text: str, top_k: int = 4) -> list[RetrievedChunk]:
        col = self._collection(namespace)
        if col.count() == 0:
            return []
        result = col.query(query_texts=[query_text], n_results=min(top_k, col.count()))
        chunks: list[RetrievedChunk] = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    score=round(1.0 - dist, 4),  # cosine distance -> similarity
                    namespace=namespace,
                )
            )
        return chunks


_store = VectorStore()


def get_vector_store() -> VectorStore:
    return _store


def seed_if_empty() -> None:
    """Idempotent seed of both namespaces - safe to call on every startup."""
    from app.rag.seed_data import CUSTOMER_PRODUCT_DOCS, DEVELOPER_DOCS

    _store.upsert("customer-product", CUSTOMER_PRODUCT_DOCS)
    _store.upsert("developer-docs", DEVELOPER_DOCS)
