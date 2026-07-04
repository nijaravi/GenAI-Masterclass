"""
Section 8 of the design doc specifies Pinecone as the vector store, with
namespace/metadata isolation between the customer-product and
developer-docs audiences sharing one index.

This module implements that contract behind a small interface
(`VectorStore.upsert` / `VectorStore.query`) with two backends:

  - ChromaVectorStore  - runs fully local/offline, used whenever
                          DEMO_MODE=true or no Pinecone key is set.
  - PineconeVectorStore - the production backend from the design doc,
                          used automatically when a real key is present.

The RAG Agent node never imports Chroma or Pinecone directly - it only
calls get_vector_store(), so the retrieval code in rag_agent.py is
identical regardless of which backend is live. This is the same
demo/production dual-mode pattern used elsewhere in this codebase
(see llm.py) so the project can be run and demoed with zero external
accounts, then pointed at real infrastructure with only env-var changes.
"""
from __future__ import annotations

from typing import Protocol

from app import config
from app.state import RetrievedChunk


class VectorStore(Protocol):
    def upsert(self, namespace: str, docs: list[dict]) -> None: ...
    def query(self, namespace: str, query_text: str, top_k: int = 4) -> list[RetrievedChunk]: ...


class ChromaVectorStore:
    """Local, in-memory Chroma collection per namespace. Uses Chroma's
    bundled ONNX MiniLM embedding function, so it needs no API key and no
    network access - this is what makes DEMO_MODE fully offline."""

    def __init__(self):
        import chromadb

        self._client = chromadb.EphemeralClient()
        self._collections: dict[str, "chromadb.Collection"] = {}

    def _collection(self, namespace: str):
        if namespace not in self._collections:
            self._collections[namespace] = self._client.get_or_create_collection(
                name=namespace
            )
        return self._collections[namespace]

    def upsert(self, namespace: str, docs: list[dict]) -> None:
        col = self._collection(namespace)
        col.upsert(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[{"source": d["source"], "namespace": namespace} for d in docs],
        )

    def query(self, namespace: str, query_text: str, top_k: int = 4) -> list[RetrievedChunk]:
        col = self._collection(namespace)
        if col.count() == 0:
            return []
        result = col.query(query_texts=[query_text], n_results=min(top_k, col.count()))
        chunks: list[RetrievedChunk] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    score=round(1.0 - dist, 4),  # cosine distance -> similarity
                    namespace=namespace,
                )
            )
        return chunks


class PineconeVectorStore:
    """Production backend matching Section 8: Pinecone with
    namespace-per-audience isolation inside one index/project."""

    def __init__(self, api_key: str, index_name: str):
        from pinecone import Pinecone
        from langchain_openai import OpenAIEmbeddings

        self._pc = Pinecone(api_key=api_key)
        if index_name not in [i["name"] for i in self._pc.list_indexes()]:
            self._pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={"model": "llama-text-embed-v2", "field_map": {"text": "text"}},
            )
        self._index = self._pc.Index(index_name)
        self._embeddings = OpenAIEmbeddings()

    def upsert(self, namespace: str, docs: list[dict]) -> None:
        vectors = self._embeddings.embed_documents([d["text"] for d in docs])
        self._index.upsert(
            vectors=[
                {
                    "id": d["id"],
                    "values": vec,
                    "metadata": {"text": d["text"], "source": d["source"], "namespace": namespace},
                }
                for d, vec in zip(docs, vectors)
            ],
            namespace=namespace,
        )

    def query(self, namespace: str, query_text: str, top_k: int = 4) -> list[RetrievedChunk]:
        vec = self._embeddings.embed_query(query_text)
        result = self._index.query(
            vector=vec, top_k=top_k, namespace=namespace, include_metadata=True
        )
        return [
            RetrievedChunk(
                text=m["metadata"]["text"],
                source=m["metadata"]["source"],
                score=round(m["score"], 4),
                namespace=namespace,
            )
            for m in result["matches"]
        ]


_singleton: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _singleton
    if _singleton is not None:
        return _singleton

    if config.LIVE_PINECONE:
        _singleton = PineconeVectorStore(config.PINECONE_API_KEY, config.PINECONE_INDEX)
    else:
        _singleton = ChromaVectorStore()
    return _singleton


def seed_if_empty() -> None:
    """Idempotent seed of both namespaces. Safe to call on every app
    startup - upsert is a no-op-equivalent for docs that already exist."""
    from app.rag.seed_data import CUSTOMER_PRODUCT_DOCS, DEVELOPER_DOCS

    store = get_vector_store()
    store.upsert("customer-product", CUSTOMER_PRODUCT_DOCS)
    store.upsert("developer-docs", DEVELOPER_DOCS)
