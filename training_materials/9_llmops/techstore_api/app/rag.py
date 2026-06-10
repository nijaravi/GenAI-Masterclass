# ============================================================
# app/rag.py
# RAG layer: ChromaDB vector store + OpenAI embeddings.
#
# Root cause fix: chromadb's built-in OpenAIEmbeddingFunction
# uses the old openai.Embedding API (removed in openai>=1.0.0).
# Solution: pass embeddings=None to ChromaDB and call the
# OpenAI embeddings API ourselves using the v1 SDK directly.
# ChromaDB stores and queries raw vectors — we handle embedding.
#
# Flow:
#   1. startup → ingest() embeds all docs via OpenAI, stores in ChromaDB
#   2. request → retrieve() embeds the query, queries ChromaDB by vector
#   3. build_context() formats chunks into the prompt context block
# ============================================================

import os
import logging
from typing import Optional

import chromadb
from openai import AsyncOpenAI, OpenAI

from data.knowledge_base import DOCUMENTS

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
COLLECTION_NAME = "techstore_knowledge"
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_data")
EMBED_MODEL     = "text-embedding-3-small"
TOP_K           = int(os.getenv("RAG_TOP_K", "3"))
# ChromaDB cosine distance range is [0, 2] — NOT [0, 1]:
#   0.0 = identical   ~0.3-0.6 = relevant   1.0 = unrelated   2.0 = opposite
# Default 1.0 passes anything meaningful. Tune down via RAG_MIN_RELEVANCE
# once you've seen real distances for your queries via GET /rag/debug.
MIN_RELEVANCE   = float(os.getenv("RAG_MIN_RELEVANCE", "1.0"))

# ── OpenAI clients ────────────────────────────────────────────
# Sync client for ingestion (runs once at startup, outside async context)
# Async client for query-time embedding (inside request handlers)
_sync_client  = OpenAI()
_async_client = AsyncOpenAI()

# ── ChromaDB ──────────────────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection:    Optional[chromadb.Collection]        = None


def _get_collection() -> chromadb.Collection:
    """
    Return the cached ChromaDB collection.
    No embedding_function is passed — we manage embeddings ourselves.
    This sidesteps the chromadb/openai>=1.0 incompatibility entirely.
    """
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # embedding_function=None → ChromaDB stores raw vectors we provide
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _embed_sync(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts synchronously using openai v1 SDK.
    Used during ingestion (startup, not in an async context).
    """
    response = _sync_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    # response.data is ordered to match input order
    return [item.embedding for item in response.data]


async def _embed_async(text: str) -> list[float]:
    """
    Embed a single query asynchronously using openai v1 SDK.
    Used at request time inside async endpoint handlers.
    """
    response = await _async_client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding


# ── Ingestion ─────────────────────────────────────────────────

def ingest(force: bool = False) -> int:
    """
    Embed and store all knowledge_base documents in ChromaDB.

    Args:
        force: Delete and re-create collection before ingesting.
               Use when the knowledge base content has changed.

    Returns:
        Number of documents newly added (0 if already up to date).
    """
    global _collection

    if force and _chroma_client is not None:
        try:
            _chroma_client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection for full re-ingestion.")
        except Exception:
            pass
        _collection = None

    collection = _get_collection()
    existing_ids = set(collection.get()["ids"])
    new_docs = [d for d in DOCUMENTS if d["id"] not in existing_ids]

    if not new_docs:
        logger.info(
            f"RAG: '{COLLECTION_NAME}' already has {len(existing_ids)} docs — "
            "nothing to ingest."
        )
        return 0

    # Rich text for embedding: title + content + keywords
    # Including keywords improves retrieval for short exact-match queries
    texts = [
        f"{doc['title']}\n\n{doc['content']}\n\nKeywords: {', '.join(doc['keywords'])}"
        for doc in new_docs
    ]

    logger.info(f"RAG: embedding {len(new_docs)} documents via OpenAI ({EMBED_MODEL})...")
    embeddings = _embed_sync(texts)

    collection.add(
        ids=[doc["id"] for doc in new_docs],
        documents=texts,
        embeddings=embeddings,          # we supply vectors directly
        metadatas=[
            {
                "category": doc["category"],
                "title":    doc["title"],
                "doc_id":   doc["id"],
            }
            for doc in new_docs
        ],
    )

    logger.info(
        f"RAG: ingested {len(new_docs)} docs into '{COLLECTION_NAME}'. "
        f"Total: {len(existing_ids) + len(new_docs)}."
    )
    return len(new_docs)


# ── Retrieval ─────────────────────────────────────────────────

async def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and return the top-k most relevant chunks.

    Chunks with cosine distance > MIN_RELEVANCE are filtered out
    so unrelated content never enters the prompt.

    Returns:
        [{id, title, category, content, distance, relevance_score}, ...]
    """
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("RAG: collection is empty — did ingest() run at startup?")
        return []

    query_embedding = await _embed_async(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        # Log every distance so you can tune MIN_RELEVANCE via /rag/debug
        logger.debug(
            f"RAG: '{results['metadatas'][0][i]['title']}' "
            f"distance={distance:.4f} threshold={MIN_RELEVANCE}"
        )
        if distance > MIN_RELEVANCE:
            continue
        chunks.append({
            "id":              results["ids"][0][i],
            "title":           results["metadatas"][0][i]["title"],
            "category":        results["metadatas"][0][i]["category"],
            "content":         results["documents"][0][i],
            "distance":        round(distance, 4),
            "relevance_score": round(1 - distance, 4),
        })

    logger.info(
        f"RAG: query='{query[:60]}' → "
        f"{len(chunks)}/{top_k} chunks passed threshold ({MIN_RELEVANCE})"
    )
    return chunks


# ── Context builder ───────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Output format:
        [1] Standard Return Window (returns)
        TechStore accepts returns within 15 days ...

        [2] Return Process — How to Return (returns)
        To initiate a return: log into your account ...
    """
    if not chunks:
        return "No relevant policy documents found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk["content"]
        if "\n\nKeywords:" in content:
            content = content.split("\n\nKeywords:")[0].strip()
        parts.append(f"[{i}] {chunk['title']} ({chunk['category']})\n{content}")

    return "\n\n".join(parts)


# ── Collection stats ──────────────────────────────────────────

def collection_info() -> dict:
    """Return basic stats about the ChromaDB collection."""
    try:
        col = _get_collection()
        return {
            "collection":     COLLECTION_NAME,
            "document_count": col.count(),
            "persist_path":   CHROMA_PATH,
            "embed_model":    EMBED_MODEL,
            "top_k":          TOP_K,
            "min_relevance":  MIN_RELEVANCE,
        }
    except Exception as e:
        return {"error": str(e)}


async def raw_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """
    Return ALL top-k results with their raw distances — NO threshold filtering.
    Used by GET /rag/debug to calibrate MIN_RELEVANCE.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_embedding = await _embed_async(query)
    n = min(top_k, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "id":              results["ids"][0][i],
            "title":           results["metadatas"][0][i]["title"],
            "category":        results["metadatas"][0][i]["category"],
            "distance":        round(results["distances"][0][i], 4),
            "relevance_score": round(1 - results["distances"][0][i], 4),
            "passes_threshold": results["distances"][0][i] <= MIN_RELEVANCE,
            "current_threshold": MIN_RELEVANCE,
        }
        for i in range(len(results["ids"][0]))
    ]
