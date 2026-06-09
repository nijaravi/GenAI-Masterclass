# ============================================================
# app/rag.py
# RAG layer: ChromaDB vector store + OpenAI embeddings.
#
# Flow:
#   1. On startup → ingest() loads knowledge_base.py into ChromaDB
#   2. On each request → retrieve() embeds the query and returns top-k chunks
#   3. build_context() formats chunks into the prompt context block
#
# ChromaDB runs in-process (no separate server needed).
# Data is persisted to ./chroma_data so re-starts don't re-embed.
# ============================================================

import os
import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from openai import AsyncOpenAI, OpenAI

from data.knowledge_base import DOCUMENTS

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
COLLECTION_NAME = "techstore_knowledge"
CHROMA_PATH      = os.getenv("CHROMA_PATH", "./chroma_data")
EMBED_MODEL      = "text-embedding-3-small"   # cheap, fast, good quality
TOP_K            = int(os.getenv("RAG_TOP_K", "3"))
MIN_RELEVANCE    = float(os.getenv("RAG_MIN_RELEVANCE", "0.35"))  # cosine distance threshold

# ── Clients ───────────────────────────────────────────────────
_sync_client  = OpenAI()       # used for ingestion (sync, runs once at startup)
_async_client = AsyncOpenAI()  # used for query embedding at request time

# ── ChromaDB setup ────────────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection:    Optional[chromadb.Collection]        = None


def _get_collection() -> chromadb.Collection:
    """Return the cached collection, initialising ChromaDB if needed."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Use OpenAI embedding function so ChromaDB auto-embeds during add()
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL,
    )

    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return _collection


# ── Ingestion ─────────────────────────────────────────────────

def ingest(force: bool = False) -> int:
    """
    Load all documents from knowledge_base.py into ChromaDB.

    Args:
        force: If True, delete and re-create the collection (full re-embed).
               If False (default), skip docs already in the store.

    Returns:
        Number of documents added.
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
            f"RAG: collection '{COLLECTION_NAME}' already has {len(existing_ids)} docs. "
            "Nothing to ingest."
        )
        return 0

    # Build documents as rich text combining title + content + keywords
    # so the embedding captures both the semantic content and key terms
    texts = [
        f"{doc['title']}\n\n{doc['content']}\n\nKeywords: {', '.join(doc['keywords'])}"
        for doc in new_docs
    ]

    collection.add(
        ids=[doc["id"] for doc in new_docs],
        documents=texts,
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
        f"RAG: ingested {len(new_docs)} documents into '{COLLECTION_NAME}'. "
        f"Total: {len(existing_ids) + len(new_docs)} docs."
    )
    return len(new_docs)


# ── Retrieval ─────────────────────────────────────────────────

async def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and return the top-k most relevant document chunks.

    Returns a list of dicts:
        [{id, title, category, content, distance, relevance_score}, ...]

    Chunks with distance > MIN_RELEVANCE threshold are filtered out
    to avoid injecting unrelated content into the prompt.
    """
    collection = _get_collection()

    # Embed the query using the async client
    embed_response = await _async_client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
    )
    query_embedding = embed_response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        if distance > MIN_RELEVANCE:
            # Too far — not relevant enough to include
            continue
        chunks.append({
            "id":              results["ids"][0][i],
            "title":           results["metadatas"][0][i]["title"],
            "category":        results["metadatas"][0][i]["category"],
            "content":         results["documents"][0][i],
            "distance":        round(distance, 4),
            "relevance_score": round(1 - distance, 4),
        })

    logger.debug(
        f"RAG retrieve: query='{query[:60]}' "
        f"returned {len(chunks)}/{top_k} chunks above threshold"
    )
    return chunks


def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a clean context block for the prompt.

    Example output:
        [1] Standard Return Window (returns)
        TechStore accepts returns within 15 days ...

        [2] Return Process — How to Return (returns)
        To initiate a return: log into your TechStore account ...
    """
    if not chunks:
        return "No relevant policy documents found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Strip the keyword line added during ingestion — not needed in the prompt
        content = chunk["content"]
        if "\n\nKeywords:" in content:
            content = content.split("\n\nKeywords:")[0].strip()
        parts.append(
            f"[{i}] {chunk['title']} ({chunk['category']})\n{content}"
        )

    return "\n\n".join(parts)


# ── Collection stats ──────────────────────────────────────────

def collection_info() -> dict:
    """Return basic stats about the ChromaDB collection."""
    try:
        col = _get_collection()
        return {
            "collection": COLLECTION_NAME,
            "document_count": col.count(),
            "persist_path": CHROMA_PATH,
            "embed_model": EMBED_MODEL,
            "top_k": TOP_K,
        }
    except Exception as e:
        return {"error": str(e)}
