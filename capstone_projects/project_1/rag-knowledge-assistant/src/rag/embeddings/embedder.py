"""Embeddings.

A thin interface over the embedding provider so the rest of the system never
hard-codes OpenAI vs. a local model. Two implementations:

  * OpenAIEmbedder    — text-embedding-3-small (1536-d). Cheap, strong, hosted.
  * LocalEmbedder     — sentence-transformers/all-MiniLM-L6-v2 (384-d). Runs
                        offline, good for air-gapped / cost-sensitive deploys.

Batching matters: embedding one-by-one is the classic ingestion bottleneck and
blows up API cost via per-request overhead. We batch.
"""
from __future__ import annotations

from typing import Protocol

from ..config import settings
from ..logging_config import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_embedding_model
        self.dim = settings.embedding_dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI accepts up to 2048 inputs/request; keep batches modest to
        # bound memory and stay well under token limits.
        out: list[list[float]] = []
        BATCH = 128
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class LocalEmbedder:
    def __init__(self, model: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model or settings.local_embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, batch_size=64, normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def build_embedder() -> Embedder:
    if settings.embedding_provider == "local":
        logger.info("using local embedder")
        return LocalEmbedder()
    logger.info("using OpenAI embedder")
    return OpenAIEmbedder()
