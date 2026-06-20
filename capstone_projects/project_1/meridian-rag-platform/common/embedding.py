"""Embeddings — a LangChain `Embeddings` object.

We don't hand-roll vectors any more; we use LangChain's embedding integrations:

  * OpenAIEmbeddings        — text-embedding-3-small, when OPENAI_API_KEY is set.
  * DeterministicFakeEmbedding — LangChain's built-in deterministic fake, used in
                              the keyless demo / tests so the same text always
                              maps to the same vector (no key, no download).

Both the data pipeline (embedding chunks) and the backend (embedding the query,
via the retriever) use whatever this returns.
"""
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_openai import OpenAIEmbeddings

from .config import settings
from .logging_setup import get_logger

logger = get_logger(__name__)


def build_embeddings():
    if settings.embedding_provider == "openai" and settings.openai_api_key:
        logger.info("using OpenAIEmbeddings (%s)", settings.openai_embedding_model)
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    logger.info("using DeterministicFakeEmbedding (keyless)")
    return DeterministicFakeEmbedding(size=settings.stub_embedding_dim)
