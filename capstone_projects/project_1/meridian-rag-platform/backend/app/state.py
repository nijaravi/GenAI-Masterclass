"""Application state: build every component once, at startup.

Builds the LangChain embeddings, Chroma vector store, the documents for BM25,
the chat models, and the RAG pipeline. Also turns on LangChain's global LLM
response cache so repeated identical calls are served from memory.
"""
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from common.config import settings
from common.embedding import build_embeddings
from common.logging_setup import get_logger
from common.vector_store import build_vectorstore, load_all_documents
from .monitoring.store import TraceStore
from .orchestration.pipeline import RAGPipeline
from .providers.llm import build_models

logger = get_logger(__name__)


class AppState:
    def __init__(self):
        logger.info("building application state")
        if settings.cache_enabled:
            set_llm_cache(InMemoryCache())   # LangChain global LLM response cache

        embeddings = build_embeddings()
        self.vectorstore = build_vectorstore(embeddings)
        self.all_docs = load_all_documents(self.vectorstore)
        if not self.all_docs:
            logger.warning("vector store is empty — run `python -m "
                           "data_pipeline.run_ingest` first")
        self.models = build_models()
        self.pipeline = RAGPipeline(self.vectorstore, self.all_docs, self.models)
        self.trace_store = TraceStore()
        logger.info("application state ready (%d docs indexed)", len(self.all_docs))


_state = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
