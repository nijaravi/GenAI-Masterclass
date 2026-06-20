"""Vector store — LangChain's Chroma integration.

`build_vectorstore(embeddings)` returns a `langchain_chroma.Chroma` that both the
data pipeline (to add documents) and the backend (to retrieve) open against the
same persist directory. `load_all_documents` pulls every stored chunk back as
LangChain `Document`s, which the backend uses to build the BM25 retriever.
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import settings
from .logging_setup import get_logger

logger = get_logger(__name__)


def build_vectorstore(embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_path,
    )


def load_all_documents(vectorstore: Chroma) -> list[Document]:
    """Return every stored chunk as a LangChain Document (for the BM25 index)."""
    data = vectorstore.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    logger.info("loaded %d documents from the vector store", len(docs))
    return docs
