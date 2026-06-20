"""Compose the full retrieval stack into one LangChain retriever.

    dense ─┐
           ├─ EnsembleRetriever (RRF) ─→ ContextualCompression (cross-encoder)
    sparse ┘

The composed object is a normal LangChain retriever: `.invoke(question)` returns
the reranked top-n Documents. Category filtering (hr/it/...) rebuilds the dense
side with a metadata filter, so we expose a small builder rather than a singleton.
"""
from langchain_core.documents import Document

from .dense import build_dense_retriever
from .hybrid import build_hybrid_retriever
from .rerank import build_reranking_retriever
from .sparse import build_sparse_retriever


def build_retriever(vectorstore, all_docs: list[Document], category: str | None = None):
    dense = build_dense_retriever(vectorstore, category)
    sparse = build_sparse_retriever(all_docs)
    hybrid = build_hybrid_retriever(dense, sparse)
    return build_reranking_retriever(hybrid)
