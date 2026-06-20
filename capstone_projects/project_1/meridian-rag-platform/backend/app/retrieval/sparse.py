"""Sparse retrieval — LangChain's BM25Retriever.

Keyword/lexical search that catches exact tokens dense search can miss (error
codes, policy numbers like "HR-LV-2024-03", acronyms). Built in memory from all
the stored chunks. Pairing this with dense search is what makes retrieval
"hybrid".
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from common.config import settings


def build_sparse_retriever(all_docs: list[Document]) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(all_docs)
    retriever.k = settings.sparse_top_k
    return retriever
