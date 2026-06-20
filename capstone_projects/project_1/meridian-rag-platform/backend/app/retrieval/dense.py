"""Dense retrieval — the vector store as a LangChain retriever.

Semantic search: embed the question and fetch the nearest chunks by cosine
similarity. `as_retriever` turns the Chroma store into a standard LangChain
Retriever we can drop straight into the ensemble below.
"""
from common.config import settings


def build_dense_retriever(vectorstore, category: str | None = None):
    search_kwargs = {"k": settings.dense_top_k}
    if category:
        search_kwargs["filter"] = {"category": category}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
