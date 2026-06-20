"""Tests for the LangChain data pipeline + retrieval stack."""
from common.embedding import build_embeddings
from common.vector_store import build_vectorstore, load_all_documents
from data_pipeline.loader import load_documents
from data_pipeline.splitter import split_documents
from backend.app.retrieval.retrievers import build_retriever


def test_loader_sets_category_and_source():
    docs = load_documents()
    assert docs, "no documents loaded"
    cats = {d.metadata["category"] for d in docs}
    assert {"hr", "it", "security", "engineering"} <= cats
    assert all("/" in d.metadata["source"] for d in docs)


def test_splitter_produces_chunks_with_ids():
    chunks = split_documents(load_documents())
    assert len(chunks) > len(load_documents())          # split into more pieces
    assert all(c.metadata.get("chunk_id") for c in chunks)
    assert all(c.metadata.get("source") for c in chunks)


def test_retriever_finds_relevant_doc(ingested):
    vs = build_vectorstore(build_embeddings())
    all_docs = load_all_documents(vs)
    retriever = build_retriever(vs, all_docs)
    results = retriever.invoke("How many days of annual leave do I get?")
    assert results
    # The hybrid (BM25 + dense) retriever should surface the leave policy doc.
    assert any("leave" in d.metadata.get("source", "") for d in results)
