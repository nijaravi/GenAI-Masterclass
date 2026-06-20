"""Shared test fixtures.

Tests run keyless (DeterministicFakeEmbedding + FakeListChatModel), so they need
no API keys or network. The `client` fixture ingests the corpus once into the
local Chroma store, then returns a FastAPI TestClient.
"""
import pytest
from fastapi.testclient import TestClient

from common.embedding import build_embeddings
from common.vector_store import build_vectorstore
from data_pipeline.indexer import index_chunks
from data_pipeline.loader import load_documents
from data_pipeline.splitter import split_documents


@pytest.fixture(scope="session")
def ingested():
    vs = build_vectorstore(build_embeddings())
    if vs._collection.count() == 0:
        index_chunks(split_documents(load_documents()))
    return True


@pytest.fixture()
def client(ingested):
    from backend.app.main import app
    return TestClient(app)


USER = {"X-API-Key": "key-ben-002"}
ADMIN = {"X-API-Key": "key-aisha-001"}
