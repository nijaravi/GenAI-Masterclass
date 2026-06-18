"""API tests using a fake pipeline so no models/keys are needed in CI.

Demonstrates the dependency-injection seam: we override get_pipeline with a
stub, so the HTTP layer is tested in isolation from embeddings/LLM.
"""
from fastapi.testclient import TestClient

from rag.api import main as api_main
from rag.models import Citation, QueryResponse


class _FakePipeline:
    class _Store:
        def count(self):
            return 42

    store = _Store()

    def query(self, query, top_n=None, category=None, use_cache=True):
        return QueryResponse(
            answer="Full-time employees get 25 days. [1]",
            citations=[Citation(chunk_id="c1", title="Leave Policy",
                                source="hr/leave-policy.md", snippet="25 days")],
            model="gpt-4o-mini", cached=False, request_id="test123",
            timings_ms={"total_ms": 12.3},
        )


def _client():
    api_main.app.dependency_overrides = {}
    # Patch the module-level singleton accessor.
    api_main.get_pipeline = lambda: _FakePipeline()  # type: ignore
    return TestClient(api_main.app)


def test_healthz():
    c = _client()
    r = c.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_query_returns_answer_and_citations():
    c = _client()
    r = c.post("/query", json={"query": "How much annual leave?"})
    assert r.status_code == 200
    body = r.json()
    assert "25 days" in body["answer"]
    assert body["citations"][0]["source"] == "hr/leave-policy.md"
    assert r.headers.get("x-request-id")


def test_empty_query_rejected():
    c = _client()
    r = c.post("/query", json={"query": "   "})
    assert r.status_code == 422
