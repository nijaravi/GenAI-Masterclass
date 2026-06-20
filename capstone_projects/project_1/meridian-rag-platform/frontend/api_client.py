"""Tiny HTTP client the Streamlit app uses to talk to the backend.

Keeps all the request/header plumbing in one place so the UI code stays about
the UI. The user's API key goes in the `X-API-Key` header on every call.
"""
import httpx

DEFAULT_BASE_URL = "http://localhost:8000"


class BackendClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url

    def _headers(self) -> dict:
        return {"X-API-Key": self.api_key}

    def ask(self, question: str, category: str | None = None) -> dict:
        resp = httpx.post(
            f"{self.base_url}/v1/ask",
            headers=self._headers(),
            json={"question": question, "category": category},
            timeout=60,
        )
        return resp.json()

    def admin_metrics(self) -> dict:
        return httpx.get(f"{self.base_url}/v1/admin/metrics",
                         headers=self._headers(), timeout=30).json()

    def admin_usage(self) -> dict:
        return httpx.get(f"{self.base_url}/v1/admin/usage",
                         headers=self._headers(), timeout=30).json()

    def admin_cost(self) -> dict:
        return httpx.get(f"{self.base_url}/v1/admin/cost",
                         headers=self._headers(), timeout=30).json()
