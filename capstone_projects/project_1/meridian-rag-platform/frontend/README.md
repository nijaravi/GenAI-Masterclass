# Frontend (Client Layer)

A simple Streamlit app: a chat interface for the five onboarded users, plus an
admin dashboard for user 1. It talks to the backend over HTTP only — it has no
direct access to the model, the vector store, or the database. That separation
is the point: the frontend is just a client.

## What's here

| File | Job |
|------|-----|
| `app.py` | the Streamlit UI — user picker, chat tab, admin tab |
| `api_client.py` | a thin HTTP client that calls the backend and attaches the user's API key |

## Run it

The backend must be running first (`make backend`). Then:

```bash
make frontend
# or:
streamlit run frontend/app.py
```

Open http://localhost:8501.

## Using it

- **Sign in as** (sidebar): pick one of the 5 users. This sets the `X-API-Key`
  sent on every request, so the backend attributes usage and cost to that user
  and rate-limits them independently.
- **Chat tab**: ask a question; the answer comes back with an expandable
  **Sources** section listing the passages it cited, plus the model used, whether
  it was a cache hit, and any guardrail notes.
- **Admin tab** (only visible when signed in as *Aisha (Admin)*): live
  performance metrics, anomaly alerts, per-user usage, and per-user cost — all
  pulled from the backend's admin endpoints.

## Backend URL

Defaults to `http://localhost:8000`. Override it in the sidebar, or set the
`BACKEND_URL` environment variable (Docker Compose sets it to `http://backend:8000`).
