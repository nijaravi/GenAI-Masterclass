"""Sample traffic to evaluate.

The eval pipeline doesn't judge every request (that would double the LLM cost).
It samples a handful of past requests that haven't been scored yet. Here that's
a random sample of unjudged, non-blocked traces from the store; in production
you'd sample a small percentage of live traffic on a schedule.
"""
from backend.app.monitoring.store import TraceStore


def sample_traces(store: TraceStore, limit: int) -> list[dict]:
    return store.sample_unjudged(limit)
