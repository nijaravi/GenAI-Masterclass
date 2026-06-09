# ============================================================
# app/monitoring.py
# Structured logging, cost tracking, and metrics aggregation.
# ============================================================

import json
import time
import logging
from typing import Optional

_logger = logging.getLogger(__name__)

# In-memory log store — swap for Postgres / ClickHouse / Loki in prod
_log_store: list[dict] = []

# ── Pricing (per 1M tokens) ───────────────────────────────────
# Update these when OpenAI changes rates
PRICING: dict[str, dict] = {
    "gpt-4o-mini":          {"input": 0.15,  "output": 0.60},
    "gpt-4o":               {"input": 5.00,  "output": 15.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


# ── Logging ───────────────────────────────────────────────────

def log_event(event: str, **kwargs) -> None:
    """Emit a structured JSON log entry and store in memory."""
    entry = {"ts": round(time.time(), 3), "event": event, **kwargs}
    _log_store.append(entry)
    _logger.info(json.dumps(entry))


def get_logs(event_type: Optional[str] = None) -> list[dict]:
    """Return stored log entries, optionally filtered by event type."""
    if event_type:
        return [e for e in _log_store if e.get("event") == event_type]
    return list(_log_store)


# ── Cost tracking ─────────────────────────────────────────────

def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Convert token counts to USD cost."""
    p = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens / 1e6 * p["input"]) + (output_tokens / 1e6 * p["output"])


# ── Metrics aggregation ───────────────────────────────────────

def get_metrics() -> dict:
    """
    Aggregate the in-memory log store into a metrics summary.
    Called by GET /metrics endpoint.
    """
    chat_logs    = get_logs("chat_success")
    blocked_logs = get_logs("input_blocked")
    eval_logs    = get_logs("eval_verdict")
    rag_logs     = get_logs("rag_retrieve")

    if not chat_logs:
        return {"message": "No requests yet."}

    latencies   = [e["latency_ms"] for e in chat_logs]
    costs       = [e["cost_usd"]   for e in chat_logs]
    latencies_s = sorted(latencies)

    eval_total  = len(eval_logs)
    pass_count  = sum(1 for e in eval_logs if e.get("verdict") == "PASS")
    fail_count  = sum(1 for e in eval_logs if e.get("verdict") == "FAIL")

    # RAG stats — avg chunks retrieved, avg relevance
    rag_chunks    = [e.get("chunks_returned", 0) for e in rag_logs]
    rag_relevance = [e.get("top_relevance", 0)   for e in rag_logs]

    model_counts: dict[str, int] = {}
    for e in chat_logs:
        m = e.get("model", "unknown")
        model_counts[m] = model_counts.get(m, 0) + 1

    return {
        "requests": {
            "total":   len(chat_logs),
            "blocked": len(blocked_logs),
            "block_rate": round(len(blocked_logs) / max(1, len(chat_logs) + len(blocked_logs)), 3),
        },
        "latency": {
            "p50_ms":  latencies_s[len(latencies_s) // 2],
            "p95_ms":  latencies_s[min(len(latencies_s) - 1, int(len(latencies_s) * 0.95))],
            "avg_ms":  round(sum(latencies) / len(latencies), 1),
        },
        "cost": {
            "total_usd":        round(sum(costs), 6),
            "avg_per_call_usd": round(sum(costs) / len(costs), 6),
            "proj_10k_usd":     round(sum(costs) / len(costs) * 10_000, 2),
        },
        "rag": {
            "total_retrievals":    len(rag_logs),
            "avg_chunks_returned": round(sum(rag_chunks) / max(1, len(rag_chunks)), 2),
            "avg_top_relevance":   round(sum(rag_relevance) / max(1, len(rag_relevance)), 3),
        },
        "eval": {
            "judged":    eval_total,
            "pass":      pass_count,
            "fail":      fail_count,
            "pass_rate": round(pass_count / eval_total, 3) if eval_total else None,
        },
        "model_usage": model_counts,
    }
