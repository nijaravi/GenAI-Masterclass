"""Metrics: aggregate the trace rows into the numbers the admin sees.

Plain Python over the rows from the trace store — no metrics library — so it's
easy to follow exactly how each number is produced:

  * latency p50 / p95, average latency
  * total requests, error/block rate, cache-hit rate
  * average quality scores (from the eval pipeline, when present)
  * per-user usage (request counts)
  * per-user and per-model cost
"""
from .store import TraceStore
from common.users import USERS


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(pct * len(s)))
    return round(s[idx], 2)


def overall_metrics(store: TraceStore) -> dict:
    rows = store.all()
    n = len(rows)
    if n == 0:
        return {"total_requests": 0}

    latencies = [r["latency_ms"] for r in rows]
    blocked = sum(1 for r in rows if r["blocked"])
    cached = sum(1 for r in rows if r["cached"])
    relevance = [r["relevance_score"] for r in rows if r["relevance_score"] is not None]
    faithfulness = [r["faithfulness_score"] for r in rows
                    if r["faithfulness_score"] is not None]

    return {
        "total_requests": n,
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "avg": round(sum(latencies) / n, 2),
        },
        "blocked_requests": blocked,
        "block_rate": round(blocked / n, 3),
        "cache_hits": cached,
        "cache_hit_rate": round(cached / n, 3),
        "judged_requests": len(relevance),
        "avg_relevance": round(sum(relevance) / len(relevance), 2) if relevance else None,
        "avg_faithfulness": round(sum(faithfulness) / len(faithfulness), 2)
            if faithfulness else None,
    }


def usage_by_user(store: TraceStore) -> list[dict]:
    rows = store.all()
    out = []
    for user in USERS:
        user_rows = [r for r in rows if r["user_id"] == user.user_id]
        out.append({
            "user_id": user.user_id,
            "name": user.name,
            "requests": len(user_rows),
            "blocked": sum(1 for r in user_rows if r["blocked"]),
            "cache_hits": sum(1 for r in user_rows if r["cached"]),
        })
    return out


def cost_by_user(store: TraceStore) -> list[dict]:
    rows = store.all()
    out = []
    total = 0.0
    for user in USERS:
        user_rows = [r for r in rows if r["user_id"] == user.user_id]
        cost = round(sum(r["cost_usd"] for r in user_rows), 6)
        tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in user_rows)
        total += cost
        out.append({
            "user_id": user.user_id,
            "name": user.name,
            "requests": len(user_rows),
            "total_tokens": tokens,
            "cost_usd": cost,
        })
    return {"per_user": out, "total_cost_usd": round(total, 6)}
