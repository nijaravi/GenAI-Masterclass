"""Anomaly alerts: simple threshold checks over the current metrics.

Not statistical anomaly detection — just clear, explainable rules that fire an
alert when something looks wrong (latency too high, too many blocks, quality
dropping). In production these become monitor rules in Datadog/CloudWatch or a
statistical detector, but threshold rules are exactly how most teams start.
"""

# Thresholds an operator would tune.
P95_LATENCY_MS = 4000
BLOCK_RATE = 0.25
MIN_FAITHFULNESS = 3.0


def check_anomalies(metrics: dict) -> list[str]:
    alerts: list[str] = []
    if metrics.get("total_requests", 0) == 0:
        return alerts

    p95 = metrics.get("latency_ms", {}).get("p95", 0)
    if p95 > P95_LATENCY_MS:
        alerts.append(f"High p95 latency: {p95} ms (threshold {P95_LATENCY_MS})")

    if metrics.get("block_rate", 0) > BLOCK_RATE:
        alerts.append(
            f"High block rate: {metrics['block_rate']:.0%} "
            f"(threshold {BLOCK_RATE:.0%}) — possible attack or misuse")

    faith = metrics.get("avg_faithfulness")
    if faith is not None and faith < MIN_FAITHFULNESS:
        alerts.append(
            f"Low faithfulness: {faith} (threshold {MIN_FAITHFULNESS}) — "
            "answers may be drifting from the source docs")

    return alerts
