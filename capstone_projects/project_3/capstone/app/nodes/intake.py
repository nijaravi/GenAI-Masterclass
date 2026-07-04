"""
Section 4.2: 'First stop for every request... capture and lightly normalize
the raw question, attach session context, and hand a clean payload to the
Planner Agent. This keeps prompt-injection surface area and malformed input
away from the reasoning agents downstream.'

Concretely: strip control characters, cap length, and record that the
request entered the graph - it does not classify or route anything.
"""
from __future__ import annotations

from app.state import OrchestratorState

_MAX_QUERY_CHARS = 2000


def _normalize(text: str) -> str:
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return cleaned.strip()[:_MAX_QUERY_CHARS]


async def intake_node(state: OrchestratorState) -> dict:
    return {
        "user_query": _normalize(state["user_query"]),
        "node_path": state.get("node_path", []) + ["intake"],
    }
