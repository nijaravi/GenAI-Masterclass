"""
Section 4.4: 'The core control node. Given the plan and the user's role, it
decides the single next destination ... After the destination agent
returns a result, control comes back to the Orchestrator, which decides
whether another step is needed (multi-hop) or whether to return the final
answer.'

This node is visited once after the Planner, then again after every
specialist node (see graph.py's loop-back edges). Each visit does three
things, in order:
  1. Pick a route (from the plan on the first visit, from a queued
     next_category on a later visit, or 'final' if nothing's left).
  2. Enforce the role gate (Section 10) - a route not allowed for this
     role resolves to 'final' with an honest refusal, never a silent swap.
  3. Increment hop_count and force-terminate at MAX_HOPS (Section 6).
"""
from __future__ import annotations

from app.config import MAX_HOPS, ROLE_ALLOWED_ROUTES
from app.state import OrchestratorState

_CATEGORY_TO_ROUTE = {
    "product_info_lookup": "rag",
    "doc_lookup": "rag",
    "account_status_internal": "mcp",
    "account_status_external": "external_agent",
    "code_task": "coder",
}


def _category_from_plan(plan: str) -> str:
    return (plan or "").split(":", 1)[0].strip()


async def orchestrator_node(state: OrchestratorState) -> dict:
    hop_count = state.get("hop_count", 0) + 1
    node_path = state.get("node_path", []) + ["orchestrator"]

    if hop_count > MAX_HOPS:
        return {
            "route": "final",
            "hop_count": hop_count,
            "agent_output": state.get("agent_output")
                or "I wasn't able to fully resolve this within the allowed number of steps.",
            "node_path": node_path,
        }

    is_first_visit = state.get("agent_output") is None
    if is_first_visit:
        route = _CATEGORY_TO_ROUTE.get(_category_from_plan(state.get("plan", "")), "rag")
    elif state.get("next_category"):
        # Multi-hop: one queued follow-up intent left to handle (Section 7.3).
        next_category = state["next_category"]
        route = _CATEGORY_TO_ROUTE.get(next_category, "final")
        return {
            "route": route if route in ROLE_ALLOWED_ROUTES.get(state["user_role"], set()) else "final",
            "plan": f"{next_category}: (follow-up) — '{state['user_query']}'",
            "next_category": None,  # consumed
            "hop_count": hop_count,
            "node_path": node_path,
        }
    else:
        route = "final"

    if route not in ROLE_ALLOWED_ROUTES.get(state["user_role"], set()):
        return {
            "route": "final",
            "hop_count": hop_count,
            "agent_output": f"This request isn't available for the '{state['user_role']}' role on this platform.",
            "node_path": node_path,
        }

    return {"route": route, "hop_count": hop_count, "node_path": node_path}
