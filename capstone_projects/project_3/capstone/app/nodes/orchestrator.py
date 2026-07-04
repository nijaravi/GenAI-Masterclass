"""
Section 4.4: 'The core control node. Given the plan and the user's role, it
decides the single next destination ... Routing considers both the nature
of the request and who is asking ... After the destination agent returns a
result, control comes back to the Orchestrator, which decides whether
another step is needed (multi-hop) or whether to return the final answer.'

This node is visited multiple times per request (it's the hub every
specialist node returns to - see graph.py's conditional edges). Each visit:
  1. Enforces the role gate from Section 10 (role-aware routing map).
  2. Picks the next route via the model, or drains pending_subtasks for
     the second/third hop of a compound request (Section 7.3).
  3. Increments hop_count and force-terminates at MAX_HOPS (Section 6).
"""
from __future__ import annotations

from app import config
from app.llm import get_chat_model
from app.state import OrchestratorState

# Section 10: role-aware routing map. A route only reachable for roles
# listed here - enforced at the agent layer as a UX control, on top of
# (not instead of) the MCP/backend server-side authorization check.
_ROLE_ALLOWED_ROUTES = {
    "customer": {"rag", "external_agent", "final"},
    "client": {"mcp", "external_agent", "final"},
    "developer": {"rag", "coder", "final"},
}

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
    role = state["user_role"]
    node_path = state.get("node_path", []) + ["orchestrator"]

    if hop_count > config.MAX_HOPS:
        return {
            "route": "final",
            "hop_count": hop_count,
            "agent_output": state.get("agent_output")
            or "I wasn't able to fully resolve this within the allowed number of steps.",
            "node_path": node_path,
        }

    # First hop after the Planner: decide from the plan. Subsequent hops
    # (after a specialist node already ran): either drain a queued
    # sub-intent (multi-hop, Section 7.3) or finish.
    if state.get("agent_output") is None:
        category = _category_from_plan(state.get("plan", ""))
        route = _CATEGORY_TO_ROUTE.get(category, "rag")
    else:
        pending = state.get("pending_subtasks", [])
        if pending:
            next_plan = pending[0]
            category = _category_from_plan(next_plan)
            route = _CATEGORY_TO_ROUTE.get(category, "final")
            return {
                "route": route if route in _ROLE_ALLOWED_ROUTES.get(role, set()) else "final",
                "plan": next_plan,
                "pending_subtasks": pending[1:],
                "hop_count": hop_count,
                "node_path": node_path,
            }
        route = "final"

    # Section 10 role gate: a route the caller's role isn't permitted to
    # reach is treated the same way a real deployment would - refuse the
    # route, not the whole request, and let 'final' compose an honest answer.
    if route not in _ROLE_ALLOWED_ROUTES.get(role, set()):
        return {
            "route": "final",
            "hop_count": hop_count,
            "agent_output": (
                f"This request isn't available for the '{role}' role on this platform."
            ),
            "node_path": node_path,
        }

    return {"route": route, "hop_count": hop_count, "node_path": node_path}
