"""
Section 4.3: 'Interprets intent and produces a short plan: what is actually
being asked, and what category of capability would answer it ... The
Planner does not execute anything itself - it hands its plan to the
Orchestrator, which owns the actual routing decision.'

The plan is deliberately just a short category string plus the original
query, not a route - Section 4.4 is explicit that the Orchestrator, not the
Planner, owns the routing decision. Keeping that split matches the doc and
is also what makes multi-hop possible: the Orchestrator can re-plan-free
route to a second destination using the same plan text (Section 7.3).
"""
from __future__ import annotations

from app.llm import get_chat_model
from app.state import OrchestratorState

_CATEGORY_LABEL = {
    "product_info_lookup": "informational lookup (product info)",
    "doc_lookup": "informational lookup (developer docs)",
    "account_status_internal": "account/status lookup (internal system)",
    "account_status_external": "account/status lookup (external/vendor system)",
    "code_task": "code-related task",
}


def _detect_categories(query: str) -> list[str]:
    """Lightweight compound-intent detection matching Section 7.3's worked
    example: a single developer turn combining a doc lookup with a code
    task. Real classification is delegated to the model below for the
    single-category case; this only exists to decide *how many* plans to
    queue, mirroring the Planner's job of surfacing sub-intents without
    itself routing them (routing stays the Orchestrator's job)."""
    q = query.lower()
    cats = []
    if any(k in q for k in ["api", "documentation", "how do i call", "reference", "endpoint"]):
        cats.append("doc_lookup")
    if any(k in q for k in ["pr #", "pull request", "jira", "coverage", "vulnerability"]):
        cats.append("code_task")
    return cats


async def planner_node(state: OrchestratorState) -> dict:
    model = get_chat_model()
    query = state["user_query"]
    compound = _detect_categories(query)

    if len(compound) >= 2:
        # Section 7.3: queue the remaining sub-intent(s) for a later hop;
        # the Orchestrator will route to the first now and come back for
        # the rest (hop_count tracks this).
        primary, *rest = compound
        pending = [f"{c}: {_CATEGORY_LABEL[c]} — '{query}'" for c in rest]
        plan = f"{primary}: {_CATEGORY_LABEL[primary]} — '{query}'"
        return {
            "plan": plan,
            "pending_subtasks": state.get("pending_subtasks", []) + pending,
            "node_path": state.get("node_path", []) + ["planner"],
        }

    prompt = (
        "Classify the user's intent into exactly one category: "
        "product_info_lookup, doc_lookup, account_status_internal, "
        "account_status_external, or code_task.\n"
        f"User role: {state['user_role']}\n"
        f"Request: {query}"
    )
    category = model.invoke(prompt).strip()
    if category not in _CATEGORY_LABEL:
        category = "product_info_lookup"

    plan = f"{category}: {_CATEGORY_LABEL[category]} — '{query}'"

    return {
        "plan": plan,
        "node_path": state.get("node_path", []) + ["planner"],
    }
