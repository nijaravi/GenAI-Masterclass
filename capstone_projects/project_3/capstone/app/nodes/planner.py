"""
Section 4.3: 'Interprets intent and produces a short plan ... The Planner
does not execute anything itself - it hands its plan to the Orchestrator,
which owns the actual routing decision.'

The plan is a category string, not a route - Section 4.4 is explicit that
routing is the Orchestrator's job, not the Planner's. That split is also
what makes the one multi-hop case in this build possible (Section 7.3): a
developer request can carry both a doc-lookup and a code-task intent in
one turn. When it does, the Planner classifies the primary intent now and
queues the other one in `next_category` for the Orchestrator to pick up
after the first hop completes.
"""
from __future__ import annotations

import asyncio

from app.llm import get_model
from app.state import OrchestratorState

_CATEGORY_LABEL = {
    "product_info_lookup": "informational lookup (product info)",
    "doc_lookup": "informational lookup (developer docs)",
    "account_status_internal": "account/status lookup (internal system)",
    "account_status_external": "account/status lookup (external/vendor system)",
    "code_task": "code-related task",
}

_CLASSIFY_PROMPT = (
    "Classify the user's intent into exactly one of these categories:\n"
    "product_info_lookup, doc_lookup, account_status_internal, "
    "account_status_external, code_task.\n"
    "Respond with only the category name, nothing else.\n\n"
    "User role: {role}\nRequest: {query}"
)


def _detect_categories(query: str) -> list[str]:
    """Cheap compound-intent detection for Section 7.3's worked example.
    Real single-category classification is delegated to the model below;
    this only decides whether there's a second intent to queue."""
    q = query.lower()
    cats = []
    if any(k in q for k in ["api", "documentation", "how do i call", "reference", "endpoint"]):
        cats.append("doc_lookup")
    if any(k in q for k in ["pr #", "pull request", "jira", "coverage", "vulnerability"]):
        cats.append("code_task")
    return cats


def _plan_text(category: str, query: str) -> str:
    return f"{category}: {_CATEGORY_LABEL[category]} — '{query}'"


def _extract_category(model_output: str) -> str:
    """Real models don't always follow 'respond with only X' perfectly -
    scan the response for a known category rather than requiring an exact
    match, and fall back to the safest default if none is found."""
    text = model_output.strip().lower()
    for category in _CATEGORY_LABEL:
        if category in text:
            return category
    return "product_info_lookup"


async def planner_node(state: OrchestratorState) -> dict:
    query = state["user_query"]
    compound = _detect_categories(query)

    if len(compound) >= 2:
        primary, follow_up = compound[0], compound[1]
        return {
            "plan": _plan_text(primary, query),
            "next_category": follow_up,
            "node_path": state.get("node_path", []) + ["planner"],
        }

    prompt = _CLASSIFY_PROMPT.format(role=state["user_role"], query=query)
    # ChatOpenAI's .invoke() is a blocking network call - run it off the
    # event loop thread so the rest of the (async) graph isn't blocked.
    raw = await asyncio.to_thread(get_model().invoke, prompt)
    category = _extract_category(raw)

    return {
        "plan": _plan_text(category, query),
        "node_path": state.get("node_path", []) + ["planner"],
    }
