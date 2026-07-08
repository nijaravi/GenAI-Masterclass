"""
Section 5.1: 'Each agent becomes a node in a StateGraph, the Orchestrator's
routing decision becomes a conditional edge, and the shared state ...
flows between nodes natively via LangGraph's typed state object.'

This module is that graph:

    intake -> planner -> orchestrator --(conditional edge on state["route"])--> {
                  ^                        rag            -> orchestrator
                  |                        coder          -> orchestrator
                  |                        mcp            -> orchestrator
                  |                        external_agent -> orchestrator
                  |                        final          -> finalize -> END
                  +------------------------(loop back for multi-hop)----+

Every specialist node returns to the Orchestrator rather than going
straight to finalize - that's what makes multi-hop (Section 7.3) and the
hop_count guard (Section 6) both live in one place.
"""
from __future__ import annotations

from app import config  # noqa: F401 — LangSmith env vars before LangGraph import

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.graph_logger import log_stage, snapshot_state
from app.nodes.orchestrator import orchestrator_node
from app.nodes.planner import planner_node
from app.nodes.specialists import (
    coder_agent_node,
    external_agent_node,
    finalize_node,
    intake_node,
    mcp_tool_node,
    rag_agent_node,
)
from app.state import OrchestratorState


def _wrap_node(node_name: str, fn):
    """Log each node's IN/OUT and handover target after it runs."""
    async def logged_node(state: OrchestratorState) -> dict:
        session_id = state.get("session_id", "unknown")
        state_in = snapshot_state(state, node_name)
        result = await fn(state)
        log_stage(
            node_name=node_name,
            session_id=session_id,
            state_in=state_in,
            state_out=result,
        )
        return result

    return logged_node


def _route_from_state(state: OrchestratorState) -> str:
    """The conditional-edge function: reads state['route'] (set by the
    Orchestrator node) and returns the LangGraph edge key to follow next."""
    return state.get("route", "final")


def build_graph():
    graph = StateGraph(OrchestratorState)

    graph.add_node("intake", _wrap_node("intake", intake_node))
    graph.add_node("planner", _wrap_node("planner", planner_node))
    graph.add_node("orchestrator", _wrap_node("orchestrator", orchestrator_node))
    graph.add_node("rag", _wrap_node("rag", rag_agent_node))
    graph.add_node("coder", _wrap_node("coder", coder_agent_node))
    graph.add_node("mcp", _wrap_node("mcp", mcp_tool_node))
    graph.add_node("external_agent", _wrap_node("external_agent", external_agent_node))
    graph.add_node("finalize", _wrap_node("finalize", finalize_node))

    graph.set_entry_point("intake")
    graph.add_edge("intake", "planner")
    graph.add_edge("planner", "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        _route_from_state,
        {
            "rag": "rag",
            "coder": "coder",
            "mcp": "mcp",
            "external_agent": "external_agent",
            "final": "finalize",
        },
    )

    # Every specialist node hands control back to the Orchestrator
    # (Section 4.4: "control comes back to the Orchestrator").
    for node in ("rag", "coder", "mcp", "external_agent"):
        graph.add_edge(node, "orchestrator")

    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())


_compiled = None


def get_graph():
    """Process-wide singleton compiled graph with an in-memory
    checkpointer. Swapping MemorySaver() for a Postgres-backed checkpointer
    is a one-line change here whenever you need conversation state to
    survive a process restart - nothing else in the codebase depends on it."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
