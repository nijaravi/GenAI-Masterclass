"""
Section 5.1: 'Each agent becomes a node in a StateGraph, the Orchestrator's
routing decision becomes a conditional edge, and the shared state ...
flows between nodes natively via LangGraph's typed state object.'

This module is that graph, built exactly that way:

    intake -> planner -> orchestrator --(conditional edge on state["route"])--> {
                  ^                        rag           -> orchestrator
                  |                        coder          -> orchestrator
                  |                        mcp            -> orchestrator
                  |                        external_agent  -> orchestrator
                  |                        final          -> finalize -> END
                  +------------------------(loop back for multi-hop)---+

Every specialist node returns to the Orchestrator rather than going
straight to finalize - that's what makes multi-hop (Section 7.3) and the
hop_count guard (Section 6) both live in one place.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.nodes.coder_agent import coder_agent_node
from app.nodes.external_agent import external_agent_node
from app.nodes.finalize import finalize_node
from app.nodes.intake import intake_node
from app.nodes.mcp_tool import mcp_tool_node
from app.nodes.orchestrator import orchestrator_node
from app.nodes.planner import planner_node
from app.nodes.rag_agent import rag_agent_node
from app.state import OrchestratorState


def _route_from_state(state: OrchestratorState) -> str:
    """The conditional-edge function Section 5.1 refers to: reads
    state['route'] (set by the Orchestrator node) and returns the LangGraph
    edge key to follow next."""
    return state.get("route", "final")


def build_graph(checkpointer=None):
    graph = StateGraph(OrchestratorState)

    graph.add_node("intake", intake_node)
    graph.add_node("planner", planner_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("rag", rag_agent_node)
    graph.add_node("coder", coder_agent_node)
    graph.add_node("mcp", mcp_tool_node)
    graph.add_node("external_agent", external_agent_node)
    graph.add_node("finalize", finalize_node)

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
    # (Section 4.4: "After the destination agent returns a result, control
    # comes back to the Orchestrator").
    for node in ("rag", "coder", "mcp", "external_agent"):
        graph.add_edge(node, "orchestrator")

    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


_compiled = None


def get_graph():
    """Process-wide singleton compiled graph with an in-memory
    checkpointer. Section 11 notes a Postgres-backed checkpointer for
    production durability - swapping MemorySaver for
    langgraph.checkpoint.postgres.PostgresSaver(CHECKPOINTER_DB_URL) is the
    only change needed; nothing in the node code depends on which
    checkpointer is active."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
