"""
Section 6 of the design doc, translated into a LangGraph state schema —
simplified for readability. Keeping it flat and typed makes every node's
contract explicit: a node only reads the keys it needs and writes the
keys it owns. This IS the graph's "shared state" that flows between nodes
natively - no hand-wired message passing.

Simplification note: the reference doc's Orchestrator can in principle
queue any number of follow-up sub-intents. This build only ever needs one
(the developer's "docs + code task" example, Section 7.3), so instead of
a generic queue we use a single optional `next_category` field. If you
outgrow one follow-up, that's the field to turn back into a list.
"""
from __future__ import annotations

from typing import Literal, TypedDict, Optional


class ToolCallRecord(TypedDict):
    node: str
    tool: str
    input: dict
    output: dict
    ok: bool


class RetrievedChunk(TypedDict):
    text: str
    source: str
    score: float
    namespace: str


class OrchestratorState(TypedDict, total=False):
    # ── identity / input ─────────────────────────────────────────────
    session_id: str
    user_role: Literal["developer", "client", "customer"]
    user_query: str

    # ── planning / routing ───────────────────────────────────────────
    plan: Optional[str]                 # set by Planner Agent
    route: Optional[
        Literal["coder", "rag", "mcp", "external_agent", "final"]
    ]
    next_category: Optional[str]        # one queued follow-up (Section 7.3)

    # ── execution trail ───────────────────────────────────────────────
    tool_calls: list[ToolCallRecord]        # audit trail, Section 10
    retrieved_context: Optional[list[RetrievedChunk]]  # RAG output
    agent_output: Optional[str]             # latest node's raw output
    collected_outputs: list[str]            # every hop's output, merged in finalize
    node_path: list[str]                    # observability trace (Section 12)

    # ── termination ───────────────────────────────────────────────────
    final_answer: Optional[str]
    hop_count: int                          # guards runaway multi-hop loops


def fresh_turn(session_id: str, user_role: str, user_query: str) -> OrchestratorState:
    """Every field a new turn needs, reset to a clean slate. Pass this
    straight into graph.ainvoke() - see main.py."""
    return OrchestratorState(
        session_id=session_id,
        user_role=user_role,  # type: ignore[typeddict-item]
        user_query=user_query,
        plan=None,
        route=None,
        next_category=None,
        tool_calls=[],
        retrieved_context=None,
        agent_output=None,
        collected_outputs=[],
        node_path=[],
        final_answer=None,
        hop_count=0,
    )
