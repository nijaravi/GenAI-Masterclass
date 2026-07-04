"""
Section 6 of the design doc, translated 1:1 into a LangGraph state schema.

Keeping it flat and typed makes every node's contract explicit: a node only
reads the keys it needs and writes the keys it owns. This IS the graph's
"shared state" that the design doc says flows between nodes natively -
no hand-wired message passing.
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
    conversation_history: list[dict]

    # ── planning / routing ───────────────────────────────────────────
    plan: Optional[str]                 # set by Planner Agent
    route: Optional[
        Literal["coder", "rag", "mcp", "tool", "external_agent", "final"]
    ]
    pending_subtasks: list[str]         # multi-hop queue (Section 7.3)

    # ── execution trail ───────────────────────────────────────────────
    tool_calls: list[ToolCallRecord]        # audit trail, Section 10
    retrieved_context: Optional[list[RetrievedChunk]]  # RAG output
    agent_output: Optional[str]             # latest node's raw output
    collected_outputs: list[str]            # every hop's output, for multi-hop merge (Section 7.3)
    node_path: list[str]                    # observability trace (Section 12)

    # ── termination ───────────────────────────────────────────────────
    final_answer: Optional[str]
    hop_count: int                          # guards runaway multi-hop loops


def new_state(session_id: str, user_role: str, user_query: str) -> OrchestratorState:
    """Factory for a fresh turn. conversation_history is passed in by the
    caller (main.py) since it's persisted across turns by the checkpointer."""
    return OrchestratorState(
        session_id=session_id,
        user_role=user_role,  # type: ignore[typeddict-item]
        user_query=user_query,
        conversation_history=[],
        plan=None,
        route=None,
        pending_subtasks=[],
        tool_calls=[],
        retrieved_context=None,
        agent_output=None,
        collected_outputs=[],
        node_path=[],
        final_answer=None,
        hop_count=0,
    )
