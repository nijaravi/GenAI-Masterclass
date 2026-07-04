"""
Composes the final_answer once the Orchestrator sets route='final'.
Not named as a distinct agent in the design doc - Section 3 just says
execution results "flow back through the graph to the user as a single
coherent answer" - so this is that composition step, kept as its own node
so it's a visible, traceable last hop in node_path (Section 12).
"""
from __future__ import annotations

from app.state import OrchestratorState


async def finalize_node(state: OrchestratorState) -> dict:
    collected = state.get("collected_outputs", [])
    if len(collected) >= 2:
        # Section 7.3: "Orchestrator merges both results into one response."
        answer = "\n\n".join(collected)
    else:
        answer = state.get("agent_output") or "I don't have an answer for that yet."
    return {
        "final_answer": answer,
        "node_path": state.get("node_path", []) + ["finalize"],
    }
