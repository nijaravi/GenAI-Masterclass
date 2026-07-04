"""
LangGraph node for the 'coder' route (Section 4.5 / 4.7). Its whole job is
the hand-off described in the design doc: send the crew a task, get back
one structured result, write it into shared state like any other node.
The graph doesn't know or care that a 3-agent CrewAI crew ran underneath.
"""
from __future__ import annotations

from app import config
from app.crew.coder_crew import build_and_run_crew, run_demo_crew
from app.state import OrchestratorState, ToolCallRecord


async def coder_agent_node(state: OrchestratorState) -> dict:
    import asyncio

    # CrewAI's kickoff() is sync; run it off the event loop thread so the
    # rest of the (async) graph isn't blocked.
    if config.LIVE_LLM:
        result = await asyncio.to_thread(build_and_run_crew)
    else:
        result = await asyncio.to_thread(run_demo_crew)

    record = ToolCallRecord(
        node="coder_agent",
        tool="crewai.coder_crew",
        input={"query": state["user_query"]},
        output={"result": result[:400]},
        ok=True,
    )

    return {
        "tool_calls": state.get("tool_calls", []) + [record],
        "agent_output": result,
        "collected_outputs": state.get("collected_outputs", []) + [f"[Code coverage/vulnerability check]\n{result}"],
        "node_path": state.get("node_path", []) + ["coder_agent"],
    }
