"""
The 'external_agent' route - the one Orchestrator destination the platform
does not own (Section 4.8). Worked example is Section 7.4: a client asking
about a return handled by a third-party logistics vendor.

On success or graceful degrade, this writes agent_output like any other
node and control returns to the Orchestrator - the design doc's point that
"everything upstream of the A2A node is the same LangGraph graph as the
other examples" holds here too.
"""
from __future__ import annotations

import re

from app.a2a.a2a_client import call_external_agent
from app.state import OrchestratorState, ToolCallRecord

_ORDER_ID_PATTERN = re.compile(r"\bord-\d+\b", re.IGNORECASE)


async def external_agent_node(state: OrchestratorState) -> dict:
    query = state["user_query"]
    match = _ORDER_ID_PATTERN.search(query)
    # Demo fallback order id so the happy path is reachable without the
    # caller having to know a seeded id in advance.
    order_id = match.group(0).lower() if match else "ord-7712"

    result = await call_external_agent("check_return_status", {"order_id": order_id})

    record = ToolCallRecord(
        node="external_agent",
        tool="a2a.check_return_status",
        input={"order_id": order_id},
        output=result.data or {"degraded_reason": result.degraded_reason},
        ok=result.ok,
    )

    if result.ok:
        output = (
            f"Your return for order {order_id} is currently "
            f"'{result.data['status']}' with carrier ETA {result.data['carrier_eta']}."
        )
    else:
        # Graceful degrade path (Section 4.8 / 11.1) - never let a slow or
        # unavailable external agent stall the whole request.
        output = (
            "I'm still checking with our logistics partner on this return - "
            "their system didn't respond in time. I'll have an update shortly "
            f"(reference: {order_id})."
        )

    return {
        "tool_calls": state.get("tool_calls", []) + [record],
        "agent_output": output,
        "collected_outputs": state.get("collected_outputs", []) + [f"[External vendor lookup]\n{output}"],
        "node_path": state.get("node_path", []) + ["external_agent"],
    }
