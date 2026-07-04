"""
The 'mcp' route from Section 4.5 / 4.4: the Orchestrator invokes a backend
system exposed as an MCP server. This node is a real MCP client - it
spawns the target server as a subprocess over stdio, opens an MCP
ClientSession, and calls the tool - exactly the protocol Section 9
describes, just with the two demo servers in app/mcp_servers/ standing in
for a real billing/CI system.

Which server+tool to call is decided from the plan text set by the
Planner (Section 7.2's worked example: 'build status' -> build-status
server; 'billing' -> billing server). In a fuller build this mapping would
itself be a small registry the Orchestrator consults so new MCP servers
can be onboarded by registration alone (Section 5.3) - kept as a simple
if/else here to keep the demo legible.
"""
from __future__ import annotations

import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.state import OrchestratorState, ToolCallRecord

_SERVER_MODULES = {
    "billing": "app.mcp_servers.billing_server",
    "build_status": "app.mcp_servers.build_status_server",
}


def _pick_tool(plan: str) -> tuple[str, str]:
    """Returns (server_key, tool_name) based on the plan text."""
    p = (plan or "").lower()
    if "billing" in p or "invoice" in p:
        return "billing", "get_billing_summary"
    return "build_status", "get_build_status"


async def _call_mcp_tool(server_key: str, tool_name: str, account_id: str) -> dict:
    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", _SERVER_MODULES[server_key]]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, {"account_id": account_id})
            # MCP tool results come back as content blocks; unwrap the text/json.
            for block in result.content:
                if hasattr(block, "text"):
                    import json

                    try:
                        return json.loads(block.text)
                    except (json.JSONDecodeError, TypeError):
                        return {"raw": block.text}
            return {"error": "empty MCP result"}


async def mcp_tool_node(state: OrchestratorState) -> dict:
    server_key, tool_name = _pick_tool(state.get("plan", ""))
    # In this demo, session_id doubles as the account_id lookup key so the
    # two seeded MCP records (client-4471 / client-9002) are reachable -
    # in production this would come from the authenticated session context
    # (Section 10), never parsed from free text.
    account_id = state["session_id"]

    try:
        output = await _call_mcp_tool(server_key, tool_name, account_id)
        ok = "error" not in output
    except Exception as exc:  # pragma: no cover - defensive
        output = {"error": str(exc)}
        ok = False

    record = ToolCallRecord(
        node="mcp_tool",
        tool=f"{server_key}.{tool_name}",
        input={"account_id": account_id},
        output=output,
        ok=ok,
    )

    return {
        "tool_calls": state.get("tool_calls", []) + [record],
        "agent_output": str(output),
        "collected_outputs": state.get("collected_outputs", []) + [f"[Account/status lookup]\n{output}"],
        "node_path": state.get("node_path", []) + ["mcp_tool"],
    }
