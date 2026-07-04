"""
Second MCP server from Section 9's example tool set:
`get_build_status(account_id)`. Kept as a separate server (rather than
bundled into billing_server.py) to mirror the real pattern - a new backend
system is onboarded as its own MCP server without touching the others or
the Orchestrator's core logic (Section 5.3).

Run standalone for manual testing:
    python -m app.mcp_servers.build_status_server
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("build-status-server")

_BUILD_DB = {
    "client-4471": {"account_id": "client-4471", "build_stage": "deploying",
                     "last_deploy_ts": "2026-07-03T21:40:00Z", "billing_flag": "ok"},
    "client-9002": {"account_id": "client-9002", "build_stage": "failed",
                     "last_deploy_ts": "2026-07-04T02:10:00Z", "billing_flag": "ok"},
}


@mcp.tool()
def get_build_status(account_id: str) -> dict:
    """Return build/CI status (stage, last deploy timestamp, billing flag)
    for a given client account_id."""
    record = _BUILD_DB.get(account_id)
    if record is None:
        return {"error": f"no build record for account_id={account_id}"}
    return record


if __name__ == "__main__":
    mcp.run(transport="stdio")
