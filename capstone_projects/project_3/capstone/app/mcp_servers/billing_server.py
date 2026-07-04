"""
Section 9: 'Backend systems ... are exposed to the Orchestrator as MCP
servers rather than bespoke integrations per system.'

This is a real MCP server (built with the official `mcp` SDK's FastMCP),
run as a subprocess over stdio and called by app/nodes/mcp_tool.py through
a standard MCP ClientSession. It stands in for a billing system that, in
production, would live behind its own service - the point of MCP is that
the Orchestrator never needs to know that detail, only the tool contract
below.

Run standalone for manual testing:
    python -m app.mcp_servers.billing_server
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("billing-server")

# Toy in-memory "billing system" so the tool has something real to look up.
_BILLING_DB = {
    "client-4471": {"account_id": "client-4471", "balance_usd": 1280.50,
                     "last_invoice_date": "2026-06-01", "status": "current"},
    "client-9002": {"account_id": "client-9002", "balance_usd": 0.0,
                     "last_invoice_date": "2026-05-15", "status": "current"},
}


@mcp.tool()
def get_billing_summary(account_id: str) -> dict:
    """Return the billing summary (balance, last invoice date, status)
    for a given client account_id. Server-side, this call is scoped to
    the authenticated account only - see Section 9's authorization note."""
    record = _BILLING_DB.get(account_id)
    if record is None:
        return {"error": f"no billing record for account_id={account_id}"}
    return record


if __name__ == "__main__":
    mcp.run(transport="stdio")
