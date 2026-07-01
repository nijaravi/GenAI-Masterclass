"""
mcp_agent.py — Model Context Protocol (MCP) integration agent.

Responsibility: Handle requests that require external system integration
via the MCP standard. In production this would connect to real MCP servers
(e.g., Salesforce, GitHub, Jira, Google Calendar).

Demo MCP servers simulated:
  - supplier-crm-mcp       (Walmart supplier relationship CRM)
  - order-management-mcp   (Walmart order fulfilment system)
  - ticketing-mcp          (Internal IT ticketing)
  - calendar-mcp           (Meeting scheduling)

MCP Protocol basics (for interview explanation):
  - Each MCP server exposes a set of "tools" discoverable via GET /tools
  - Claude calls POST /tools/{tool_name} with JSON arguments
  - Servers return structured JSON results
  - This allows LLMs to use ANY external API through a standard interface
"""
import re
from typing import Dict, Any, Optional
from backend.config import AgentStep, UserRole
from backend.utils.llm_client import run_mcp_agent

# ─── Mock MCP Server Responses ───────────────────────────────────────────────

MOCK_MCP_SERVERS = {
    "supplier-crm-mcp": {
        "description": "Walmart supplier CRM — contacts, POs, interactions",
        "tools": ["get_supplier_contact", "list_open_pos", "create_crm_note"],
        "mock_response": {
            "server": "supplier-crm-mcp",
            "tool": "get_supplier_contact",
            "result": {
                "supplier_id": "SUP-48291",
                "name": "Acme Consumer Goods LLC",
                "contact": "Sarah Chen",
                "email": "s.chen@acmecg.com",
                "last_interaction": "2024-11-19",
                "open_pos": 3,
                "ytd_gmv": "$4.2M",
            },
        },
    },
    "order-management-mcp": {
        "description": "Walmart order fulfilment — create, update, cancel orders",
        "tools": ["get_order_details", "update_order_status", "create_return"],
        "mock_response": {
            "server": "order-management-mcp",
            "tool": "get_order_details",
            "result": {
                "order_id": "WMT-2024-88421",
                "status": "Fulfilling",
                "items": 2,
                "destination": "Phoenix, AZ 85001",
                "sla_breach_risk": False,
            },
        },
    },
    "ticketing-mcp": {
        "description": "Internal IT ticketing — create and track tickets",
        "tools": ["create_ticket", "get_ticket_status", "escalate_ticket"],
        "mock_response": {
            "server": "ticketing-mcp",
            "tool": "create_ticket",
            "result": {
                "ticket_id": "TECH-29841",
                "priority": "P2",
                "assigned_to": "Platform Team",
                "sla_resolution": "4 hours",
                "status": "Open",
            },
        },
    },
    "calendar-mcp": {
        "description": "Google Calendar / Outlook integration — schedule meetings",
        "tools": ["create_event", "find_free_slots", "send_invite"],
        "mock_response": {
            "server": "calendar-mcp",
            "tool": "find_free_slots",
            "result": {
                "available_slots": [
                    "2024-11-25 10:00 AM EST",
                    "2024-11-25 02:00 PM EST",
                    "2024-11-26 09:00 AM EST",
                ],
                "timezone": "America/New_York",
            },
        },
    },
}


def _select_mcp_server(message: str) -> str:
    lower = message.lower()
    if re.search(r"supplier|vendor|partner|crm", lower):
        return "supplier-crm-mcp"
    if re.search(r"order|fulfilment|shipment", lower):
        return "order-management-mcp"
    if re.search(r"ticket|issue|bug|incident|support", lower):
        return "ticketing-mcp"
    if re.search(r"meeting|calendar|schedule|slot|available", lower):
        return "calendar-mcp"
    return "supplier-crm-mcp"


def _format_mcp_response(server_name: str, mcp_data: Dict[str, Any]) -> str:
    result = mcp_data["result"]
    tool = mcp_data["tool"]

    lines = [
        f"**MCP Server: `{server_name}`** | Tool: `{tool}`\n",
        "---",
    ]
    for key, value in result.items():
        label = key.replace("_", " ").title()
        lines.append(f"- **{label}:** {value}")

    lines.append(
        f"\n*Connected via MCP protocol — server `{server_name}` responded with structured data.*"
    )
    return "\n".join(lines)


class MCPAgent:
    """
    MCPAgent selects the appropriate MCP server and invokes a tool
    to handle requests needing external system integration.
    """

    name = "MCPAgent"

    def run(self, user_message: str, user_role: UserRole) -> AgentStep:
        server_name = _select_mcp_server(user_message)
        server_config = MOCK_MCP_SERVERS[server_name]
        mock_response = server_config["mock_response"]

        answer = _format_mcp_response(server_name, mock_response)

        return AgentStep(
            agent=self.name,
            input=user_message,
            output=answer,
            metadata={
                "mcp_server": server_name,
                "tool_invoked": mock_response["tool"],
                "available_tools": server_config["tools"],
                "protocol": "MCP (Model Context Protocol)",
            },
        )
