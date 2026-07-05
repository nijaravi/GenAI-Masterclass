"""
Every LangGraph node except the Planner and Orchestrator (which live in
their own files because they're the two most conceptually interesting
nodes) lives here, grouped by concern. Each one follows the same shape:
read what it needs from state, do its one job, return a small dict of the
keys it owns - LangGraph merges that into the shared state for you.
"""
from __future__ import annotations

import asyncio
import json
import re

from app.crew.coder_crew import run_crew
from app.llm import get_model
from app.mcp_servers.billing_server import mcp as billing_server
from app.mcp_servers.build_status_server import mcp as build_status_server
from app.a2a.vendor_agent import call_external_agent
from app.rag.vector_store import get_vector_store
from app.state import OrchestratorState, ToolCallRecord

# ═══════════════════════════════════════════════════════════════════════
# INTAKE  (Section 4.2)
# First stop for every request: lightly normalize the raw text and hand a
# clean payload downstream. Deliberately does not classify or route -
# that keeps malformed/adversarial input away from the reasoning nodes.
# ═══════════════════════════════════════════════════════════════════════
_MAX_QUERY_CHARS = 2000


def _normalize(text: str) -> str:
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return cleaned.strip()[:_MAX_QUERY_CHARS]


async def intake_node(state: OrchestratorState) -> dict:
    return {
        "user_query": _normalize(state["user_query"]),
        "node_path": state.get("node_path", []) + ["intake"],
    }


# ═══════════════════════════════════════════════════════════════════════
# RAG AGENT  (Section 4.6 / 8)
# Dedicated node, not a generic tool call, because retrieval has its own
# lifecycle. Namespace is picked by role so a customer can never retrieve
# developer docs and vice versa (Section 8.1 / 10).
# ═══════════════════════════════════════════════════════════════════════
_NAMESPACE_BY_ROLE = {
    "customer": "customer-product",
    "developer": "developer-docs",
}


async def rag_agent_node(state: OrchestratorState) -> dict:
    namespace = _NAMESPACE_BY_ROLE.get(state["user_role"], "customer-product")
    chunks = get_vector_store().query(namespace, state["user_query"], top_k=4)

    if not chunks:
        answer = "No matching passages were found for this query."
    else:
        context_block = "\n".join(f"- ({c['source']}) {c['text']}" for c in chunks)
        prompt = (
            "Answer the question using only the retrieved context below. "
            "Cite the source file for each claim. Be concise.\n\n"
            f"Question: {state['user_query']}\n\nRetrieved context:\n{context_block}"
        )
        # ChatOpenAI's .invoke() is a blocking network call - keep it off
        # the event loop thread, same as the Planner does.
        answer = await asyncio.to_thread(get_model().invoke, prompt)

    return {
        "retrieved_context": chunks,
        "agent_output": answer,
        "collected_outputs": state.get("collected_outputs", []) + [f"[Docs/Product lookup]\n{answer}"],
        "node_path": state.get("node_path", []) + ["rag_agent"],
    }


# ═══════════════════════════════════════════════════════════════════════
# CODER AGENT  (Section 4.5 / 4.7)
# Hands one task to the nested CrewAI crew and writes back one structured
# result - the graph doesn't know or care that three sub-agents ran
# underneath. See app/crew/coder_crew.py for the crew itself.
# ═══════════════════════════════════════════════════════════════════════
async def coder_agent_node(state: OrchestratorState) -> dict:
    # crew.kickoff() is sync and makes real LLM calls - run it off the
    # event loop thread so the rest of the (async) graph isn't blocked.
    result = await asyncio.to_thread(run_crew)

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


# ═══════════════════════════════════════════════════════════════════════
# MCP TOOL  (Section 4.5 / 9)
# Calls a real MCP server's tool - the servers in app/mcp_servers/ are
# genuine FastMCP servers with @mcp.tool()-decorated functions. Calling
# server.call_tool(...) directly (rather than opening a client session
# over a transport) keeps this build simple while still exercising the
# real tool-registration and tool-call machinery from the MCP SDK.
# ═══════════════════════════════════════════════════════════════════════
def _pick_server_and_tool(plan: str) -> tuple[object, str]:
    if "billing" in (plan or "").lower() or "invoice" in (plan or "").lower():
        return billing_server, "get_billing_summary"
    return build_status_server, "get_build_status"


async def mcp_tool_node(state: OrchestratorState) -> dict:
    server, tool_name = _pick_server_and_tool(state.get("plan", ""))
    # session_id doubles as the account_id lookup key for this demo's two
    # seeded records (client-4471 / client-9002). In production this would
    # come from the authenticated session context (Section 10), never
    # parsed from free text.
    account_id = state["session_id"]

    try:
        blocks = await server.call_tool(tool_name, {"account_id": account_id})
        output = json.loads(blocks[0].text)
        ok = "error" not in output
    except Exception as exc:  # pragma: no cover - defensive
        output = {"error": str(exc)}
        ok = False

    record = ToolCallRecord(
        node="mcp_tool", tool=f"{server.name}.{tool_name}",
        input={"account_id": account_id}, output=output, ok=ok,
    )
    return {
        "tool_calls": state.get("tool_calls", []) + [record],
        "agent_output": str(output),
        "collected_outputs": state.get("collected_outputs", []) + [f"[Account/status lookup]\n{output}"],
        "node_path": state.get("node_path", []) + ["mcp_tool"],
    }


# ═══════════════════════════════════════════════════════════════════════
# EXTERNAL AGENT  (Section 4.8 / 7.4 - the A2A boundary)
# The one Orchestrator destination the platform doesn't own. On timeout or
# failure it degrades gracefully instead of stalling the request.
# ═══════════════════════════════════════════════════════════════════════
_ORDER_ID_PATTERN = re.compile(r"\bord-\d+\b", re.IGNORECASE)


async def external_agent_node(state: OrchestratorState) -> dict:
    match = _ORDER_ID_PATTERN.search(state["user_query"])
    order_id = match.group(0).lower() if match else "ord-7712"  # demo fallback

    ok, result = await call_external_agent("check_return_status", order_id)

    record = ToolCallRecord(
        node="external_agent", tool="a2a.check_return_status",
        input={"order_id": order_id},
        output=result if ok else {"degraded_reason": result}, ok=ok,
    )

    if ok:
        output = (f"Your return for order {order_id} is currently "
                  f"'{result['status']}' with carrier ETA {result['carrier_eta']}.")
    else:
        output = (
            "I'm still checking with our logistics partner on this return - "
            f"their system didn't respond in time. Reference: {order_id}."
        )

    return {
        "tool_calls": state.get("tool_calls", []) + [record],
        "agent_output": output,
        "collected_outputs": state.get("collected_outputs", []) + [f"[External vendor lookup]\n{output}"],
        "node_path": state.get("node_path", []) + ["external_agent"],
    }


# ═══════════════════════════════════════════════════════════════════════
# FINALIZE
# Composes final_answer. If more than one hop produced output (the
# multi-hop developer example), merge them in order rather than keeping
# only the last one (Section 7.3: "Orchestrator merges both results").
# ═══════════════════════════════════════════════════════════════════════
async def finalize_node(state: OrchestratorState) -> dict:
    collected = state.get("collected_outputs", [])
    answer = "\n\n".join(collected) if len(collected) >= 2 else (
        state.get("agent_output") or "I don't have an answer for that yet."
    )
    return {
        "final_answer": answer,
        "node_path": state.get("node_path", []) + ["finalize"],
    }
