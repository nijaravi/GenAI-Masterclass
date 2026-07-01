"""
llm_client.py — OpenAI wrapper with graceful demo-mode fallback.

When DEMO_MODE=true, all LLM calls return deterministic canned responses
so the full agent pipeline runs without spending API credits.
"""
import json
import re
from typing import List, Dict, Any, Optional
from backend.config import OPENAI_API_KEY, LLM_MODEL, DEMO_MODE, RouteDecision

# ─── Demo-mode canned responses ──────────────────────────────────────────────

DEMO_PLANNER: Dict[str, str] = {
    "default": "The user wants information that may require retrieval from the knowledge base. I'll prepare a structured intent summary: user seeks factual information. Best strategy: retrieve relevant docs, then synthesize an answer.",
}

DEMO_ROUTES: Dict[str, RouteDecision] = {
    # Coder routes — checked FIRST (more specific)
    r"write.*code|generate.*code|implement|script|function|class|how do i.*code|code.*example|code.*snippet|python.*api|write.*function|show.*code": RouteDecision.CODER,
    # MCP routes — checked SECOND
    r"mcp|model context protocol|crm server|check.*server|calendar|jira|ticket|github|slack|salesforce|external.*system|open.*ticket|create.*ticket": RouteDecision.MCP,
    # Tool-call routes
    r"calculate|compute|discount|how much|total cost|inventory|stock level|billing.*summary|show.*billing|check.*stock": RouteDecision.TOOL_CALL,
    # RAG fallback (broad)
    r".*": RouteDecision.RAG,
}

DEMO_CODER_RESPONSE = """Here is a Python implementation:

```python
import requests

def get_walmart_product(item_id: str, bearer_token: str) -> dict:
    \"\"\"Fetch product details from Walmart Item API v3.\"\"\"
    url = f"https://marketplace.walmartapis.com/v3/items/{item_id}"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "WM_SVC.NAME": "Walmart Marketplace",
        "WM_QOS.CORRELATION_ID": "demo-correlation-id",
        "Accept": "application/json",
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()
```

This function hits the Walmart Item API with proper authentication headers. 
Token acquisition uses the OAuth 2.0 client credentials flow — call `/v3/token` first."""

DEMO_TOOL_RESPONSE = {
    "calculate_discount": "Based on the tool calculation: $1,299 × 0.62 discount = $805.38 savings. Final price: **$493.62**. This matches the advertised Black Friday price of $499 (slight difference due to rounding in the promotion).",
    "default": "Tool executed successfully. Result: The computed value based on your inputs is **$247.50**. This includes applicable taxes and Walmart+ member discount of 5%.",
}

DEMO_MCP_RESPONSE = "MCP server integration triggered. Connected to the Walmart Supplier CRM via MCP protocol. Retrieved latest interaction log: last contact was 3 days ago regarding PO #WMT-2024-88421. No pending action items flagged."


def _demo_route(message: str) -> RouteDecision:
    msg_lower = message.lower()
    for pattern, route in DEMO_ROUTES.items():
        if re.search(pattern, msg_lower):
            return route
    return RouteDecision.RAG


def _demo_rag_answer(question: str, context_docs: List[Dict]) -> str:
    if not context_docs:
        return "I couldn't find specific information about that in the Walmart knowledge base. Please contact support for more details."
    top = context_docs[0]
    return f"Based on the Walmart knowledge base: {top['text']}"


# ─── Real OpenAI client ───────────────────────────────────────────────────────

def _get_openai_client():
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


def chat_completion(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    """
    Single-turn chat completion.
    Returns the assistant's text content.
    """
    if DEMO_MODE or not OPENAI_API_KEY:
        # Return a plausible demo string so the pipeline stays runnable
        return f"[DEMO] Processed: {user_message[:120]}"

    client = _get_openai_client()
    if not client:
        return f"[DEMO] Processed: {user_message[:120]}"

    kwargs: Dict[str, Any] = {
        "model": LLM_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ─── Agent-specific helpers ───────────────────────────────────────────────────

def run_planner(user_message: str, persona: str) -> str:
    """Planner agent: decompose user intent into a structured plan."""
    if DEMO_MODE or not OPENAI_API_KEY:
        return (
            f"Planner analysis for [{persona}] query: '{user_message[:80]}'. "
            "Intent: information retrieval. Confidence: high. "
            "Recommended strategy: route to appropriate specialist agent."
        )
    system = (
        "You are the Planner Agent in a multi-agent AI platform. "
        "Your job is to analyse the user's question and produce a brief intent summary "
        "(2-3 sentences max). Identify: what the user wants, their persona type, "
        "and whether this needs retrieval, computation, code generation, or external tool use. "
        f"User persona: {persona}."
    )
    return chat_completion(system, user_message, temperature=0.1)


def run_orchestrator_routing(planner_output: str, user_message: str, persona: str) -> RouteDecision:
    """Orchestrator agent: decide which specialist to invoke."""
    if DEMO_MODE or not OPENAI_API_KEY:
        return _demo_route(user_message)

    system = """You are the Orchestrator Agent. Given a planner's intent summary and original query,
decide the best routing. Reply with ONLY a JSON object like:
{"route": "rag"}   -- options: rag | tool_call | coder | mcp

Rules:
- rag: factual Q&A, document lookup, policy questions
- tool_call: calculations, structured data lookups (pricing, inventory)
- coder: requests to write / explain / debug code
- mcp: tasks needing external systems (CRM, ticketing, calendar, email)"""

    prompt = f"Planner output: {planner_output}\n\nOriginal query: {user_message}\nPersona: {persona}"
    try:
        raw = chat_completion(system, prompt, json_mode=True)
        data = json.loads(raw)
        return RouteDecision(data.get("route", "rag"))
    except Exception:
        return _demo_route(user_message)


def run_rag_agent(question: str, context_docs: List[Dict], persona: str) -> str:
    """RAG agent: synthesise answer from retrieved documents."""
    if DEMO_MODE or not OPENAI_API_KEY:
        return _demo_rag_answer(question, context_docs)

    context_text = "\n\n".join(
        f"[Source {i+1}] {doc['text']}" for i, doc in enumerate(context_docs)
    )
    system = (
        f"You are a helpful assistant on Walmart's AI Platform serving a {persona}. "
        "Answer questions using ONLY the provided context. "
        "If the context doesn't contain the answer, say so honestly. "
        "Be concise and professional."
    )
    user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"
    return chat_completion(system, user_msg, temperature=0.3)


def run_coder_agent(request: str, persona: str) -> str:
    """Coder agent: generate or explain code."""
    if DEMO_MODE or not OPENAI_API_KEY:
        return DEMO_CODER_RESPONSE

    system = (
        f"You are a coding assistant for Walmart's internal {persona} platform. "
        "Write clean, well-commented Python (or the language requested). "
        "Always follow Walmart API conventions (OAuth 2.0, proper headers). "
        "Keep responses focused and practical."
    )
    return chat_completion(system, request, temperature=0.2)


def run_tool_agent(request: str) -> str:
    """Tool-call agent: invoke deterministic tools (pricing, inventory, etc.)."""
    if DEMO_MODE or not OPENAI_API_KEY:
        lower = request.lower()
        if "discount" in lower or "price" in lower:
            return DEMO_TOOL_RESPONSE["calculate_discount"]
        return DEMO_TOOL_RESPONSE["default"]

    system = (
        "You are a tool-call agent. Use the available tools to answer the user's request. "
        "Available tools: calculate_price_discount, check_inventory_level, get_billing_summary. "
        "Show your work step by step."
    )
    return chat_completion(system, request, temperature=0.1)


def run_mcp_agent(request: str, persona: str) -> str:
    """MCP agent: connect to external MCP servers."""
    if DEMO_MODE or not OPENAI_API_KEY:
        return DEMO_MCP_RESPONSE

    system = (
        f"You are an MCP (Model Context Protocol) agent for Walmart's {persona} platform. "
        "You have access to external MCP servers: supplier-crm-mcp, order-management-mcp, "
        "ticketing-mcp, calendar-mcp. Describe what action you would take and what the "
        "MCP server would return."
    )
    return chat_completion(system, request, temperature=0.2)
