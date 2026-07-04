"""
Section 4.1: 'One conversational UI for all three audiences. It
authenticates the user, attaches their role ... and forwards every message
into the agent graph.'

This is that surface: a single POST /chat endpoint. The caller supplies
user_role (standing in for real authentication - Section 10 notes role is
attached at auth time in production, not chosen by the user) and the
LangGraph checkpointer keeps each session's state under its session_id as
the thread_id, so conversation state persists across turns without this
file managing any of that itself.
"""
from __future__ import annotations

import atexit
import subprocess
import sys
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import config
from app.graph import get_graph
from app.rag.vector_store import seed_if_empty

_vendor_process: subprocess.Popen | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_if_empty()

    global _vendor_process
    if config.DEMO_MODE:
        # Auto-launch the external vendor agent (Section 4.8) as a
        # separate process, purely for demo convenience - in reality this
        # is a service the platform doesn't run at all.
        _vendor_process = subprocess.Popen(
            [sys.executable, "-m", "app.a2a.external_vendor_agent"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_vendor_agent()

    yield

    if _vendor_process is not None:
        _vendor_process.terminate()


def _wait_for_vendor_agent(timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(f"{config.EXTERNAL_AGENT_URL}/.well-known/agent.json", timeout=0.5)
            return
        except httpx.HTTPError:
            time.sleep(0.2)


app = FastAPI(
    title="Multi-Audience Multi-Agent Orchestrator (Capstone)",
    description=(
        "A single chat interface serving developers, clients, and "
        "customers, routed by a LangGraph orchestrator to RAG, MCP tools, "
        "a CrewAI-backed Coder Agent, or an A2A external agent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    user_role: str  # "developer" | "client" | "customer"
    message: str


class ChatResponse(BaseModel):
    final_answer: str
    route_taken: list[str]
    tool_calls: list[dict]
    hop_count: int


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.user_role not in ("developer", "client", "customer"):
        raise HTTPException(400, "user_role must be developer, client, or customer")

    graph = get_graph()
    thread_config = {"configurable": {"thread_id": req.session_id}}

    turn_input = {
        "session_id": req.session_id,
        "user_role": req.user_role,
        "user_query": req.message,
        # Reset turn-scoped fields explicitly - the checkpointer would
        # otherwise carry last turn's route/hop_count/agent_output forward.
        "plan": None,
        "route": None,
        "pending_subtasks": [],
        "retrieved_context": None,
        "agent_output": None,
        "collected_outputs": [],
        "final_answer": None,
        "node_path": [],
        "hop_count": 0,
    }

    try:
        result = await graph.ainvoke(turn_input, config=thread_config)
    except Exception as exc:
        raise HTTPException(500, f"graph execution failed: {exc}") from exc

    return ChatResponse(
        final_answer=result.get("final_answer", ""),
        route_taken=result.get("node_path", []),
        tool_calls=result.get("tool_calls", []),
        hop_count=result.get("hop_count", 0),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "demo_mode": config.DEMO_MODE,
        "live_llm": config.LIVE_LLM,
        "live_pinecone": config.LIVE_PINECONE,
    }


@app.get("/personas")
def personas():
    return {
        "developer": "Documentation lookups (RAG) and code tasks (CrewAI-backed Coder Agent).",
        "client": "Account/billing/build status (MCP) and external vendor lookups (A2A).",
        "customer": "Product information (RAG).",
    }


@app.get("/agent-map")
def agent_map():
    return {
        "intake": "Normalizes the raw request, attaches session context.",
        "planner": "Classifies intent into a short plan; does not route.",
        "orchestrator": "Owns the routing decision; revisited after every specialist node (multi-hop).",
        "rag": "Pinecone/Chroma-backed retrieval, namespace-isolated per audience.",
        "coder": "Nested CrewAI crew: code-retrieval, requirements-reading, review sub-agents.",
        "mcp": "Real MCP client calling billing/build-status MCP servers.",
        "external_agent": "A2A boundary node calling an independently-run vendor agent.",
        "finalize": "Composes the final_answer once route == 'final'.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
