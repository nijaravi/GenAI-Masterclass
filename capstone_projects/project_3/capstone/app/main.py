"""
Section 4.1: 'One conversational UI for all three audiences. It
authenticates the user, attaches their role ... and forwards every message
into the agent graph.'

This is that surface: a single POST /chat endpoint. user_role stands in
for real authentication (Section 10 notes role is attached at auth time
in production, not chosen by the user). The LangGraph checkpointer keeps
each session's state under its session_id as the thread_id, so
conversation state persists across turns without this file managing any
of that itself.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import config  # noqa: F401 — LangSmith env vars before graph/LLM imports
from app.graph import get_graph
from app.graph_logger import log_turn_end, log_turn_error, log_turn_start
from app.rag.vector_store import seed_if_empty
from app.state import fresh_turn
from app.tracing import build_graph_run_config, is_tracing_enabled


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_if_empty()
    yield


app = FastAPI(
    title="Multi-Audience Multi-Agent Orchestrator (Capstone)",
    description=(
        "A single chat interface serving developers, clients, and "
        "customers, routed by a LangGraph orchestrator to RAG, an MCP "
        "tool, a CrewAI-backed Coder Agent, or an A2A external agent."
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
    turn_input = fresh_turn(req.session_id, req.user_role, req.message)
    turn_id = f"{req.session_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"

    log_turn_start(
        session_id=req.session_id,
        user_role=req.user_role,
        message=req.message,
        turn_id=turn_id,
    )

    try:
        result = await graph.ainvoke(
            turn_input,
            config=build_graph_run_config(
                session_id=req.session_id,
                user_role=req.user_role,
                turn_id=turn_id,
                message=req.message,
            ),
        )
    except Exception as exc:
        log_turn_error(turn_id=turn_id, session_id=req.session_id, error=str(exc))
        raise HTTPException(500, f"graph execution failed: {exc}") from exc

    response = ChatResponse(
        final_answer=result.get("final_answer", ""),
        route_taken=result.get("node_path", []),
        tool_calls=result.get("tool_calls", []),
        hop_count=result.get("hop_count", 0),
    )
    log_turn_end(
        turn_id=turn_id,
        session_id=req.session_id,
        response=response.model_dump(),
        node_path=result.get("node_path", []),
        hop_count=result.get("hop_count", 0),
    )
    return response


@app.get("/health")
def health():
    from app import config

    return {
        "status": "ok",
        "langsmith_tracing": is_tracing_enabled(),
        "langsmith_project": config.LANGSMITH_PROJECT if is_tracing_enabled() else None,
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
        "intake": "Normalizes the raw request.",
        "planner": "Classifies intent into a short plan; does not route.",
        "orchestrator": "Owns the routing decision; revisited after every specialist node (multi-hop).",
        "rag": "Chroma-backed retrieval, namespace-isolated per audience.",
        "coder": "Nested CrewAI crew: code-retrieval, requirements-reading, review sub-agents.",
        "mcp": "Real MCP tool call against the billing/build-status MCP servers.",
        "external_agent": "A2A-style boundary node calling the simulated vendor agent.",
        "finalize": "Composes the final_answer once route == 'final'.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
