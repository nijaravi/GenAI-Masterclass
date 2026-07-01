"""
pipeline.py — Top-level pipeline that wires all agents together.

Flow:
  User Message
      │
      ▼
  PlannerAgent        ← analyses intent
      │
      ▼
  OrchestratorAgent   ← decides route (RAG / tool_call / coder / mcp)
      │
      ├─── RAG ──────► RAGAgent
      ├─── TOOL ─────► ToolAgent
      ├─── CODER ────► CoderAgent
      └─── MCP ──────► MCPAgent
                            │
                            ▼
                       ChatResponse (answer + steps + route + sources)
"""
from typing import List

from backend.config import (
    ChatRequest,
    ChatResponse,
    RouteDecision,
    AgentStep,
)
from backend.agents.planner_agent import PlannerAgent
from backend.agents.orchestrator_agent import OrchestratorAgent
from backend.agents.rag_agent import RAGAgent
from backend.agents.tool_agent import ToolAgent
from backend.agents.coder_agent import CoderAgent
from backend.agents.mcp_agent import MCPAgent

# Singleton agent instances
_planner = PlannerAgent()
_orchestrator = OrchestratorAgent()
_rag = RAGAgent()
_tool = ToolAgent()
_coder = CoderAgent()
_mcp = MCPAgent()


def run_pipeline(request: ChatRequest) -> ChatResponse:
    """
    Execute the full multi-agent pipeline for a single user request.
    Returns a ChatResponse with the final answer and all agent trace steps.
    """
    steps: List[AgentStep] = []

    # ── Step 1: Planner ───────────────────────────────────────────────────
    planner_step = _planner.run(
        user_message=request.message,
        user_role=request.user_role,
    )
    steps.append(planner_step)

    # ── Step 2: Orchestrator (routing decision) ───────────────────────────
    orchestrator_step, route = _orchestrator.run(
        planner_step=planner_step,
        original_message=request.message,
        user_role=request.user_role,
    )
    steps.append(orchestrator_step)

    # ── Step 3: Specialist Agent ──────────────────────────────────────────
    if route == RouteDecision.RAG:
        specialist_step = _rag.run(request.message, request.user_role)

    elif route == RouteDecision.TOOL_CALL:
        specialist_step = _tool.run(request.message, request.user_role)

    elif route == RouteDecision.CODER:
        specialist_step = _coder.run(request.message, request.user_role)

    elif route == RouteDecision.MCP:
        specialist_step = _mcp.run(request.message, request.user_role)

    else:
        # Fallback to RAG if route is unknown
        specialist_step = _rag.run(request.message, request.user_role)
        route = RouteDecision.RAG

    steps.append(specialist_step)

    # ── Collect sources (from RAG step if applicable) ─────────────────────
    sources: List[str] = []
    if route == RouteDecision.RAG:
        sources = specialist_step.metadata.get("sources", [])

    return ChatResponse(
        answer=specialist_step.output,
        route_taken=route,
        steps=steps,
        user_role=request.user_role,
        sources=sources,
    )
