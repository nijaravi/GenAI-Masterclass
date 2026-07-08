"""
LangSmith observability for the orchestrator graph.

Set LANGSMITH_API_KEY and LANGSMITH_TRACING=true in .env to send traces
to https://smith.langchain.com. LangGraph node runs and LangChain LLM
calls are captured automatically; CrewAI and external-agent calls use
@traceable so they nest under the same trace.
"""
from __future__ import annotations

from typing import Any

from app import config


def is_tracing_enabled() -> bool:
    return config.LANGSMITH_TRACING_ENABLED


def build_graph_run_config(
    *,
    session_id: str,
    user_role: str,
    turn_id: str,
    message: str,
) -> dict[str, Any]:
    """RunnableConfig for graph.ainvoke with LangSmith tags and metadata."""
    return {
        "configurable": {"thread_id": session_id},
        "run_name": f"chat/{user_role}",
        "tags": ["capstone", "chat", user_role],
        "metadata": {
            "session_id": session_id,
            "user_role": user_role,
            "turn_id": turn_id,
            "message_preview": message[:200],
            "project": config.LANGSMITH_PROJECT,
        },
    }
