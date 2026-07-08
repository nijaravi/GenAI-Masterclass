"""
Step-by-step trace logging for graph execution.

Each /chat request writes a turn block to logs/graph_trace.log with
every node's inputs, return payload, and handover target.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = Path(os.getenv("GRAPH_LOG_FILE", str(_LOG_DIR / "graph_trace.log")))

_LOGGER_NAME = "capstone.graph"
_logger: logging.Logger | None = None
_turn_id: contextvars.ContextVar[str] = contextvars.ContextVar("turn_id", default="unknown")

# Keys each node is expected to read (for concise "IN" snapshots).
_NODE_INPUT_KEYS: dict[str, tuple[str, ...]] = {
    "intake": ("session_id", "user_role", "user_query"),
    "planner": ("session_id", "user_role", "user_query", "node_path"),
    "orchestrator": (
        "session_id",
        "user_role",
        "user_query",
        "plan",
        "next_category",
        "hop_count",
        "agent_output",
        "node_path",
    ),
    "rag": ("session_id", "user_role", "user_query", "plan", "node_path"),
    "coder": ("session_id", "user_role", "user_query", "plan", "node_path"),
    "mcp": ("session_id", "user_role", "user_query", "plan", "session_id", "node_path"),
    "external_agent": ("session_id", "user_role", "user_query", "plan", "node_path"),
    "finalize": (
        "session_id",
        "user_role",
        "user_query",
        "agent_output",
        "collected_outputs",
        "node_path",
    ),
}

_HANDOVER_HINTS: dict[str, str] = {
    "intake": "planner",
    "planner": "orchestrator",
    "rag": "orchestrator",
    "coder": "orchestrator",
    "mcp": "orchestrator",
    "external_agent": "orchestrator",
    "finalize": "END",
}


def get_log_file() -> Path:
    return _LOG_FILE


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    _logger = logger
    return logger


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return str(value)


def _format_block(data: Any, indent: int = 2) -> str:
    serialized = _serialize(data)
    if isinstance(serialized, (dict, list)):
        return json.dumps(serialized, indent=indent, ensure_ascii=False)
    return str(serialized)


def snapshot_state(state: dict, node_name: str) -> dict:
    keys = _NODE_INPUT_KEYS.get(node_name, tuple(state.keys()))
    return {key: state.get(key) for key in keys if key in state}


def _handover_target(node_name: str, result: dict) -> str:
    if node_name == "orchestrator":
        return str(result.get("route", "final"))
    return _HANDOVER_HINTS.get(node_name, "unknown")


def log_turn_start(
    *,
    session_id: str,
    user_role: str,
    message: str,
    turn_id: str,
) -> None:
    _turn_id.set(turn_id)
    logger = _get_logger()
    ts = datetime.now(timezone.utc).isoformat()
    logger.info(
        "\n"
        f"{'=' * 80}\n"
        f"TURN START  turn_id={turn_id}  session={session_id}  role={user_role}  at={ts}\n"
        f"{'-' * 80}\n"
        f"REQUEST\n"
        f"{_format_block({'message': message})}\n"
        f"{'-' * 80}"
    )


def log_stage(
    *,
    node_name: str,
    session_id: str,
    state_in: dict,
    state_out: dict,
) -> None:
    logger = _get_logger()
    turn_id = _turn_id.get()
    handover = _handover_target(node_name, state_out)
    logger.info(
        f"\nSTAGE: {node_name}  turn_id={turn_id}  session={session_id}\n"
        f"  IN:\n"
        f"{_indent_block(_format_block(state_in), 4)}\n"
        f"  OUT (return):\n"
        f"{_indent_block(_format_block(state_out), 4)}\n"
        f"  HANDOVER -> {handover}\n"
        f"{'-' * 80}"
    )


def log_turn_end(
    *,
    turn_id: str,
    session_id: str,
    response: dict,
    node_path: list[str],
    hop_count: int,
) -> None:
    logger = _get_logger()
    ts = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"\nTURN END  turn_id={turn_id}  session={session_id}  at={ts}\n"
        f"  route_taken: {node_path}\n"
        f"  hop_count: {hop_count}\n"
        f"  RESPONSE\n"
        f"{_indent_block(_format_block(response), 4)}\n"
        f"{'=' * 80}\n"
    )


def log_turn_error(*, turn_id: str, session_id: str, error: str) -> None:
    logger = _get_logger()
    ts = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"\nTURN ERROR  turn_id={turn_id}  session={session_id}  at={ts}\n"
        f"  error: {error}\n"
        f"{'=' * 80}\n"
    )


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())
