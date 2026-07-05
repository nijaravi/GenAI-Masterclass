"""
Section 12: 'An evaluation harness scores a labeled set of representative
requests per audience for routing accuracy.' This is that harness, using
the four worked examples from Section 7 as the labeled set.

Run with:  python tests/test_routing.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph import get_graph
from app.rag.vector_store import seed_if_empty
from app.state import fresh_turn

# (user_role, message, expected specialist node hit)
LABELED_SET = [
    ("customer", "Does the TrailPro backpack come in a larger size?", "rag_agent"),
    ("client", "What's the build status on our latest deployment?", "mcp_tool"),
    ("developer", "Show me how to call the internal auth API, and check if PR #482 covers ticket JIRA-1190.", "coder_agent"),
    ("client", "Can you check the status of my return, order ord-7712, with the logistics partner?", "external_agent"),
]


async def _run_case(role: str, message: str, session_id: str):
    graph = get_graph()
    turn_input = fresh_turn(session_id, role, message)
    return await graph.ainvoke(turn_input, config={"configurable": {"thread_id": session_id}})


def test_routing_accuracy():
    seed_if_empty()
    correct = 0
    for i, (role, message, expected_node) in enumerate(LABELED_SET):
        result = asyncio.run(_run_case(role, message, f"eval-session-{i}"))
        node_path = result["node_path"]
        hit = expected_node in node_path
        correct += int(hit)
        assert result["final_answer"], f"no final answer for: {message}"
        print(f"[{'PASS' if hit else 'FAIL'}] role={role!r} expected={expected_node!r} path={node_path}")
    accuracy = correct / len(LABELED_SET)
    print(f"\nRouting accuracy: {accuracy:.0%} ({correct}/{len(LABELED_SET)})")
    assert accuracy >= 0.75


if __name__ == "__main__":
    test_routing_accuracy()
