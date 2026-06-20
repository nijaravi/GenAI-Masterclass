"""Tracing helper: assemble a Trace row from the pieces a request produces.

Small on purpose — it gathers the fields and computes cost so the request
handler doesn't build the Trace inline. The handler calls `build_trace(...)`
then `store.insert(trace)`.
"""
from common.types import Trace
from .cost import compute_cost


def build_trace(*, request_id: str, user_id: str, question: str, answer: str,
                model: str, cached: bool, blocked: bool,
                guardrail_notes: list[str], prompt_tokens: int,
                completion_tokens: int, latency_ms: float,
                timings_ms: dict, context_used: str) -> Trace:
    cost = compute_cost(model, prompt_tokens, completion_tokens)
    return Trace(
        request_id=request_id, user_id=user_id, question=question, answer=answer,
        model=model, cached=cached, blocked=blocked, guardrail_notes=guardrail_notes,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        cost_usd=cost, latency_ms=latency_ms, timings_ms=timings_ms,
        context_used=context_used)
