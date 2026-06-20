"""User-facing endpoint: POST /v1/ask  (ask a question -> get an answer).

This handler is deliberately written so the whole request lifecycle reads top to
bottom — you can see each layer do its job in order:

    auth (dependency) -> validate -> rate limit -> INPUT guardrails
        -> orchestration pipeline -> OUTPUT guardrails -> monitoring trace
        -> response

Each step lives in its own module; this just calls them in sequence.
"""
import time
import uuid

from fastapi import APIRouter, Depends

from common.logging_setup import get_logger
from common.types import AskRequest, AskResponse
from common.users import User
from ..gateway.auth import require_user
from ..gateway.rate_limit import check_rate_limit
from ..gateway.validation import validate_ask
from ..guardrails.input_guards import check_input
from ..guardrails.output_guards import check_output
from ..monitoring.tracing import build_trace
from ..state import get_state

logger = get_logger(__name__)
router = APIRouter()


@router.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest, user: User = Depends(require_user)) -> AskResponse:
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    state = get_state()

    # --- gateway: validation + rate limiting ---
    validate_ask(req)
    check_rate_limit(user.user_id)

    # --- input guardrails (PII redaction, injection block) ---
    gin = check_input(req.question)
    if gin.blocked:
        return _finish(state, request_id, user, req.question,
                       answer="[blocked: input guardrail]", model="none",
                       cached=False, blocked=True, notes=gin.notes,
                       prompt_tokens=0, completion_tokens=0, context="",
                       citations=[], timings={}, start=start)

    question = gin.safe_text   # PII redacted

    # --- orchestration: the RAG pipeline ---
    result = state.pipeline.run(question, req.category, req.use_cache)

    # --- output guardrails (toxicity/PII block, groundedness flag) ---
    gout = check_output(result.answer, result.context_used)
    notes = gin.notes + gout.notes
    answer = "[blocked: output guardrail]" if gout.blocked else result.answer
    citations = [] if gout.blocked else result.citations

    return _finish(state, request_id, user, question, answer=answer,
                   model=result.model, cached=result.cached, blocked=gout.blocked,
                   notes=notes, prompt_tokens=result.prompt_tokens,
                   completion_tokens=result.completion_tokens,
                   context=result.context_used, citations=citations,
                   timings=result.timings_ms, start=start)


def _finish(state, request_id, user, question, *, answer, model, cached, blocked,
            notes, prompt_tokens, completion_tokens, context, citations, timings,
            start) -> AskResponse:
    """Write the monitoring trace and build the response (shared by both paths)."""
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    trace = build_trace(
        request_id=request_id, user_id=user.user_id, question=question,
        answer=answer, model=model, cached=cached, blocked=blocked,
        guardrail_notes=notes, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, latency_ms=latency_ms,
        timings_ms=timings, context_used=context)
    state.trace_store.insert(trace)

    return AskResponse(
        answer=answer, citations=citations, model=model, cached=cached,
        request_id=request_id, blocked=blocked, guardrail_notes=notes,
        timings_ms=timings)
