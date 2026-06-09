# ============================================================
# app/llm.py
# LLM call wrapper and LLM-as-judge evaluation.
# ============================================================

import json
import time
import logging
from openai import AsyncOpenAI

from app.prompts import SUPPORT_SYSTEM, SUPPORT_USER_TEMPLATE, JUDGE_SYSTEM

logger  = logging.getLogger(__name__)
client  = AsyncOpenAI()


# ── Core LLM call ─────────────────────────────────────────────

async def call_llm(
    question: str,
    context:  str,
    model:    str = "gpt-4o-mini",
    max_tokens: int = 400,
) -> dict:
    """
    Call the LLM with RAG context injected into the user message.

    Args:
        question:   The user's original question (PII-scrubbed)
        context:    Retrieved document chunks formatted by rag.build_context()
        model:      Model name from router.route_model()
        max_tokens: Max response tokens

    Returns:
        {response, model, input_tokens, output_tokens, latency_ms}
    """
    user_message = SUPPORT_USER_TEMPLATE.format(
        context=context,
        question=question,
    )

    start      = time.monotonic()
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUPPORT_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=0.2,   # low temp for factual support responses
    )

    return {
        "response":      completion.choices[0].message.content,
        "model":         model,
        "input_tokens":  completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
        "latency_ms":    round((time.monotonic() - start) * 1000, 1),
    }


# ── LLM-as-Judge ─────────────────────────────────────────────

async def judge_response(question: str, response: str) -> dict:
    """
    Score a Q/A pair using a stronger model as judge.

    Uses gpt-4o as judge regardless of which model generated the response —
    you want the judge to be at least as capable as the model being judged.

    Returns a dict with keys: accuracy, completeness, tone, groundedness,
    overall, verdict ("PASS"|"REVIEW"|"FAIL"), reason.
    """
    try:
        result = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",
                 "content": f"Question: {question}\nResponse: {response}"},
            ],
            temperature=0,     # deterministic scoring
            max_tokens=200,
        )
        raw = result.choices[0].message.content.strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Judge returned non-JSON: {e}")
        return {"error": "judge_parse_failed", "raw": raw}
    except Exception as e:
        logger.error(f"Judge call failed: {e}")
        return {"error": str(e)}


async def background_judge(question: str, response: str, request_id: str) -> None:
    """
    Run judge asynchronously after the response has been returned to the user.
    Results are written to the monitoring log store.
    """
    from app.monitoring import log_event   # local import to avoid circular

    verdict = await judge_response(question, response)
    log_event("eval_verdict", request_id=request_id, **verdict)

    overall = verdict.get("overall", 10)
    if isinstance(overall, int) and overall < 6:
        log_event(
            "eval_alert",
            request_id=request_id,
            message=f"Low quality score: {overall}/10 — review flagged",
        )
        logger.warning(f"[{request_id}] Quality alert: judge score {overall}/10")
