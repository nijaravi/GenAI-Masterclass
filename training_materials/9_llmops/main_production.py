# ============================================================
# Section 9 — Chapter 15 Capstone: Production FastAPI Service
# GenAI Decoded by Nij
#
# Full stack: guardrails → routing → LLM → output validation
#             → structured logging → background eval sampling
#
# Run:
#   pip install fastapi uvicorn openai pydantic python-dotenv
#   uvicorn main_production:app --reload
#
# Test endpoints:
#   POST /chat          — main support endpoint
#   GET  /health        — liveness check
#   GET  /metrics       — aggregated stats
#   POST /eval/judge    — run LLM-as-judge on any Q/A pair
# ============================================================

import os
import re
import json
import time
import uuid
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

load_dotenv()

# ── App & client ─────────────────────────────────────────────
app = FastAPI(
    title="TechStore Support API — Production",
    description="Full production LLM service with guardrails, monitoring, and eval",
    version="2.0.0",
)

client = AsyncOpenAI()

# ── Structured logger ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)

# In-memory store for this demo — swap for a real DB / log aggregator in prod
_log_store: list[dict] = []

def log_event(event: str, **kwargs):
    entry = {"ts": round(time.time(), 3), "event": event, **kwargs}
    _log_store.append(entry)
    _logger.info(json.dumps(entry))


# ── Pricing (per 1M tokens) — update when rates change ───────
PRICING = {
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
    "gpt-4o":      {"input": 5.00,  "output": 15.00},
}

def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens / 1e6 * p["input"]) + (output_tokens / 1e6 * p["output"])


# ── Prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """TechStore support agent.
Scope: products, orders, returns, shipping.
Tone: professional, concise (max 3 sentences).
Uncertainty: say "I don't know" rather than guessing."""

JUDGE_SYSTEM = """Quality evaluator for a customer support AI.
Score the response (1-10 each): accuracy, completeness, tone, groundedness.
Respond ONLY with this JSON (no markdown):
{"accuracy":<1-10>,"completeness":<1-10>,"tone":<1-10>,"groundedness":<1-10>,
 "overall":<1-10>,"verdict":"PASS"|"REVIEW"|"FAIL","reason":"<one sentence>"}
Verdict: PASS if overall>=8 and accuracy>=7; FAIL if accuracy<=4 or groundedness<=4; else REVIEW."""


# ============================================================
# GUARDRAILS
# ============================================================

@dataclass
class GuardrailResult:
    passed: bool
    flags: list = field(default_factory=list)
    reason: Optional[str] = None

INJECTION_PATTERNS = [
    (r"ignore.*previous.*instructions",              "prompt_injection"),
    (r"ignore.*system.*prompt",                      "prompt_injection"),
    (r"you are now\b",                               "persona_override"),
    (r"act as if you have no",                       "constraint_bypass"),
    (r"\bDAN\b",                                     "jailbreak_dan"),
    (r"pretend you are.{0,30}(evil|no rules|unfiltered)", "jailbreak_roleplay"),
    (r"reveal.*system.*prompt",                      "data_exfiltration"),
    (r"print.*context.*window",                      "data_exfiltration"),
    (r"\bjailbreak\b",                               "jailbreak_explicit"),
]

PII_PATTERNS = [
    (r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",          "credit_card"),
    (r"\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b",                          "ssn"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b",          "email"),
    (r"\+971[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{4}",               "uae_phone"),
    (r"784[\-\s]?\d{4}[\-\s]?\d{7}[\-\s]?\d",                     "emirates_id"),
    (r"\bAE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}\b",     "iban"),
]

PII_REDACTIONS = [
    (r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",  "[CARD_REDACTED]"),
    (r"\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b",                  "[SSN_REDACTED]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b",  "[EMAIL_REDACTED]"),
]

REFUSAL_PATTERNS = [
    r"I cannot help with",
    r"I'm unable to",
    r"As an AI.{0,30}(can't|cannot|won't|not able)",
    r"I don't have access to",
]


def check_input(text: str, max_length: int = 2000) -> GuardrailResult:
    """Input guardrail — runs before the LLM call."""
    flags = []

    if len(text) > max_length:
        flags.append(f"input_too_long")

    for pattern, label in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(label)
            break

    for pattern, label in PII_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(f"pii:{label}")

    passed = len(flags) == 0
    reason = None if passed else "Your message was flagged. Please rephrase or remove sensitive information."
    return GuardrailResult(passed=passed, flags=flags, reason=reason)


def scrub_pii(text: str) -> str:
    """Redact PII from text (used on both input before RAG and output before return)."""
    for pattern, replacement in PII_REDACTIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


async def validate_output(response: str, context: dict = {}) -> dict:
    """Output guardrail — runs after the LLM call, before returning to user."""
    issues = []
    cleaned = response

    # 1. Redact PII that may have leaked through
    cleaned_pii = scrub_pii(cleaned)
    if cleaned_pii != cleaned:
        issues.append("pii_redacted")
        cleaned = cleaned_pii

    # 2. Empty / too-short response
    if len(cleaned.strip()) < 15:
        return {"passed": False, "action": "retry", "cleaned": cleaned, "issues": ["too_short"]}

    # 3. Refusal on a valid question
    for p in REFUSAL_PATTERNS:
        if re.search(p, cleaned, re.IGNORECASE):
            issues.append("refusal_detected")
            break

    # 4. JSON schema check (when caller sets expects_json=True in context)
    if context.get("expects_json"):
        try:
            json.loads(cleaned)
        except json.JSONDecodeError:
            return {"passed": False, "action": "retry_json", "cleaned": cleaned, "issues": ["invalid_json"]}

    return {"passed": True, "action": "send", "cleaned": cleaned, "issues": issues}


# ============================================================
# MODEL ROUTING
# ============================================================

def route_model(task_type: str = "support", input_len: int = 0,
                p95_latency_ms: float = 0.0) -> str:
    """Route each request to the most cost-effective model that fits the task."""
    if task_type in ["classify", "extract", "translate", "sentiment"]:
        return "gpt-4o-mini"
    if p95_latency_ms > 1500:          # under latency pressure → cheaper/faster
        return "gpt-4o-mini"
    if task_type in ["reason", "code_critical", "plan"]:
        return "gpt-4o"
    if input_len > 4000:               # long context → capable model
        return "gpt-4o"
    return "gpt-4o-mini"               # default


# ============================================================
# LLM CALL
# ============================================================

async def call_llm(message: str, model: str = "gpt-4o-mini",
                   system: str = SYSTEM_PROMPT, max_tokens: int = 300) -> dict:
    start = time.monotonic()
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": message},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return {
        "response":      completion.choices[0].message.content,
        "model":         model,
        "input_tokens":  completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
        "latency_ms":    round((time.monotonic() - start) * 1000, 1),
    }


# ============================================================
# LLM-AS-JUDGE (background eval)
# ============================================================

async def judge_response(question: str, response: str) -> dict:
    """Score a Q/A pair using a stronger model as judge. Runs asynchronously."""
    try:
        result = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",
                 "content": f"Question: {question}\nResponse: {response}"},
            ],
            temperature=0,
            max_tokens=200,
        )
        return json.loads(result.choices[0].message.content.strip())
    except Exception as e:
        return {"error": str(e)}


async def _background_judge(question: str, response: str, request_id: str):
    """Background task — called after response is already returned to user."""
    verdict = await judge_response(question, response)
    log_event("eval_verdict", request_id=request_id, **verdict)
    if verdict.get("overall", 10) < 6:
        log_event("eval_alert", request_id=request_id,
                  message=f"Low quality score: {verdict.get('overall')}/10")


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    task_type: str = "support"     # passed to model router

class ChatResponse(BaseModel):
    request_id: str
    response: str
    model: str
    latency_ms: float
    cost_usd: float
    output_issues: list[str] = []

class JudgeRequest(BaseModel):
    question: str
    response: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health():
    """Liveness check. Used by Docker healthcheck and Railway."""
    return {
        "status": "ok",
        "model": "gpt-4o-mini",
        "total_requests": len([e for e in _log_store if e["event"] == "chat_success"]),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main production endpoint — full pipeline:
    input guard → model route → LLM → output validate → log → background eval
    """
    request_id = str(uuid.uuid4())[:8]
    pipeline_start = time.monotonic()

    # ── Step 1: Input guardrails ─────────────────────────────
    guard = check_input(req.message)
    if not guard.passed:
        log_event("input_blocked",
                  request_id=request_id,
                  session_id=req.session_id,
                  flags=guard.flags)
        raise HTTPException(
            status_code=400,
            detail={"blocked": True, "reason": guard.reason, "flags": guard.flags},
        )

    # ── Step 2: Scrub PII before it touches the LLM ──────────
    clean_message = scrub_pii(req.message)

    # ── Step 3: Model routing ────────────────────────────────
    model = route_model(task_type=req.task_type, input_len=len(clean_message))

    # ── Step 4: LLM call ─────────────────────────────────────
    try:
        result = await call_llm(clean_message, model=model)
    except Exception as e:
        log_event("llm_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=502, detail="LLM call failed. Please retry.")

    response_text = result["response"]

    # ── Step 5: Output validation ─────────────────────────────
    validated = await validate_output(response_text, {"session_id": req.session_id})
    if not validated["passed"]:
        response_text = (
            "I wasn't able to generate a reliable answer. "
            "Please rephrase your question and try again."
        )
        validated["issues"].append("output_blocked")
    else:
        response_text = validated["cleaned"]

    # ── Step 6: Structured log ────────────────────────────────
    total_latency_ms = round((time.monotonic() - pipeline_start) * 1000, 1)
    cost = calc_cost(model, result["input_tokens"], result["output_tokens"])

    log_event("chat_success",
              request_id=request_id,
              session_id=req.session_id,
              model=model,
              input_tokens=result["input_tokens"],
              output_tokens=result["output_tokens"],
              latency_ms=total_latency_ms,
              cost_usd=round(cost, 6),
              output_issues=validated["issues"])

    # ── Step 7: Background eval (10% sampling) ────────────────
    if __import__("random").random() < 0.10:
        background_tasks.add_task(
            _background_judge, req.message, response_text, request_id
        )

    return ChatResponse(
        request_id=request_id,
        response=response_text,
        model=model,
        latency_ms=total_latency_ms,
        cost_usd=round(cost, 6),
        output_issues=validated["issues"],
    )


@app.get("/metrics")
async def metrics():
    """Aggregated stats for the last N requests."""
    chat_logs    = [e for e in _log_store if e["event"] == "chat_success"]
    blocked_logs = [e for e in _log_store if e["event"] == "input_blocked"]
    eval_logs    = [e for e in _log_store if e["event"] == "eval_verdict"]

    if not chat_logs:
        return {"message": "No requests yet."}

    latencies   = [e["latency_ms"]  for e in chat_logs]
    costs       = [e["cost_usd"]    for e in chat_logs]
    latencies_s = sorted(latencies)

    pass_count   = sum(1 for e in eval_logs if e.get("verdict") == "PASS")
    eval_total   = len(eval_logs)

    return {
        "total_requests":   len(chat_logs),
        "blocked_requests": len(blocked_logs),
        "latency": {
            "p50_ms":  latencies_s[len(latencies_s) // 2],
            "p95_ms":  latencies_s[min(len(latencies_s)-1, int(len(latencies_s)*0.95))],
            "avg_ms":  round(sum(latencies) / len(latencies), 1),
        },
        "cost": {
            "total_usd":        round(sum(costs), 6),
            "avg_per_call_usd": round(sum(costs) / len(costs), 6),
            "proj_10k_calls":   round(sum(costs) / len(costs) * 10_000, 2),
        },
        "eval": {
            "judged":    eval_total,
            "pass_rate": round(pass_count / eval_total, 3) if eval_total else None,
        },
        "model_usage": {
            model: sum(1 for e in chat_logs if e.get("model") == model)
            for model in set(e.get("model") for e in chat_logs)
        },
    }


@app.post("/eval/judge")
async def run_judge(req: JudgeRequest):
    """
    Admin endpoint: run LLM-as-judge on any question/response pair.
    Use this to test your judge prompt and calibrate scoring.
    """
    verdict = await judge_response(req.question, req.response)
    return verdict


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_production:app", host="0.0.0.0", port=8000, reload=True)
