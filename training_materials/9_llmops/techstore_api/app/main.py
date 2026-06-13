# ============================================================
# app/main.py
# FastAPI application — thin orchestration layer.
#
# This file only handles:
#   - App setup and startup lifecycle
#   - Route definitions and request/response schemas
#   - Wiring the modules together in the correct pipeline order
#
# Business logic lives in:
#   app/rag.py          → retrieval
#   app/guardrails.py   → input + output validation
#   app/llm.py          → LLM call + judge
#   app/router.py       → model routing
#   app/monitoring.py   → logging + metrics
#   app/prompts.py      → all prompt strings
# ============================================================

import random
import uuid
import time
import logging
from pathlib import Path

from dotenv import load_dotenv

# Must run before `from app import ...` — rag.py creates OpenAI() at import time.
# Key lives in app/.env (same folder as this file), not techstore_api/.env.
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app import rag, guardrails, llm, router, monitoring

monitoring.setup_logging()
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="TechStore Support API",
    description=(
        "Production RAG-powered customer support service. "
        "Pipeline: guardrails → RAG retrieval → LLM generation → "
        "output validation → structured logging → background eval."
    ),
    version="3.0.0",
)


# ── Startup: ingest knowledge base into ChromaDB ──────────────
@app.on_event("startup")
async def startup():
    logger.info("Starting up — ingesting knowledge base into ChromaDB...")
    added = rag.ingest()
    info  = rag.collection_info()
    logger.info(
        f"RAG ready: {info.get('document_count')} docs in collection "
        f"'{info.get('collection')}' (added {added} new this run)"
    )


# ── Schemas ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"
    task_type:  str = "support"   # passed to model router

class ChatResponse(BaseModel):
    request_id:    str
    response:      str
    model:         str
    latency_ms:    float
    cost_usd:      float
    chunks_used:   int            # how many RAG chunks contributed
    output_issues: list[str] = []

class JudgeRequest(BaseModel):
    question: str
    response: str


# ── Routes ────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """Liveness check — used by Docker HEALTHCHECK and Railway."""
    info = rag.collection_info()
    return {
        "status":          "ok",
        "rag_docs":        info.get("document_count", 0),
        "total_requests":  len(monitoring.get_logs("chat_success")),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main support endpoint — full RAG + LLMOps pipeline:

    1. Input guardrails   (injection, PII, length)
    2. PII scrub          (before RAG query)
    3. RAG retrieval      (ChromaDB → top-k chunks)
    4. Model routing      (task type + context length → model)
    5. LLM call           (system prompt + context + question)
    6. Output validation  (PII scrub, length, refusal detection)
    7. Structured log     (latency, tokens, cost, RAG stats)
    8. Background eval    (10% sampled LLM-as-judge, async)
    """
    request_id     = str(uuid.uuid4())[:8]
    pipeline_start = time.monotonic()

    # ── 1. Input guardrails ───────────────────────────────────
    guard = guardrails.check_input(req.message)
    if not guard.passed:
        monitoring.log_event(
            "input_blocked",
            request_id=request_id,
            session_id=req.session_id,
            flags=guard.flags,
        )
        raise HTTPException(
            status_code=400,
            detail={"blocked": True, "reason": guard.reason, "flags": guard.flags},
        )

    # ── 2. Scrub PII before it touches RAG or the LLM ─────────
    clean_message = guardrails.scrub_pii(req.message)

    # ── 3. RAG retrieval ──────────────────────────────────────
    chunks  = await rag.retrieve(clean_message)
    context = rag.build_context(chunks)

    monitoring.log_event(
        "rag_retrieve",
        request_id=request_id,
        query=clean_message[:80],
        chunks_returned=len(chunks),
        top_relevance=chunks[0]["relevance_score"] if chunks else 0,
        chunk_ids=[c["id"] for c in chunks],
    )

    # ── 4. Model routing ──────────────────────────────────────
    # Include context length so router can escalate on long RAG contexts
    total_input_len = len(clean_message) + len(context)
    model = router.route_model(
        task_type=req.task_type,
        input_len=total_input_len,
    )

    # ── 5. LLM call ───────────────────────────────────────────
    try:
        result = await llm.call_llm(
            question=clean_message,
            context=context,
            model=model,
        )
    except Exception as e:
        monitoring.log_event("llm_error", request_id=request_id, error=str(e))
        logger.error(f"[{request_id}] LLM call failed: {e}")
        raise HTTPException(status_code=502, detail="LLM call failed. Please retry.")

    response_text = result["response"]

    # ── 6. Output validation ──────────────────────────────────
    validated = await guardrails.validate_output(
        response_text,
        context={"session_id": req.session_id},
    )
    if not validated.passed:
        response_text = (
            "I wasn't able to generate a reliable answer. "
            "Please rephrase your question and try again."
        )
        validated.issues.append("output_blocked")
    else:
        response_text = validated.cleaned

    # ── 7. Structured log ─────────────────────────────────────
    total_latency_ms = round((time.monotonic() - pipeline_start) * 1000, 1)
    cost = monitoring.calc_cost(model, result["input_tokens"], result["output_tokens"])

    monitoring.log_event(
        "chat_success",
        request_id=request_id,
        session_id=req.session_id,
        model=model,
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        latency_ms=total_latency_ms,
        cost_usd=round(cost, 6),
        chunks_used=len(chunks),
        output_issues=validated.issues,
    )

    # ── 8. Background eval (10% sampling) ─────────────────────
    if random.random() < 0.10:
        background_tasks.add_task(
            llm.background_judge,
            req.message,
            response_text,
            request_id,
        )

    return ChatResponse(
        request_id=request_id,
        response=response_text,
        model=model,
        latency_ms=total_latency_ms,
        cost_usd=round(cost, 6),
        chunks_used=len(chunks),
        output_issues=validated.issues,
    )


@app.get("/metrics")
async def metrics():
    """Aggregated stats: requests, latency, cost, RAG, eval quality."""
    return monitoring.get_metrics()


@app.get("/rag/info")
async def rag_info():
    """ChromaDB collection stats — doc count, model, config."""
    return rag.collection_info()


@app.get("/rag/search")
async def rag_search(q: str, top_k: int = 3):
    """
    Retrieve chunks for a query applying the current MIN_RELEVANCE threshold.
    Empty results mean all chunks exceeded the threshold — use /rag/debug to see raw distances.
    """
    chunks = await rag.retrieve(q, top_k=top_k)
    return {
        "query":             q,
        "top_k":             top_k,
        "threshold":         rag.MIN_RELEVANCE,
        "chunks_returned":   len(chunks),
        "results":           chunks,
    }


@app.get("/rag/debug")
async def rag_debug(q: str, top_k: int = 10):
    """
    Raw retrieval — returns ALL top-k results with distances, NO threshold filtering.

    Use this to calibrate MIN_RELEVANCE:
      - Good matches should have distance ~0.3–0.6
      - Unrelated chunks typically score > 0.8
    Set RAG_MIN_RELEVANCE env var to a value that separates relevant from irrelevant.
    """
    raw = await rag.raw_retrieve(q, top_k=top_k)
    return {
        "query":            q,
        "current_threshold": rag.MIN_RELEVANCE,
        "tip": "Look at distances for relevant chunks. Set RAG_MIN_RELEVANCE to exclude noise.",
        "results":          raw,
    }


@app.post("/eval/judge")
async def run_judge(req: JudgeRequest):
    """
    Admin: run LLM-as-judge on any question/response pair.
    Use this to calibrate your judge prompt.
    """
    verdict = await llm.judge_response(req.question, req.response)
    return verdict
