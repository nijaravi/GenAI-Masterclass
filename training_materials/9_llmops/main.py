# ============================================================
# Section 9 — Chapter 3: Minimal FastAPI LLM Service
# GenAI Decoded by Nij
#
# Run:
#   pip install fastapi uvicorn openai pydantic python-dotenv
#   uvicorn main:app --reload
#
# Test:
#   curl -X POST http://localhost:8000/chat \
#     -H "Content-Type: application/json" \
#     -d '{"message": "How long is the return window?"}'
#
# Docs (auto-generated):
#   http://localhost:8000/docs
# ============================================================

import os
import time
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI

load_dotenv()

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="TechStore Support API",
    description="Minimal LLM-powered customer support service",
    version="1.0.0",
)

client = AsyncOpenAI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """TechStore support agent.
Scope: products, orders, returns, shipping.
Tone: professional, concise (max 3 sentences).
Uncertainty: say "I don't know" rather than guessing."""

# ── Request / Response schemas ────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    latency_ms: float
    tokens_used: int

# ── Endpoints ────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Liveness check — used by Docker, load balancers, Railway."""
    return {"status": "ok", "model": "gpt-4o-mini"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main support endpoint.
    Accepts a customer message, returns an LLM-generated response.
    """
    start = time.monotonic()

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": req.message},
            ],
            max_tokens=300,
            temperature=0.3,
        )
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        raise HTTPException(status_code=502, detail="LLM call failed. Please retry.")

    latency_ms = (time.monotonic() - start) * 1000
    response_text = completion.choices[0].message.content
    tokens_used   = completion.usage.total_tokens

    logger.info(
        f"session={req.session_id} tokens={tokens_used} "
        f"latency={latency_ms:.0f}ms"
    )

    return ChatResponse(
        response=response_text,
        latency_ms=round(latency_ms, 1),
        tokens_used=tokens_used,
    )


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
