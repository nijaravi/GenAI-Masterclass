# ============================================================
# app/router.py
# Model routing — pick the right model for each request.
# ============================================================

import logging

logger = logging.getLogger(__name__)

# ── Available models ──────────────────────────────────────────
MODELS = {
    "cheap":    "gpt-4o-mini",
    "capable":  "gpt-4o",
}


def route_model(
    task_type: str   = "support",
    input_len: int   = 0,
    p95_latency_ms: float = 0.0,
) -> str:
    """
    Return the most cost-effective model name for this request.

    Routing logic (in priority order):
      1. Simple pattern tasks → cheap model always
      2. Latency pressure (p95 > 1500ms) → cheap model to shed load
      3. Complex reasoning tasks → capable model
      4. Long context (> 4000 chars) → capable model
      5. Default → cheap model (try cheap first, escalate if quality fails)
    """
    model = MODELS["cheap"]   # default

    if task_type in {"classify", "extract", "translate", "sentiment"}:
        model = MODELS["cheap"]

    elif p95_latency_ms > 1500:
        model = MODELS["cheap"]
        logger.info(f"Router: latency pressure ({p95_latency_ms}ms) → forcing cheap model")

    elif task_type in {"reason", "code_critical", "plan"}:
        model = MODELS["capable"]

    elif input_len > 4000:
        model = MODELS["capable"]
        logger.info(f"Router: long context ({input_len} chars) → capable model")

    logger.debug(f"Router: task={task_type} input_len={input_len} → {model}")
    return model
