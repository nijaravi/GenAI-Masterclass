"""Input guardrails: checked BEFORE the question reaches the LLM orchestration.

Two checks:
  * Prompt injection — phrases that try to override the system prompt
    ("ignore previous instructions", "reveal your system prompt"). We BLOCK
    these. (Real-world swap: a dedicated prompt-injection classifier / Llama
    Guard.)
  * PII in the question — emails, phone numbers, card/SSN-like strings. We
    REDACT them and let the question through with a note, so we don't store raw
    PII in logs/traces. (Real-world swap: Microsoft Presidio.)
"""
from common.logging_setup import get_logger
from . import policies
from .policies import GuardOutcome

logger = get_logger(__name__)


def check_input(question: str) -> GuardOutcome:
    notes: list[str] = []
    safe = question

    # 1) prompt injection -> block
    for pattern in policies.INJECTION_PATTERNS:
        if pattern.search(question):
            logger.warning("input blocked: possible prompt injection")
            return GuardOutcome(
                blocked=True,
                notes=["input: possible prompt-injection attempt"],
                safe_text=question,
            )

    # 2) PII -> redact, allow with a note
    for label, pattern in policies.PII_PATTERNS.items():
        if pattern.search(safe):
            safe = pattern.sub(f"[REDACTED_{label.upper()}]", safe)
            notes.append(f"input: redacted {label}")

    return GuardOutcome(blocked=False, notes=notes, safe_text=safe)
