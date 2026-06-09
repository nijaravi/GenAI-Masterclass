# ============================================================
# app/guardrails.py
# Input and output validation layer.
#
# Input guardrails  → run BEFORE the LLM sees the message
# Output guardrails → run AFTER the LLM responds, BEFORE return
# ============================================================

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Input patterns ────────────────────────────────────────────

INJECTION_PATTERNS = [
    (r"ignore.*previous.*instructions",                    "prompt_injection"),
    (r"ignore.*system.*prompt",                            "prompt_injection"),
    (r"you are now\b",                                     "persona_override"),
    (r"act as if you have no",                             "constraint_bypass"),
    (r"\bDAN\b",                                           "jailbreak_dan"),
    (r"pretend you are.{0,30}(evil|no rules|unfiltered)",  "jailbreak_roleplay"),
    (r"reveal.*system.*prompt",                            "data_exfiltration"),
    (r"print.*context.*window",                            "data_exfiltration"),
    (r"output.*system.*prompt",                            "data_exfiltration"),
    (r"\bjailbreak\b",                                     "jailbreak_explicit"),
]

# Patterns used for DETECTION (blocking PII from reaching the LLM)
PII_DETECT_PATTERNS = [
    (r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",          "credit_card"),
    (r"\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b",                          "ssn"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b",          "email"),
    (r"\+971[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{4}",               "uae_phone"),
    (r"784[\-\s]?\d{4}[\-\s]?\d{7}[\-\s]?\d",                     "emirates_id"),
    (r"\bAE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}\b",     "iban"),
]

# Patterns used for REDACTION (replacing PII with a placeholder)
PII_REDACT_PATTERNS = [
    (r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",  "[CARD_REDACTED]"),
    (r"\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b",                  "[SSN_REDACTED]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b",  "[EMAIL_REDACTED]"),
    (r"\+971[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{4}",        "[PHONE_REDACTED]"),
    (r"784[\-\s]?\d{4}[\-\s]?\d{7}[\-\s]?\d",             "[EMIRATESID_REDACTED]"),
]

REFUSAL_PATTERNS = [
    r"I cannot help with",
    r"I'm unable to",
    r"As an AI.{0,30}(can't|cannot|won't|not able)",
    r"I don't have access to",
]


# ── Data classes ──────────────────────────────────────────────

@dataclass
class InputGuardrailResult:
    passed: bool
    flags:  list  = field(default_factory=list)
    reason: Optional[str] = None

@dataclass
class OutputGuardrailResult:
    passed:  bool
    action:  str          # "send" | "retry" | "block"
    cleaned: str
    issues:  list = field(default_factory=list)


# ── Input guardrail ───────────────────────────────────────────

def check_input(text: str, max_length: int = 2000) -> InputGuardrailResult:
    """
    Validate incoming user message before it reaches the LLM.

    Checks:
      1. Length limit
      2. Prompt injection / jailbreak patterns
      3. PII presence (block to avoid logging sensitive data)
    """
    flags = []

    # 1. Length
    if len(text) > max_length:
        flags.append("input_too_long")

    # 2. Injection / jailbreak
    for pattern, label in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(label)
            break  # one injection flag is enough to block

    # 3. PII detection
    for pattern, label in PII_DETECT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(f"pii:{label}")
            # Don't break — collect all PII types present

    passed = len(flags) == 0
    reason = (
        None if passed
        else "Your message was flagged. Please rephrase or remove sensitive information."
    )
    return InputGuardrailResult(passed=passed, flags=flags, reason=reason)


# ── PII scrubber ──────────────────────────────────────────────

def scrub_pii(text: str) -> str:
    """
    Replace PII with redaction placeholders.
    Used on the cleaned message before it enters the RAG query
    and on the LLM response before it is returned to the user.
    """
    for pattern, replacement in PII_REDACT_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── Output guardrail ──────────────────────────────────────────

async def validate_output(
    response: str,
    context: Optional[dict] = None,
) -> OutputGuardrailResult:
    """
    Validate the LLM response before returning it to the user.

    Checks:
      1. PII in generated text (auto-redact, don't block)
      2. Response too short (likely a model error → retry)
      3. Model refusal on a valid question (log for review)
      4. JSON schema validity (when expects_json=True in context)
    """
    context = context or {}
    issues  = []
    cleaned = response

    # 1. PII redaction in output
    cleaned_pii = scrub_pii(cleaned)
    if cleaned_pii != cleaned:
        issues.append("pii_redacted")
        cleaned = cleaned_pii
        logger.warning("Output PII redacted before returning to user.")

    # 2. Too short
    if len(cleaned.strip()) < 15:
        return OutputGuardrailResult(
            passed=False, action="retry", cleaned=cleaned, issues=["too_short"]
        )

    # 3. Refusal on a presumably valid question
    for p in REFUSAL_PATTERNS:
        if re.search(p, cleaned, re.IGNORECASE):
            issues.append("refusal_detected")
            logger.info("Output refusal pattern detected — logged for review.")
            break

    # 4. JSON schema check
    if context.get("expects_json"):
        try:
            json.loads(cleaned)
        except json.JSONDecodeError:
            return OutputGuardrailResult(
                passed=False, action="retry_json", cleaned=cleaned, issues=["invalid_json"]
            )

    return OutputGuardrailResult(passed=True, action="send", cleaned=cleaned, issues=issues)
