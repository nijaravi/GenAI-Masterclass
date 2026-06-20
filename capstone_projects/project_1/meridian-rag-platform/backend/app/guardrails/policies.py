"""Guardrail policies: the shared patterns, thresholds, and the result type.

Both the input and output guards import from here, so the rules live in one
place. These are intentionally lightweight, dependency-free heuristics — enough
to demonstrate where each check goes in the pipeline. In production you'd swap
them for real services (named in each guard file), but the wiring stays the same.
"""
import re
from dataclasses import dataclass, field


@dataclass
class GuardOutcome:
    blocked: bool = False
    notes: list[str] = field(default_factory=list)
    safe_text: str = ""     # input text after redaction (output guards ignore this)


# ---- PII patterns (used on both input and output) ----
PII_PATTERNS = {
    "email": re.compile(r"[\w.\-]+@[\w.\-]+\.\w+"),
    "phone": re.compile(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

# ---- Prompt-injection phrases (case-insensitive) ----
INJECTION_PATTERNS = [
    re.compile(r"ignore (all |the )?(previous|prior|above) (instructions|prompts)", re.I),
    re.compile(r"disregard (the |your )?(system|previous) (prompt|instructions)", re.I),
    re.compile(r"reveal (your |the )?(system prompt|instructions)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"pretend (to be|you are)", re.I),
]

# ---- Toxicity wordlist (kept tiny and obvious for the demo) ----
TOXIC_WORDS = {"idiot", "stupid", "hate", "kill", "moron"}

# ---- Thresholds (groundedness comes from config) ----
from common.config import settings  # noqa: E402

GROUNDEDNESS_MIN_OVERLAP = settings.groundedness_min_overlap

# The exact phrase the model is told to use when it can't answer — a low
# groundedness score on this is expected, not a hallucination.
NO_ANSWER_PHRASE = "i don't have that information"
