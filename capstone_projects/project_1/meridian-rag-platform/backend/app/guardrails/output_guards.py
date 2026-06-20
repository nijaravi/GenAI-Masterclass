"""Output guardrails: checked AFTER the LLM answers, before we return it.

Three checks:
  * Groundedness (hallucination) — what fraction of the answer's words actually
    appear in the retrieved context? Below a threshold, the model is likely
    making things up, so we FLAG it (note), but don't block — the "I don't have
    that information" reply is allowed to score low. (Real-world swap: an
    NLI/groundedness model or an LLM-as-judge check.)
  * Toxicity — a tiny wordlist; BLOCK if hit. (Real-world swap: a real toxicity
    classifier such as Detoxify.)
  * PII leak — the same PII patterns run on the answer; BLOCK if the model is
    about to hand back an email/card/SSN. (Real-world swap: Presidio.)
"""
import re

from common.logging_setup import get_logger
from . import policies
from .policies import GuardOutcome

logger = get_logger(__name__)

_WORD = re.compile(r"[A-Za-z0-9]+")


def _word_overlap(answer: str, context: str) -> float:
    answer_words = {w.lower() for w in _WORD.findall(answer)}
    if not answer_words:
        return 1.0
    context_words = {w.lower() for w in _WORD.findall(context)}
    overlap = answer_words & context_words
    return len(overlap) / len(answer_words)


def check_output(answer: str, context: str) -> GuardOutcome:
    notes: list[str] = []

    # 1) toxicity -> block
    lower = answer.lower()
    if any(word in lower for word in policies.TOXIC_WORDS):
        logger.warning("output blocked: toxic content")
        return GuardOutcome(blocked=True, notes=["output: toxic content detected"])

    # 2) PII leak -> block
    for label, pattern in policies.PII_PATTERNS.items():
        if pattern.search(answer):
            logger.warning("output blocked: PII leak (%s)", label)
            return GuardOutcome(blocked=True, notes=[f"output: PII leak ({label})"])

    # 3) groundedness -> flag only (do not block)
    if policies.NO_ANSWER_PHRASE not in lower:
        overlap = _word_overlap(answer, context)
        if overlap < policies.GROUNDEDNESS_MIN_OVERLAP:
            notes.append(f"output: low groundedness ({overlap:.2f}) — possible hallucination")

    return GuardOutcome(blocked=False, notes=notes)
