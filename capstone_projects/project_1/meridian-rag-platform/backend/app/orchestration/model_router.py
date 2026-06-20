"""Model routing: choose which model name to use for a question.

Cheap heuristic — short/simple questions go to the fast (cheap) model, long or
multi-part ones to the smart model. Falls back through what's actually
configured, ending at the keyless stub so there's always a model to call. The
pipeline looks the chosen name up in the dict from build_models().
"""
from common.config import settings


def route(question: str, available: list[str]) -> str:
    is_complex = len(question) > 180 or question.count("?") > 1

    if is_complex and "smart" in available:
        return "smart"
    if settings.prefer_groq_for_fast and "groq" in available:
        return "groq"
    if "fast" in available:
        return "fast"
    if "groq" in available:
        return "groq"
    return "stub"
