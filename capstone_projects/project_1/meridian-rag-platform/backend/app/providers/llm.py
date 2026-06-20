"""Model provider — LangChain chat models.

`build_models()` returns the available chat models keyed by a simple name the
router understands:

  * "fast"  / "smart"  — ChatOpenAI (gpt-4o-mini / gpt-4o), if OPENAI_API_KEY set
  * "groq"             — ChatGroq (Llama 3.1 8B), if GROQ_API_KEY set
  * "stub"             — LangChain's FakeListChatModel, always present, so the
                         pipeline runs with no key

Every model is a LangChain Runnable, so it drops straight into an LCEL chain
(`prompt | model | parser`).
"""
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from common.config import settings
from common.logging_setup import get_logger

logger = get_logger(__name__)

_STUB_RESPONSE = (
    "[stub answer — no LLM key configured] I can only return a canned response "
    "without an API key, but retrieval, guardrails, cost, and monitoring all ran."
)


def build_models() -> dict:
    models: dict[str, object] = {
        "stub": FakeListChatModel(responses=[_STUB_RESPONSE]),
    }

    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        models["fast"] = ChatOpenAI(model=settings.fast_model,
                                    temperature=settings.temperature,
                                    api_key=settings.openai_api_key)
        models["smart"] = ChatOpenAI(model=settings.smart_model,
                                     temperature=settings.temperature,
                                     api_key=settings.openai_api_key)

    if settings.groq_api_key:
        from langchain_groq import ChatGroq
        models["groq"] = ChatGroq(model=settings.groq_model,
                                  temperature=settings.temperature,
                                  api_key=settings.groq_api_key)

    logger.info("models available: %s", list(models))
    return models
