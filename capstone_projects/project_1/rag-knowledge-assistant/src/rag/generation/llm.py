"""LLM client with cost-aware model routing.

Resume claim made concrete: route between hosted GPT-4o / GPT-4o-mini and
open-source Llama via Groq. The router picks a tier from a cheap heuristic
(query length / complexity); in a fuller build you'd add a small classifier.
Both OpenAI and Groq expose an OpenAI-compatible API, so one client class
handles both by swapping base_url + key.

Routing rule of thumb:
  * short, factual lookups        -> fast model (mini / Llama-8b on Groq)
  * long or multi-part questions  -> smart model (gpt-4o)
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import settings
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResult:
    text: str
    model: str


class LLMClient:
    def __init__(self) -> None:
        from openai import OpenAI

        self._openai = (
            OpenAI(api_key=settings.openai_api_key)
            if settings.openai_api_key
            else None
        )
        self._groq = (
            OpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            if settings.groq_api_key
            else None
        )

    def _route(self, query: str) -> tuple[str, str]:
        """Return (provider, model)."""
        complex_query = len(query) > 180 or query.count("?") > 1
        if complex_query:
            return "openai", settings.smart_model
        if settings.prefer_groq_for_fast and self._groq is not None:
            return "groq", settings.groq_model
        return "openai", settings.fast_model

    def generate(
        self, system: str, user: str, query_for_routing: str
    ) -> LLMResult:
        provider, model = self._route(query_for_routing)
        client = self._groq if provider == "groq" else self._openai
        if client is None:
            # Fall back to whichever provider is configured.
            client = self._openai or self._groq
            if client is None:
                raise RuntimeError("No LLM provider configured (set an API key).")

        logger.info("llm call", extra={"extra": {"provider": provider, "model": model}})
        resp = client.chat.completions.create(
            model=model,
            temperature=settings.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return LLMResult(text=resp.choices[0].message.content or "", model=model)
