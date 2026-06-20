"""A LangChain callback handler that captures token usage per request.

This is how the monitoring layer hooks into the LLM call without the pipeline
having to know about tokens: we pass this handler in the chain's `config`
callbacks, and LangChain calls `on_llm_end` with the model's response, from which
we read the token usage. If the model doesn't report usage (e.g. the keyless
stub), we estimate from text length so the cost dashboard isn't all zeros.

`llm_runs == 0` after an invoke means the LLM was never actually called — i.e.
the response came from the cache.
"""
from langchain_core.callbacks import BaseCallbackHandler

from common.logging_setup import get_logger

logger = get_logger(__name__)


class CostCallback(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_runs = 0
        self._prompt_chars = 0

    # chat models call this one; plain LLMs call on_llm_start
    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._prompt_chars = sum(
            len(getattr(m, "content", "")) for chat in messages for m in chat)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._prompt_chars = sum(len(p) for p in prompts)

    def on_llm_end(self, response, **kwargs):
        self.llm_runs += 1
        usage = (response.llm_output or {}).get("token_usage") \
            or (response.llm_output or {}).get("usage") or {}

        prompt_t = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_t = usage.get("completion_tokens") or usage.get("output_tokens") or 0

        # Fallback: estimate ~4 chars/token when usage isn't reported.
        if not prompt_t and not completion_t:
            gen_text = ""
            try:
                gen_text = response.generations[0][0].text
            except Exception:
                pass
            prompt_t = self._prompt_chars // 4
            completion_t = len(gen_text) // 4

        self.prompt_tokens += prompt_t
        self.completion_tokens += completion_t
