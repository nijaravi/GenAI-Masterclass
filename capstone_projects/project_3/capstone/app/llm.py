"""
One place to get a chat model from. Every node calls get_chat_model() and
gets back something with an .invoke(prompt: str) -> str interface.

DEMO_MODE (or no OPENAI_API_KEY) returns a DemoChatModel: a small,
deterministic, rule-based stand-in. It exists so the whole graph - routing,
RAG synthesis, the CrewAI crew - runs end-to-end with zero API cost and zero
network calls, which matters for a repeatable interview demo. Swapping in a
real model is a one-line change (set DEMO_MODE=false and OPENAI_API_KEY) and
nothing else in the codebase has to change, because every node talks to this
interface, not to OpenAI directly.
"""
from __future__ import annotations

from app import config


class DemoChatModel:
    """Deterministic stand-in for a real LLM. Not a toy for its own sake -
    it mirrors what the real prompts ask for, just without a live model
    generating the text. Each node's prompt embeds enough structure
    (keywords, role, JSON hints) that simple pattern matching reproduces
    the same decisions a real model would make on these worked examples."""

    name = "demo-rule-based-model"

    def invoke(self, prompt: str) -> str:
        p = prompt.lower()

        # --- Planner-style prompts: classify intent -------------------
        if "classify the user's intent" in p:
            if any(k in p for k in ["build status", "billing", "deployment", "invoice", "return", "order status", "logistics"]):
                if "logistics" in p or "vendor" in p or "carrier" in p:
                    return "account_status_external"
                return "account_status_internal"
            if any(k in p for k in ["pr #", "pull request", "jira", "coverage", "vulnerability", "code review"]):
                return "code_task"
            if any(k in p for k in ["api", "documentation", "how do i call", "reference", "endpoint"]):
                return "doc_lookup"
            return "product_info_lookup"

        # --- RAG synthesis prompts --------------------------------------
        if "answer using only the retrieved context" in p:
            return "(demo synthesis) Based on the retrieved passages, here is a grounded answer citing the sources shown above."

        # --- Final answer composer --------------------------------------
        if "compose the final answer" in p:
            return "Here is what I found:"

        return "(demo model) acknowledged."


class OpenAIChatModel:
    """Thin wrapper around langchain_openai.ChatOpenAI so nodes don't
    import langchain_openai directly."""

    def __init__(self, model: str, api_key: str):
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.2)
        self.name = model

    def invoke(self, prompt: str) -> str:
        result = self._llm.invoke(prompt)
        return result.content


_singleton = None


def get_chat_model():
    global _singleton
    if _singleton is not None:
        return _singleton

    if config.LIVE_LLM:
        _singleton = OpenAIChatModel(config.OPENAI_MODEL, config.OPENAI_API_KEY)
    else:
        _singleton = DemoChatModel()
    return _singleton
