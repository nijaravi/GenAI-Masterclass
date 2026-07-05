"""
The one place every node gets a model from. Every node calls get_model()
and gets back something with .invoke(prompt: str) -> str - Planner,
RAG synthesis, and the CrewAI crew all go through here (the crew's own
LLM is configured separately in app/crew/coder_crew.py using the same
config.OPENAI_MODEL, since CrewAI wants its own LLM object per agent).
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from app import config


class OpenAIChatModel:
    def __init__(self, model: str, api_key: str):
        self._llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.2)
        self.name = model

    def invoke(self, prompt: str) -> str:
        result = self._llm.invoke(prompt)
        return result.content


_model = OpenAIChatModel(config.OPENAI_MODEL, config.OPENAI_API_KEY)


def get_model() -> OpenAIChatModel:
    return _model
