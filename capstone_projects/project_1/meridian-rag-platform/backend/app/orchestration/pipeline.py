"""The RAG orchestration pipeline, built on LCEL.

Per request:
    build the composed retriever (dense + sparse -> ensemble -> rerank)
      -> retrieve documents
      -> format them into a numbered context block
      -> route to a model
      -> run the LCEL chain:  ANSWER_PROMPT | model | StrOutputParser()
      -> return answer + citations + token usage + timings

Token usage is captured by a LangChain callback (CostCallback) passed in the
chain's config. LLM response caching is configured globally in state.py
(set_llm_cache); if the callback sees zero LLM runs, the answer came from cache.
"""
import time
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser

from common.config import settings
from common.logging_setup import get_logger
from common.types import Citation
from ..monitoring.cost_callback import CostCallback
from ..retrieval.retrievers import build_retriever
from .model_router import route
from .prompts import ANSWER_PROMPT, build_citations, format_docs

logger = get_logger(__name__)

# Map the router's choice to a concrete model id (used for the response + pricing).
_MODEL_ID = {
    "fast": settings.fast_model,
    "smart": settings.smart_model,
    "groq": settings.groq_model,
    "stub": "stub-model",
}


@dataclass
class PipelineResult:
    answer: str
    citations: list[Citation]
    model: str
    cached: bool
    context_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timings_ms: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, vectorstore, all_docs, models):
        self.vectorstore = vectorstore
        self.all_docs = all_docs
        self.models = models

    def run(self, question: str, category: str | None = None,
            use_cache: bool = True) -> PipelineResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        # --- retrieve (dense + sparse -> ensemble -> rerank) ---
        retriever = build_retriever(self.vectorstore, self.all_docs, category)
        docs = retriever.invoke(question)
        docs = docs[: settings.rerank_top_n]
        timings["retrieve_ms"] = _ms(t0)

        context = format_docs(docs)

        # --- route + generate via an LCEL chain ---
        t1 = time.perf_counter()
        model_name = route(question, list(self.models))
        model = self.models[model_name]
        chain = ANSWER_PROMPT | model | StrOutputParser()

        cost_cb = CostCallback()
        answer = chain.invoke(
            {"context": context, "question": question},
            config={"callbacks": [cost_cb]},
        )
        timings["generate_ms"] = _ms(t1)
        timings["total_ms"] = _ms(t0)

        cached = cost_cb.llm_runs == 0   # cache short-circuited the LLM call
        return PipelineResult(
            answer=answer,
            citations=build_citations(docs),
            model=_MODEL_ID.get(model_name, model_name),
            cached=cached,
            context_used=context,
            prompt_tokens=cost_cb.prompt_tokens,
            completion_tokens=cost_cb.completion_tokens,
            timings_ms=timings,
        )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
