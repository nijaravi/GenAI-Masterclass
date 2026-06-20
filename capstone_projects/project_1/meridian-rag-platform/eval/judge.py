"""LLM-as-judge built as an LCEL chain with a structured output parser.

The judge scores an answer for relevance and faithfulness (1-5). Instead of
hand-parsing JSON, we use LangChain's PydanticOutputParser: it injects format
instructions into the prompt and parses the model's reply straight into a typed
object. Chain shape:  judge_prompt | model | PydanticOutputParser.

With no API key the model is the stub, which can't produce real JSON, so we
short-circuit to neutral scores and say so — real evaluation needs a key.
"""
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend.app.providers.llm import build_models
from common.logging_setup import get_logger

logger = get_logger(__name__)


class JudgeScore(BaseModel):
    relevance: int = Field(ge=1, le=5, description="does the answer address the question?")
    faithfulness: int = Field(ge=1, le=5, description="is every claim supported by the context?")
    reason: str = Field(description="one sentence explaining the scores")


JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a QUESTION, the CONTEXT the system "
    "retrieved, and the ANSWER it gave, score relevance and faithfulness from 1 "
    "to 5 (faithfulness: 5 = fully grounded in the context, 1 = made up).\n"
    "{format_instructions}"
)


class Judge:
    def __init__(self):
        models = build_models()
        name = "fast" if "fast" in models else "groq" if "groq" in models else "stub"
        self.is_stub = name == "stub"
        if self.is_stub:
            logger.warning("no LLM key — judge returns neutral stub scores")
            return

        parser = PydanticOutputParser(pydantic_object=JudgeScore)
        prompt = ChatPromptTemplate.from_messages([
            ("system", JUDGE_SYSTEM),
            ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"),
        ]).partial(format_instructions=parser.get_format_instructions())
        self.chain = prompt | models[name] | parser

    def score(self, question: str, context: str, answer: str) -> dict:
        if self.is_stub:
            return {"relevance": 3, "faithfulness": 3, "reason": "stub judge"}
        try:
            result = self.chain.invoke(
                {"question": question, "context": context, "answer": answer})
            return result.model_dump()
        except Exception as e:
            logger.warning("judge parse failed: %s", e)
            return {"relevance": 0, "faithfulness": 0, "reason": "unparseable"}
