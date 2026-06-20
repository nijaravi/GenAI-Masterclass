"""Prompt assembly with LangChain's ChatPromptTemplate, plus helpers to format
retrieved documents into a numbered context block and into citations.

The system message carries the anti-hallucination rules (answer only from
context, say so if it's missing, cite passage numbers). The context block numbers
each passage so the model can cite [1], [2], and we can map those back to sources.
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from common.config import settings
from common.types import Citation

SYSTEM = (
    "You are Meridian's internal knowledge assistant. Answer employee questions "
    "using ONLY the numbered context passages provided.\n"
    "Rules:\n"
    "- If the answer is not in the context, reply exactly: "
    '"I don\'t have that information in the available documents." Do not guess.\n'
    "- Be concise and specific. Prefer the exact policy or figure over a paraphrase.\n"
    "- Cite the passages you used with their bracketed numbers, e.g. [2].\n"
    "- Never invent policy numbers, dates, or figures."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Context passages:\n\n{context}\n\nQuestion: {question}\n\n"
              "Answer using only the context above and cite the passage numbers you used."),
])


def format_docs(docs: list[Document]) -> str:
    parts, used = [], 0
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "?")
        title = d.metadata.get("title", "")
        block = f"[{i}] (source: {src}; title: {title})\n{d.page_content}"
        if used + len(block) > settings.max_context_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_citations(docs: list[Document]) -> list[Citation]:
    return [
        Citation(
            title=d.metadata.get("title", ""),
            source=d.metadata.get("source", ""),
            snippet=d.page_content[:200],
        )
        for d in docs
    ]
