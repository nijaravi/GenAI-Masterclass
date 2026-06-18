"""Prompt construction.

The system prompt is the cheapest, highest-leverage guardrail in RAG. Three
rules do most of the work against hallucination:
  1. Answer ONLY from the provided context.
  2. If the context doesn't contain the answer, say so — don't guess.
  3. Cite the chunk ids you used.

We also number the context chunks so the model can cite [1], [2] and we can map
those back to real sources for the API response.
"""
from __future__ import annotations

from ..models import ScoredChunk

SYSTEM_PROMPT = (
    "You are Meridian's internal knowledge assistant. Answer employee questions "
    "using ONLY the numbered context passages provided.\n"
    "Rules:\n"
    "- If the answer is not contained in the context, reply exactly: "
    '"I don\'t have that information in the available documents." Do not guess.\n'
    "- Be concise and specific. Prefer the exact policy/figure over paraphrase.\n"
    "- Cite the passages you used with their bracketed numbers, e.g. [2].\n"
    "- Never invent policy numbers, dates, or figures."
)


def build_context_block(chunks: list[ScoredChunk], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for i, sc in enumerate(chunks, start=1):
        block = f"[{i}] (source: {sc.chunk.source}; title: {sc.chunk.title})\n{sc.chunk.text}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_user_prompt(query: str, context: str) -> str:
    return (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above and cite the passage numbers you relied on."
    )
