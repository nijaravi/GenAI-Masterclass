"""Chunking.

The single biggest lever on RAG quality that nobody talks about in demos.
Chunks that are too big dilute the embedding (one vector trying to represent
five topics) and waste context budget; too small and you sever the context a
fact needs to be understood. We use a two-stage strategy:

  1. Structural split  — break on markdown headings so a chunk rarely straddles
     two unrelated sections.
  2. Token-window split — within an oversized section, slide a fixed-size token
     window with overlap so a sentence cut at a boundary still appears whole in
     a neighbouring chunk.

Token counting uses tiktoken (the tokenizer GPT models actually use), so
`chunk_tokens` means the same thing here as it does in the context budget.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import settings
from ..logging_config import get_logger
from ..models import Chunk, Document

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class _Section:
    heading: str
    text: str


def _encoder():
    """Lazy tiktoken loader with a graceful fallback for offline/CI use."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding(settings.tokenizer_encoding)
        return enc.encode, enc.decode
    except Exception:  # pragma: no cover - fallback path
        logger.warning("tiktoken unavailable; using whitespace approximation")
        return (
            lambda s: s.split(),
            lambda toks: " ".join(toks),
        )


def _split_into_sections(text: str) -> list[_Section]:
    """Split a markdown doc on headings, keeping each heading with its body."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [_Section(heading="", text=text.strip())]

    sections: list[_Section] = []
    # Preamble before the first heading, if any.
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(_Section(heading="", text=pre))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(2).strip()
        body = text[start:end].strip()
        sections.append(_Section(heading=heading, text=body))
    return sections


def _window_tokens(
    tokens: list, size: int, overlap: int
) -> list[tuple[int, int]]:
    """Yield (start, end) token spans with overlap. Stride must be positive."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    stride = max(1, size - overlap)
    spans = []
    i = 0
    n = len(tokens)
    while i < n:
        spans.append((i, min(i + size, n)))
        if i + size >= n:
            break
        i += stride
    return spans


def chunk_document(doc: Document) -> list[Chunk]:
    encode, decode = _encoder()
    chunks: list[Chunk] = []
    position = 0

    for section in _split_into_sections(doc.text):
        toks = encode(section.text)
        if len(toks) <= settings.chunk_tokens:
            windows = [(0, len(toks))]
        else:
            windows = _window_tokens(
                toks, settings.chunk_tokens, settings.chunk_overlap_tokens
            )

        for (s, e) in windows:
            piece = decode(toks[s:e]).strip()
            if not piece:
                continue
            # Prepend the section heading so the chunk is self-describing for
            # both the embedding model and the reader/citation.
            body = f"{section.heading}\n\n{piece}" if section.heading else piece
            chunk_id = f"{doc.doc_id}::{position}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source=doc.source,
                    category=doc.category,
                    text=body,
                    position=position,
                    metadata={**doc.metadata, "heading": section.heading},
                )
            )
            position += 1

    logger.info(
        "chunked document",
        extra={"extra": {"doc_id": doc.doc_id, "chunks": len(chunks)}},
    )
    return chunks
