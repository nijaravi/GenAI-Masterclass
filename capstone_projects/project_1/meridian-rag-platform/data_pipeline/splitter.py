"""Step 2 of loading: split documents with LangChain splitters.

Two-stage, header-aware splitting — the approach the course settled on for
markdown:

  1. MarkdownHeaderTextSplitter — split on the markdown heading structure
     (#, ##, ###) so a chunk doesn't mix unrelated sections, and the headings
     are captured as metadata.
  2. RecursiveCharacterTextSplitter (token-based, via tiktoken) — further split
     any section that's still longer than the token budget, with overlap so a
     sentence cut at a boundary survives in the next chunk.

We prepend the captured headings back onto each chunk so it stays
self-describing for the embedding model and for citations.
"""
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from common.config import settings
from common.logging_setup import get_logger

logger = get_logger(__name__)

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _build_splitters():
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )
    # Token-accurate sizing using the same tokenizer GPT models use.
    size_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )
    return header_splitter, size_splitter


def split_documents(docs: list[Document]) -> list[Document]:
    header_splitter, size_splitter = _build_splitters()
    chunks: list[Document] = []

    for doc in docs:
        # 1) split on headings
        sections = header_splitter.split_text(doc.page_content)
        # carry the parent metadata onto each section
        for s in sections:
            s.metadata = {**doc.metadata, **s.metadata}
        # 2) enforce the token budget
        sized = size_splitter.split_documents(sections)

        for i, chunk in enumerate(sized):
            heading = chunk.metadata.get("h3") or chunk.metadata.get("h2") \
                or chunk.metadata.get("h1") or ""
            if heading and not chunk.page_content.startswith(heading):
                chunk.page_content = f"{heading}\n\n{chunk.page_content}"
            chunk.metadata["chunk_id"] = f"{doc.metadata['source']}::{len(chunks)}"
            chunk.metadata["position"] = len(chunks)
            chunks.append(chunk)

    logger.info("split %d documents into %d chunks", len(docs), len(chunks))
    return chunks
