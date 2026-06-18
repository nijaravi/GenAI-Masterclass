"""Tests for chunking — the highest-leverage and most testable pure logic."""
from rag.config import settings
from rag.ingestion.chunking import _window_tokens, chunk_document
from rag.models import Document


def test_window_tokens_has_overlap():
    spans = _window_tokens(list(range(100)), size=40, overlap=10)
    # stride = 30, so windows start at 0, 30, 60, 90
    assert spans[0] == (0, 40)
    assert spans[1][0] == 30  # overlap of 10 with previous window's tail
    assert spans[-1][1] == 100  # last window reaches the end


def test_window_tokens_short_input_single_window():
    spans = _window_tokens(list(range(10)), size=40, overlap=10)
    assert spans == [(0, 10)]


def test_chunk_ids_are_stable_and_ordered():
    doc = Document(
        doc_id="abc", title="T", source="s.md", category="hr",
        text="# A\n\npara one\n\n## B\n\npara two",
    )
    chunks = chunk_document(doc)
    assert [c.chunk_id for c in chunks] == [f"abc::{i}" for i in range(len(chunks))]
    # heading is prepended into the chunk body so it's self-describing
    assert any("A" in c.text for c in chunks)


def test_oversized_section_is_split():
    big = "# H\n\n" + " ".join(f"word{i}" for i in range(4000))
    doc = Document(doc_id="d", title="T", source="s.md", category="it", text=big)
    chunks = chunk_document(doc)
    assert len(chunks) > 1  # must split beyond chunk_tokens
