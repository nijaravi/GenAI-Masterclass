"""Tests for hybrid retrieval — RRF math and BM25 lexical matching."""
from rag.ingestion.chunking import chunk_document
from rag.models import Chunk, Document, ScoredChunk
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.hybrid import reciprocal_rank_fusion


def _chunk(cid: str, text: str = "x") -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", title="t", source="s.md",
                 category="hr", text=text, position=0)


def test_rrf_rewards_agreement():
    # Doc 'b' is ranked highly by BOTH lists -> should win.
    list_a = [ScoredChunk(chunk=_chunk("a"), score=9),
              ScoredChunk(chunk=_chunk("b"), score=8)]
    list_b = [ScoredChunk(chunk=_chunk("b"), score=9),
              ScoredChunk(chunk=_chunk("c"), score=8)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=10)
    assert fused[0].chunk.chunk_id == "b"
    assert fused[0].retriever == "fused"


def test_rrf_dedupes():
    same = [ScoredChunk(chunk=_chunk("a"), score=1)]
    fused = reciprocal_rank_fusion([same, same], top_k=10)
    ids = [f.chunk.chunk_id for f in fused]
    assert ids.count("a") == 1


# A handful of docs so BM25 IDF is meaningful. (On a 2-doc corpus, terms in
# exactly half the docs get zero IDF — a degenerate edge, not real behaviour.)
_CORPUS = [
    Document(doc_id="1", title="VPN", source="it/vpn.md", category="it",
             text="Error GP-1107 means the device certificate expired. Renew it."),
    Document(doc_id="2", title="Leave", source="hr/leave.md", category="hr",
             text="Employees accrue 25 days of annual leave each year."),
    Document(doc_id="3", title="Pwd", source="it/pwd.md", category="it",
             text="Reset your password in the self service portal."),
    Document(doc_id="4", title="Sec", source="security/dc.md", category="security",
             text="Restricted data uses AES-256 encryption at rest."),
    Document(doc_id="5", title="Oncall", source="eng/oncall.md", category="engineering",
             text="Primary on-call must acknowledge a page within five minutes."),
    Document(doc_id="6", title="Exp", source="hr/exp.md", category="hr",
             text="Reset of expense claims requires director approval over 500 dollars."),
]


def test_bm25_finds_exact_token():
    chunks = [c for d in _CORPUS for c in chunk_document(d)]
    bm25 = BM25Retriever(chunks)
    res = bm25.search("GP-1107 certificate", top_k=3)
    assert res and res[0].chunk.source == "it/vpn.md"


def test_bm25_category_filter():
    chunks = [c for d in _CORPUS for c in chunk_document(d)]
    bm25 = BM25Retriever(chunks)
    res = bm25.search("reset", top_k=5, category="hr")
    assert res and all(r.chunk.category == "hr" for r in res)
