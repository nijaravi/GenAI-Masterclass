# Meridian RAG Knowledge Assistant

A production-shaped **retrieval-augmented generation** service over an internal
document set. Ask a plain-English question, get a grounded answer with citations
back to the source documents.

This is a **portfolio / learning project** — a fictional company ("Meridian")
with synthetic internal docs (HR, IT, security, engineering). It exists to
demonstrate, end to end, the techniques a GenAI engineer is expected to know:
chunking, embeddings, a vector store (ChromaDB **or** pgvector), **hybrid search
(BM25 + dense) fused with Reciprocal Rank Fusion**, **cross-encoder reranking**,
grounded generation with model routing, a **semantic cache**, an **evaluation
harness** (retrieval / faithfulness / latency), and a containerised FastAPI
deployment.

> Honesty note: the data and company are invented. Describe this as a system you
> built to learn production RAG — not as shipped work for any employer. You can
> defend every line because you wrote and understand every line.

## Architecture at a glance

```
            ┌──────────── ingestion (batch) ────────────┐
 docs ─▶ load ─▶ chunk ─▶ embed ─▶ vector store (Chroma/pgvector)
                                   │
 query ─▶ embed ─┬─▶ dense search ─┘
                 └─▶ BM25 sparse search
                          │
                      RRF fusion ─▶ cross-encoder rerank ─▶ top N
                          │
                  grounded prompt ─▶ LLM (routed) ─▶ answer + citations
                          ▲
                   semantic cache (short-circuits on hit)
```

Full design rationale: **[ARCHITECTURE.md](./ARCHITECTURE.md)**.
Interview Q&A drill: **[INTERVIEW_PREP.md](./INTERVIEW_PREP.md)**.

## Quickstart

```bash
# 1. install
make install                 # or: pip install -r requirements.txt

# 2. configure
cp .env.example .env         # add OPENAI_API_KEY (and optionally GROQ_API_KEY)

# 3. generate mock data + build the index
make data                    # writes data/raw/*.md and data/eval/eval_set.jsonl
make ingest                  # chunk -> embed -> upsert into the vector store

# 4. run the API
make serve                   # http://localhost:8000/docs
```

Query it:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"How many days of annual leave do full-time employees get?"}' | jq
```

```json
{
  "answer": "Full-time employees accrue 25 days of paid annual leave per year. [1]",
  "citations": [{"title": "Annual Leave & Time-Off Policy",
                 "source": "hr/leave-policy.md", "snippet": "..."}],
  "model": "gpt-4o-mini",
  "cached": false,
  "timings_ms": {"embed_ms": 41.2, "retrieve_ms": 18.7,
                 "rerank_ms": 120.4, "generate_ms": 610.0, "total_ms": 790.3}
}
```

## Running without API keys

Everything except the final LLM answer runs locally:

```bash
# fully local embeddings + reranker, retrieval/latency eval only
EMBEDDING_PROVIDER=local EMBEDDING_DIM=384 make ingest
make eval-retrieval          # Hit@k, MRR, p50/p95 latency — no LLM judge
```

## Evaluation

```bash
make eval                    # retrieval + latency + LLM-as-judge answer scoring
```

Use it to justify design choices with numbers, e.g. toggle `USE_RERANKER` and
re-run to show the reranker's effect on Hit-rate/MRR.

## Tests

```bash
make test                    # 11 unit tests: chunking, RRF, BM25, API
```

## Layout

| Path | What it is |
|------|------------|
| `src/rag/ingestion/` | loader, chunking, ingestion pipeline |
| `src/rag/embeddings/` | OpenAI / local embedder behind one interface |
| `src/rag/stores/` | ChromaStore + PgVectorStore (same `VectorStore` Protocol) |
| `src/rag/retrieval/` | BM25, RRF fusion, cross-encoder reranker |
| `src/rag/generation/` | prompts + routed LLM client |
| `src/rag/cache/` | semantic cache |
| `src/rag/pipeline.py` | end-to-end query orchestration |
| `src/rag/api/` | FastAPI service |
| `eval/run_eval.py` | evaluation harness |
| `scripts/` | mock-data generator + ingest CLI |
| `tests/` | unit tests |
