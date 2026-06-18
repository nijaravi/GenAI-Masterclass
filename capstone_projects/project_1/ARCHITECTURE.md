# Architecture & Best Practices

This document explains **how the codebase is structured and why**, the
engineering best practices it demonstrates, and the trade-offs behind each
decision. Read it alongside the code — every claim here maps to a file you can
open.

---

## 1. Design principles

**Interfaces at every boundary.** The embedder, the vector store, the reranker,
and the LLM client are each accessed through a small interface (`Protocol` or a
`build_*()` factory). The query pipeline depends on the *interface*, never a
concrete class. This is why "ChromaDB **or** pgvector" and "OpenAI **or** local
embeddings" are config flags, not rewrites — and why the whole pipeline is
testable with fakes (see `tests/test_api.py`).

**Config is centralised and validated.** `src/rag/config.py` is the single
source of truth, built on `pydantic-settings`. No module reads `os.environ`
directly. The same container image runs in dev, staging, and prod — only the
environment differs. Invalid config fails fast at startup, not mid-request.

**Ingestion and query are separated.** They have opposite runtime profiles:
ingestion is batch, write-heavy, and scheduled; querying is latency-sensitive
and read-only. Keeping them in different modules stops batch-only dependencies
from leaking into the hot request path.

**Everything is observable.** Each pipeline stage records its latency; every log
line is structured JSON carrying a `request_id` so one user query is traceable
across embed → retrieve → rerank → generate.

---

## 2. Module-by-module

```
src/rag/
├── config.py            # validated settings (pydantic-settings)
├── logging_config.py    # structured JSON logging + request-id context
├── models.py            # pydantic schemas shared across the app
├── ingestion/
│   ├── loader.py        # source docs -> Document objects (swap for connectors)
│   ├── chunking.py      # structural + token-window chunking
│   └── pipeline.py      # load -> chunk -> embed -> upsert
├── embeddings/
│   └── embedder.py      # OpenAIEmbedder | LocalEmbedder (one interface)
├── stores/
│   ├── base.py          # VectorStore Protocol
│   ├── chroma_store.py  # dev / single-node default
│   └── pgvector_store.py# prod path on Postgres (HNSW index)
├── retrieval/
│   ├── bm25.py          # lexical sparse retriever
│   ├── hybrid.py        # Reciprocal Rank Fusion
│   └── reranker.py      # cross-encoder, retrieve-wide-rerank-narrow
├── generation/
│   ├── prompts.py       # grounding system prompt + context builder
│   └── llm.py           # routed client (GPT-4o / mini / Groq Llama)
├── cache/
│   └── semantic_cache.py# embed-and-compare query cache
├── pipeline.py          # the orchestrator the API calls
└── api/
    ├── dependencies.py  # build heavy singletons once at startup
    └── main.py          # FastAPI app, middleware, handlers
```

---

## 3. The retrieval pipeline in depth

### 3.1 Chunking (`ingestion/chunking.py`)

The most under-appreciated quality lever in RAG. Two stages:

1. **Structural split** on markdown headings, so a chunk rarely straddles two
   unrelated sections. Each chunk keeps its heading prepended, making it
   self-describing for both the embedding model and the human reading a citation.
2. **Token-window split** within any oversized section: a fixed-size token
   window (`chunk_tokens=512`) slides with `chunk_overlap_tokens=64` of overlap,
   so a sentence cut at one boundary still appears intact in a neighbour.

Token counting uses **tiktoken** (the tokenizer GPT models actually use), so
"512 tokens" means the same thing for chunking and for the context budget.
`chunk_id` is deterministic (`{doc_id}::{position}`) which makes re-ingestion
idempotent — you overwrite, you don't duplicate.

**Trade-off:** bigger chunks = more context per hit but a blurrier embedding
(one vector trying to represent several ideas) and more wasted context budget;
smaller chunks = sharper embeddings but risk severing the context a fact needs.
512/64 is a sensible default for prose docs; code or tables want different
settings.

### 3.2 Embeddings (`embeddings/embedder.py`)

`text-embedding-3-small` (1536-d) by default — cheap and strong. A local
`all-MiniLM-L6-v2` (384-d) option runs offline for air-gapped or cost-sensitive
deployments. **Batching** is deliberate: embedding one chunk per request is the
classic ingestion bottleneck and multiplies API cost via per-call overhead.

### 3.3 Vector store (`stores/`)

Both implementations satisfy one `VectorStore` Protocol:

- **ChromaDB** — zero-ops, persists to disk, no server. Right for dev and
  small/medium corpora.
- **pgvector** — when you already run Postgres: vectors live next to relational
  data, you reuse existing backup/HA/permissions, and you filter on metadata in
  plain SQL. Uses an **HNSW** index (`vector_cosine_ops`) for fast approximate
  nearest-neighbour search at scale.

We compute embeddings ourselves and pass them in explicitly, so the *exact same
vectors* flow to either store. Cosine space throughout (vectors normalised),
which is why Chroma's cosine *distance* is converted to a similarity
(`1 - distance`) on the way out.

### 3.4 Hybrid search + RRF (`retrieval/bm25.py`, `retrieval/hybrid.py`)

Dense embeddings capture **meaning** but are blind to exact tokens the model
never trained on — error codes (`GP-1107`), SKUs, policy numbers, acronyms.
**BM25** is a lexical scorer that nails those. Running both and fusing is why
hybrid beats either alone.

Fusing is done with **Reciprocal Rank Fusion**, not score addition, because
cosine (`[0,1]`) and BM25 (unbounded) live on different scales and normalising
them is fragile. RRF fuses on **rank**:

```
rrf_score(d) = Σ_retrievers  1 / (k + rank_r(d))     # k = 60
```

A doc ranked highly by *either* retriever rises; one ranked highly by *both*
wins. It's parameter-light and consistently strong — the reason it's the default
fusion method in production hybrid search.

### 3.5 Reranking (`retrieval/reranker.py`)

Retrieval scores query and chunk **independently** (fast, but never sees the
pair together). A **cross-encoder** scores `(query, chunk)` **jointly** — far
better at separating "mentions the keyword" from "actually answers the
question". The cost is latency, so we apply it only to the ~12 fused candidates,
not the corpus: **retrieve wide, rerank narrow.** This is the single biggest
precision win in the pipeline. If the model can't load, we pass through
gracefully (degrade, don't crash).

### 3.6 Generation (`generation/`)

The **system prompt is the cheapest anti-hallucination guardrail**: answer only
from context, say "I don't have that" when the context is silent, cite the
bracketed passage numbers. Context passages are numbered so the model's `[2]`
maps back to a real source in the API response. A `max_context_chars` guard
prevents context overflow.

**Model routing** (`llm.py`): short factual lookups go to the fast tier
(`gpt-4o-mini`, or Llama-3.1-8b on Groq); long/multi-part questions go to
`gpt-4o`. Because OpenAI and Groq share an OpenAI-compatible API, one client
class handles both by swapping `base_url` + key. This is the cost/latency lever
the resume refers to.

### 3.7 Semantic cache (`cache/semantic_cache.py`)

An exact-string cache misses "what's the leave policy?" vs "how many vacation
days do I get?". The semantic cache embeds the query and serves a stored answer
when cosine similarity to a past query clears a **strict** 0.95 threshold —
strict on purpose, because a loose cache hit returns a confidently wrong answer.
Cached hits skip retrieval, rerank, **and** the LLM call. In-process LRU here;
back it with Redis (e.g. RedisVL) in production so it's shared across replicas.

---

## 4. The service (`api/`)

- **Warm-up on startup** via FastAPI's `lifespan`: heavy singletons (embedder,
  cross-encoder, BM25 index) are built once, so the first user request doesn't
  pay for model loading. Rebuilding a CrossEncoder per request is a classic
  RAG-in-prod mistake that adds seconds of latency.
- **Request-id middleware** stamps every request and response (`x-request-id`)
  and logs method/path/status/latency.
- **Global exception handler** logs full detail server-side but returns a
  generic message — never leak stack traces to callers.
- **`/healthz`** is dependency-free so the orchestrator's liveness probe stays
  honest; **`/stats`** exposes index size and active config for debugging.

---

## 5. Logging & observability (`logging_config.py`)

JSON logs because in any real deployment logs are shipped to a collector
(CloudWatch / ELK / Datadog) and queried as **fields**, not grepped as prose. A
`request_id` is carried in a `ContextVar` and injected into every record, so a
single query is traceable across all stages without threading the id through
every function. Per-stage timings (`embed_ms`, `retrieve_ms`, `rerank_ms`,
`generate_ms`) are logged on every query so latency regressions are visible and
attributable to a stage.

**What to add for full production observability:** Prometheus metrics
(histograms for the same timings, counters for cache hit-rate and "no answer"
responses), distributed tracing (OpenTelemetry spans per stage), and token/cost
accounting per request.

---

## 6. Evaluation (`eval/run_eval.py`)

You cannot tune what you cannot measure. Two layers:

**Retrieval metrics** (no LLM — cheap, deterministic): given a gold set mapping
questions to the documents that should answer them:
- **Hit@k** — did a relevant chunk reach the top-k?
- **MRR** — how high did the first relevant chunk rank?
These isolate the *retriever*: if the right chunk never arrives, no prompt
tuning will save the answer.

**Answer metrics** (LLM-as-judge): given good context, is the answer good?
- **Relevance** — does it address the question? (1–5)
- **Faithfulness** — is every claim grounded in the retrieved context, i.e. NOT
  hallucinated? (1–5)
- **Latency** — p50 / p95 / mean, from the pipeline's own timings.

This harness is how you answer "how do you know X helped?" — flip the flag,
re-run, compare. Example experiments: `USE_RERANKER` on/off (precision),
`chunk_tokens` sweep (context granularity), dense-only vs hybrid (recall on
keyword-heavy queries).

**Limits to be honest about:** LLM-as-judge is correlated with, not identical
to, human judgement; the gold set is small; and these are offline metrics —
production also needs online signals (thumbs up/down, click-through, escalation
rate).

---

## 7. Testing strategy (`tests/`)

Unit tests target the **pure, high-leverage logic** that's cheap to test and
expensive to get wrong: token-window overlap, chunk-id stability, RRF ranking
math, BM25 lexical matching and filtering, and the API contract via a **fake
pipeline** (no models or keys needed — that's the dependency-injection seam
paying off). What's intentionally *not* unit-tested: third-party clients
(OpenAI, Chroma) — those belong in integration tests against real or mocked
services in CI.

---

## 8. Deployment (`Dockerfile`, `docker-compose.yml`)

- **Multi-stage-friendly slim base**, deps installed before source copy so the
  layer cache survives source-only changes.
- **Non-root user** — basic container hardening.
- **Container `HEALTHCHECK`** mirrors the orchestrator probe.
- **docker-compose** brings up the API next to a `pgvector/pgvector:pg16`
  Postgres so the pgvector path is exercisable with one command.
- CI/CD (GitHub Actions in a real repo): lint → test → build image → push →
  deploy with a manual prod gate (mirrors `engineering/deployment-runbook.md`).

---

## 9. Production hardening checklist (what's deliberately out of scope here)

A learning project shouldn't pretend to be a hardened platform. If asked "what
would you add for real production?", this is the honest list:

- **Auth** on the API (OAuth/JWT, per-tenant isolation).
- **Rate limiting** and request quotas.
- **Redis-backed** semantic cache shared across replicas.
- **Async/concurrent** dense+sparse retrieval (they're independent).
- **Incremental ingestion** with change detection rather than full re-index.
- **Document-level access control** — filter retrieval by the caller's
  permissions so the LLM never sees a chunk the user can't read (critical: a RAG
  system is an exfiltration risk if retrieval ignores ACLs).
- **PII handling / redaction** before embedding sensitive corpora.
- **Prometheus + OpenTelemetry**, cost/token dashboards, alerting on faithfulness
  drops.
- **Eval in CI** as a quality gate, with a larger human-reviewed gold set.
