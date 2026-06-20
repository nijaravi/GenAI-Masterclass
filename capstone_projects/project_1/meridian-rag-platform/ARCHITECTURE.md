# Architecture

How the platform is structured and **why**, layer by layer. Two ideas drive it:

1. A real RAG product is a request passing through layers (gateway → guardrails →
   orchestration → provider → monitoring), with an offline data pipeline feeding
   it and an offline eval loop watching it. Each is a separate, readable piece.
2. The RAG plumbing is built on **LangChain** — the stack the course teaches for
   real work — while the production wrapper around it (gateway, guardrails,
   monitoring) stays custom, because that's not what a RAG framework is for.

## Why LangChain for the plumbing, custom for the rest

LangChain gives you battle-tested, swappable components for exactly the RAG steps
that are tedious and easy to get subtly wrong: loading, splitting, embedding,
vector storage, the dense/sparse/ensemble/rerank retrieval stack, prompt
templating, the chain that ties model + parser together, and structured output
parsing. Re-implementing those by hand would be slower, more bug-prone, and
wouldn't reflect how the course says to build. So the data pipeline and the
orchestration/retrieval layers are LangChain.

The gateway, auth, rate limiting, guardrails, the per-user trace store, cost
attribution, metrics, and anomaly alerts are **app/infra concerns** LangChain has
no opinion on, so they're plain custom code. Keeping that line clear is part of
the design.

Dependency direction is one-way: `common/` is shared; the data pipeline and the
backend both depend on `common/` but **not on each other**; the frontend speaks
only HTTP.

## The request lifecycle (`POST /v1/ask`)

`backend/app/routers/user_routes.py` reads top to bottom:

```
1. Auth            gateway/auth.py        X-API-Key -> User (401 if unknown)
2. Validation      gateway/validation.py  non-empty question, known category
3. Rate limit      gateway/rate_limit.py  per-user window (429 if over)
4. Input guards    guardrails/input_guards.py  block injection; redact PII
5. Orchestration   orchestration/pipeline.py   the LCEL RAG pipeline (below)
6. Output guards   guardrails/output_guards.py block toxicity/PII leak; flag hallucination
7. Monitoring      monitoring/tracing.py + store.py   write one trace row
8. Response        answer + citations + notes + timings
```

Guardrails and monitoring wrap the orchestration *in the router*, so the pipeline
stays purely about RAG and each layer is testable alone.

## Layer 1 — Data pipeline (offline, LangChain)

`load → split → embed+index`, run via `python -m data_pipeline.run_ingest`.

- **Load** — `DirectoryLoader` + `TextLoader` read `documents/<category>/*.md`;
  we enrich each `Document`'s metadata with category, a clean `source`, and a
  `title`.
- **Split** — the course's markdown approach: `MarkdownHeaderTextSplitter` splits
  on the heading structure (so a chunk doesn't mix sections and headings become
  metadata), then `RecursiveCharacterTextSplitter.from_tiktoken_encoder` enforces
  a token budget with overlap. Headings are prepended back so each chunk is
  self-describing.
- **Index** — `langchain_chroma.Chroma.add_documents(..., ids=chunk_ids)`. Stable
  ids make re-ingest idempotent.

See `data_pipeline/README.md`.

## Layer 2 — API gateway

Plain FastAPI. Auth and admin-gating are dependencies (`Depends(require_user)` /
`Depends(require_admin)`). Rate limiting is a per-user fixed window in memory.
Validation beyond Pydantic's checks lives in `validation.py`.

## Layer 3 — Guardrails (custom)

Input/output checks with shared patterns in `policies.py`:
- **Input**: injection phrases → **block**; PII (email/phone/card/SSN) → **redact
  and continue**.
- **Output**: toxicity → **block**; PII leak → **block**; low groundedness
  (answer-word overlap with the context) → **flag** (the "I don't have that
  information" reply is allowed to score low).

Deliberately simple heuristics; each file names the production swap (Presidio, a
real injection/toxicity classifier, an NLI/judge model). LangChain isn't used
here on purpose — it has no first-class guardrail primitive without extra deps,
and the *placement* of the checks is the lesson.

## Layer 4 — LLM orchestration (LangChain + LCEL)

`orchestration/pipeline.py` composes the retrieval stack and runs an LCEL chain:

```
retrieval/retrievers.py  build_retriever():
    dense (vectorstore.as_retriever)  ─┐
    sparse (BM25Retriever)             ├─ EnsembleRetriever (RRF)
                                       ┘     -> ContextualCompressionRetriever
                                                  (CrossEncoderReranker, top_n)

pipeline.run():
    docs   = retriever.invoke(question)
    context= format_docs(docs)                         # numbered [1],[2] passages
    model  = models[ route(question) ]                 # cheap vs smart
    answer = (ANSWER_PROMPT | model | StrOutputParser()).invoke(...)
```

- **Hybrid retrieval** is `EnsembleRetriever`, which fuses the dense and sparse
  result lists with Reciprocal Rank Fusion internally — no score normalisation
  needed. Dense gives meaning; BM25 gives exact-token matches (codes, policy
  numbers).
- **Reranking** is a cross-encoder wrapped as `ContextualCompressionRetriever`, so
  the whole retrieve-wide-then-rerank-narrow flow is still one retriever. Falls
  back to the base retriever if the model can't load.
- **Generation** is an LCEL chain: `ChatPromptTemplate | model | StrOutputParser`.
- **Model routing** picks a cheap or smart model per question by a length/
  complexity heuristic.
- **Caching** uses LangChain's global LLM cache (`set_llm_cache(InMemoryCache())`,
  set in `state.py`). A request is "cached" if the cost callback saw zero LLM
  runs. (Semantic caching is available via LangChain integrations like a Redis
  semantic cache; we use the in-memory exact cache to stay dependency-light. The
  keyless fake model bypasses caching.)

## Layer 5 — Model provider (LangChain)

`providers/llm.py` → `build_models()` returns `ChatOpenAI` (fast/smart),
`ChatGroq`, and always a `FakeListChatModel` so the system runs keyless. Each is a
LangChain Runnable that drops into the chain. Adding a provider is one more branch
— no interface to implement.

## Layer 6 — Monitoring & logging

One SQLite table (`monitoring/store.py`), one row per request:
- **Token capture** is a LangChain callback (`cost_callback.py`) passed in the
  chain's `config`; `on_llm_end` reads usage (with a char-based estimate fallback
  for models that don't report it). This is the "maximal LangChain" touch in a
  wrapper layer.
- `cost.py` prices tokens per model; `metrics.py` aggregates the rows (latency
  p50/p95, block/cache/quality); `anomaly.py` applies threshold alerts.

The same table is what the eval pipeline samples, so quality scores live beside
latency and cost.

## Layer 7 — Eval pipeline (offline, LangChain + LCEL)

`eval/` samples unjudged traces and runs an **LLM-as-judge** built as an LCEL
chain with a **`PydanticOutputParser`** — the parser injects format instructions
and parses the reply straight into a typed `JudgeScore` (relevance, faithfulness,
reason), instead of hand-parsing JSON. Scores are written back onto the trace and
surface in the admin metrics and anomaly check. Runs on a schedule, off the
request path.

## Stores

- **Vector store**: `langchain_chroma.Chroma`, persistent on disk, cosine space.
  `load_all_documents` pulls chunks back to build the BM25 retriever at startup.
- **Trace store**: SQLite, one table, above.

## What I'd add for production (honest list)

- Real auth (OIDC/JWT) and per-user document entitlements feeding the retrieval
  filter — today any user can retrieve any chunk.
- Guardrails as real models (Presidio, injection/toxicity classifiers, an
  NLI/judge groundedness model).
- A shared cache + rate limiter (Redis); a LangChain Redis **semantic** cache
  instead of the in-memory exact cache.
- Real observability (LangChain callbacks → LangSmith, or OpenTelemetry →
  Datadog) instead of the SQLite trace table; statistical anomaly detection.
- Streaming responses; enforced per-user spend caps; retrieval ACL filtering.
- Retrieval eval with a gold set (Hit@k / MRR) and scheduled judge runs trending
  quality over time.

Knowing where these go — and that they're not built — is the point of the layered
structure.
