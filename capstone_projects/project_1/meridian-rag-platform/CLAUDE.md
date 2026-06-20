# CLAUDE.md — Meridian RAG Platform

Guidance for AI coding agents (and humans) working in this repo. Read this before
making changes.

## What this is

A layered RAG platform over a fictional company's internal docs ("Meridian
Corp"), built on the **LangChain** stack the course teaches. Layers: data
pipeline → gateway → guardrails → orchestration → provider → monitoring → eval,
with a Streamlit frontend.

**This is a portfolio / learning project, not a shipped product.** The company
and documents are synthetic. Do **not** rewrite the docs to claim production use
— the honest framing is deliberate (see `INTERVIEW_PREP.md`). Keep it.

## Standing rule: build on the taught stack

These projects exist to show the learner building with the tools the course
taught. **Use the taught stack — LangChain (loaders, splitters, embeddings,
vector stores, retrievers, LCEL, output parsers, callbacks, caching) — not
hand-rolled equivalents.** If you find yourself re-implementing a primitive
LangChain already provides (a text splitter, RRF fusion, a JSON parser), stop and
use the library component instead. Bare-code reimplementations defeat the point.

## Audience & style

Reader is a **mid-level Python developer**. Plain classes and small factory
functions (`build_embeddings()`, `build_models()`, `build_retriever()`); type
hints; short docstrings explaining *why*. No `Protocol`, ABCs, metaclasses, DI
frameworks, or custom middleware. Clarity over cleverness — any file should read
top to bottom.

## Where LangChain is used (and where it isn't)

LangChain handles the RAG plumbing: `data_pipeline/` (loaders, splitters,
embeddings, Chroma), `backend/app/retrieval/` (dense/sparse/ensemble/rerank
retrievers), `backend/app/orchestration/` (ChatPromptTemplate + LCEL chain +
model routing), `backend/app/providers/` (ChatOpenAI/ChatGroq/FakeListChatModel),
the cost **callback** in `monitoring/`, the LLM **cache** in `state.py`, and the
**PydanticOutputParser** judge in `eval/`.

Custom (no LangChain): `gateway/` (auth, rate limit, validation), `guardrails/`
(heuristics), `monitoring/store.py`/`metrics.py`/`anomaly.py`, the trace store.
That split is intentional — don't try to force LangChain into the gateway or the
guardrails.

## Commands

`PYTHONPATH=.` is required (the Makefile sets it).

```bash
make install     # LangChain stack + supporting libs
make ingest      # Layer 1: load -> split -> embed -> Chroma
make backend     # Layers 2-6: FastAPI on :8000
make frontend    # Client: Streamlit on :8501 (backend must be running)
make eval        # Layer 7: LLM-as-judge over sampled traces
make test        # 20 tests, keyless
docker compose up
```

## Layer map

```
common/            config, logging, types (API + Trace),
                   embedding (build_embeddings), vector_store (build_vectorstore, load_all_documents), users
data_pipeline/     loader -> splitter -> indexer -> run_ingest        (LangChain, offline)
frontend/          app.py (Streamlit), api_client.py
backend/app/
  gateway/         auth.py, rate_limit.py, validation.py              (custom)
  guardrails/      input_guards.py, output_guards.py, policies.py     (custom heuristics)
  orchestration/   pipeline.py (LCEL), prompts.py, model_router.py    (LangChain)
  retrieval/       dense, sparse, hybrid, rerank, retrievers          (LangChain)
  providers/       llm.py (build_models)                              (LangChain)
  monitoring/      store, cost, cost_callback (LC), metrics, anomaly, tracing
  routers/         user_routes.py (/v1/ask), admin_routes.py
  state.py         builds everything once; sets the global LLM cache
  main.py          FastAPI app
eval/              sampler, judge (LCEL + PydanticOutputParser), run_eval
```

## Pinned versions — important

LangChain **0.3.x line** (`langchain~=0.3`, `langchain-community~=0.3`, etc.),
because that's the API the course uses and where the classic import paths live
(`from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever`).
Do not bump to LangChain 1.x without updating every import — 1.x reorganised the
module layout and is sunsetting `langchain-community`.

## Invariants — do not break

- **Use LangChain components for RAG steps** (see standing rule above).
- **Dependency direction**: `common/` shared; `data_pipeline/` and `backend/`
  don't import each other; `frontend/` is HTTP-only.
- **Config is the single source of truth** (`common/config.py`). No inline magic
  numbers (chunk size, top-k, thresholds, prices, limits).
- **Keyless path must keep working**: no key -> `DeterministicFakeEmbedding` +
  `FakeListChatModel`, so ingest, the request flow, and all tests run with no
  network. Don't add a hard dependency on a real key. (The fake model bypasses
  the LLM cache, so "cached" only shows with a real model — that's expected.)
- **Token capture stays in the LangChain callback** (`cost_callback.py`), passed
  via the chain's `config`. A new provider just needs to report usage (or the
  callback's char estimate covers it).
- **One trace per request** powers monitoring, cost, and eval. New fields go on
  `common/types.Trace` and the `traces` table in `monitoring/store.py`.
- **Guardrails stay behind `check_input`/`check_output` returning `GuardOutcome`**
  even if you swap a heuristic for a real model.

## Adding things

- **More documents**: drop `.md` under `data_pipeline/documents/<category>/`, re-`make ingest`.
- **Another model**: one branch in `build_models()` + a price row in `monitoring/cost.py`.
- **Another retriever/technique**: prefer a LangChain retriever; wire it into
  `retrieval/retrievers.py`.
- **Another guardrail**: add to `input_guards.py`/`output_guards.py` with
  patterns/thresholds in `policies.py`.

## Testing

`make test` — 20 tests across the LangChain pipeline (loader, splitter, composed
retriever), guardrails, cost, gateway (auth + rate limit), and the API end to
end. Keyless via the fake embedding/model; ingests once in a fixture. Keep new
work covered and the suite keyless.
