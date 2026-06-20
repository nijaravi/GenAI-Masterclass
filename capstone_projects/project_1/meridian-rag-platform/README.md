# Meridian RAG Platform

A retrieval-augmented-generation system for a fictional company's internal docs
("Meridian Corp"), built as **separate, clearly-labelled layers** and on the
**LangChain stack** the course teaches — so each part of a production RAG system
can be read and understood on its own, using the tools you'd actually reach for.

```
Client / Frontend        Streamlit chat + admin dashboard (5 users)
        │  HTTP
API Gateway (FastAPI)     auth · rate limiting · request validation · routing
        │
Guardrails Layer          input: PII redaction, injection block
        │                 output: groundedness (hallucination), toxicity, PII leak
LLM Orchestration         LangChain: loaders · splitters · Chroma · hybrid retriever
        │                 (dense + BM25 + ensemble/RRF) · cross-encoder rerank ·
        │                 ChatPromptTemplate · LCEL chain · output parsers · routing
Model Provider            ChatOpenAI / ChatGroq — the LLM call  (+ keyless fake)
        │
Monitoring & Logging      latency · cost per user (via LC callback) · quality · alerts
        │
Eval Pipeline             LLM-as-judge as an LCEL chain, run async on sampled traffic
```

> **Honest framing.** This is a learning project. The company and documents are
> synthetic, and it's presented as a system built **to understand** how a
> production RAG platform fits together — not as something shipped in production.
> Keep that framing (see `INTERVIEW_PREP.md`).

## Built on the taught stack (LangChain)

This project deliberately uses LangChain end to end, because that's what the
course teaches for real builds:

| Concern | LangChain component used |
|---------|--------------------------|
| Load documents | `DirectoryLoader` + `TextLoader` |
| Split | `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` (token-based) |
| Embed | `OpenAIEmbeddings` (or `DeterministicFakeEmbedding` keyless) |
| Vector store | `langchain_chroma.Chroma` |
| Dense retrieval | `vectorstore.as_retriever(...)` |
| Sparse retrieval | `BM25Retriever` |
| Hybrid fusion | `EnsembleRetriever` (RRF under the hood) |
| Reranking | `ContextualCompressionRetriever` + `CrossEncoderReranker` |
| Prompt | `ChatPromptTemplate` |
| Chain | LCEL: `prompt \| model \| StrOutputParser()` |
| Model | `ChatOpenAI`, `ChatGroq` (`FakeListChatModel` keyless) |
| Structured eval output | `PydanticOutputParser` |
| Token/cost capture | a LangChain `BaseCallbackHandler` |
| LLM response cache | `set_llm_cache(InMemoryCache())` |

What stays **custom** (LangChain has no opinion here): the API gateway,
auth/rate-limit, the guardrail heuristics, the per-user trace store, metrics, and
anomaly alerts. That split — *framework for the RAG plumbing, custom code for the
production platform around it* — is itself the architecture story.

## Repository layout

```
common/             config, logging, types, users (5; user 1 = admin),
                    embedding (LangChain), vector_store (LangChain Chroma)
data_pipeline/      LAYER 1  — loader -> splitter -> indexer -> run_ingest  (offline)
frontend/           CLIENT   — Streamlit chat + admin dashboard
backend/app/
  gateway/          LAYER 2  — auth, rate limit, validation
  guardrails/       LAYER 3  — input_guards, output_guards, policies (custom heuristics)
  orchestration/    LAYER 4  — pipeline (LCEL), prompts, model_router
  retrieval/        LAYER 4  — dense, sparse, hybrid, rerank, retrievers (all LangChain)
  providers/        LAYER 5  — llm.py (ChatOpenAI/ChatGroq/fake)
  monitoring/       LAYER 6  — store (SQLite traces), cost, cost_callback, metrics, anomaly
  routers/          user_routes (/v1/ask), admin_routes (/v1/admin/*)
eval/               LAYER 7  — sampler, judge (LCEL + parser), run_eval
tests/              20 tests across the layers
```

`data_pipeline/` and `frontend/` each have their own README.

## Quickstart

```bash
make install                       # dependencies (LangChain stack)
cp .env.example .env               # optional: add OPENAI_API_KEY / GROQ_API_KEY
make ingest                        # Layer 1: load -> split -> embed -> Chroma
make backend                       # Layers 2-6: FastAPI on :8000  (/docs)
make frontend                      # Client: Streamlit on :8501  (second terminal)
make eval                          # Layer 7: judge a sample of recent traffic
```

Or run backend + frontend together: `docker compose up`.

### Keyless demo mode

With **no API keys**, the platform still runs end to end: `DeterministicFakeEmbedding`
handles vectors and `FakeListChatModel` returns a canned answer, so the whole flow
— gateway, guardrails, retrieval, monitoring, cost, eval — works. Set
`OPENAI_API_KEY` or `GROQ_API_KEY` in `.env` for real answers and meaningful eval
scores. (Note: the fake model bypasses LangChain's LLM cache, so "cached" only
registers with a real model.)

## The 5 users + admin

Five users are seeded in `common/users.py`, each with an API key sent in the
`X-API-Key` header. **User 1 (Aisha) is also the admin.** Every request is
attributed to its user for rate limiting and cost.

| Endpoint | Who | Returns |
|----------|-----|---------|
| `POST /v1/ask` | any user | answer + citations + guardrail notes + timings |
| `GET /v1/admin/metrics` | admin | latency p50/p95, block rate, cache-hit rate, quality, alerts |
| `GET /v1/admin/usage` | admin | request counts per user |
| `GET /v1/admin/cost` | admin | **dollar cost per user** + total |
| `GET /v1/admin/traces` | admin | recent request traces |

```bash
curl -s -X POST localhost:8000/v1/ask \
  -H "X-API-Key: key-ben-002" -H "Content-Type: application/json" \
  -d '{"question":"How many days of annual leave do I get?"}'
```

## Tests

```bash
make test     # 20 tests; run keyless, no network needed
```

## What's intentionally simplified

Guardrails are lightweight heuristics (regex PII, pattern-based injection,
toxicity wordlist, word-overlap groundedness), not ML classifiers — each file
names the production swap. Auth is API-key lookup against five hard-coded users.
Rate limiting is in-memory. The full "what I'd add for production" list is in
`ARCHITECTURE.md`.
