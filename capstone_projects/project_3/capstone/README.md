# Multi-Agent Orchestrator — Capstone Reference Build

A single chat interface serving **three audiences** — developers, clients,
and customers — routed by a **LangGraph** orchestrator to the right
capability behind the scenes: a **CrewAI**-backed coding crew, a real
**MCP** tool server, Chroma-backed **RAG**, or an **A2A**-style external
agent boundary.

This is a portfolio/capstone build modeled on a documented enterprise
reference architecture (three-audience chat platform, four-technology
stack). It makes real LLM calls end to end — the Planner's classification,
the RAG Agent's synthesis, and the CrewAI crew's `kickoff()` all go
through a live OpenAI model — because that's the actual skill this build
exists to practice. What's simplified is everything *around* the LLM
calls: no separate MCP transport process, no separate A2A network
service, no generic multi-hop queue. See "What's simplified" below.

Companion document: `docs/WALKTHROUGH.docx` explains the code flow
section by section, in the order a request actually moves through it.

## Why these four frameworks, and only these

| Technology | Where it lives | Why |
|---|---|---|
| **LangGraph** | The whole graph (every node) | Control plane. Routing = a conditional edge, shared state flows natively. |
| **CrewAI** | Nested inside *one* node (`coder`) | Role-based collaboration where it earns its keep; not a second router. |
| **MCP** | The `mcp` route's backend calls | Default integration path for tools/data the platform doesn't own. |
| **A2A** (simplified) | One boundary node (`external_agent`) | Only for delegating to another team's/vendor's *independently-run* agent. |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY - required, there's no offline fallback
uvicorn app.main:app --reload
```

The app will refuse to start with a clear error if `OPENAI_API_KEY` isn't
set. Startup also seeds an in-memory Chroma RAG index — retrieval uses a
free, local embedding model (~80MB, downloaded once) rather than an
OpenAI embedding call, which is a normal, common production choice, not
a shortcut around using a real LLM (that's what the Planner, RAG
synthesis, and the CrewAI crew are for).

Open `frontend/index.html` directly in a browser (it talks to
`http://localhost:8000` by default), or drive it with curl — try the four
worked examples from the design doc's Section 7:

```bash
# customer -> RAG
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id":"s1","user_role":"customer","message":"Does the TrailPro backpack come in a larger size?"}'

# client -> MCP (billing/build-status server)
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id":"client-4471","user_role":"client","message":"What is the build status on our latest deployment?"}'

# developer -> multi-hop: RAG doc lookup, then CrewAI-backed coder agent
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id":"s3","user_role":"developer","message":"Show me how to call the internal auth API, and check if PR #482 covers ticket JIRA-1190."}'

# client -> A2A-style external vendor lookup
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id":"s4","user_role":"client","message":"Can you check the status of my return, order ord-7712, with the logistics partner?"}'
```

Note: the developer example makes several real LLM calls in one turn
(classification, RAG synthesis, then three CrewAI agents) — expect it to
take noticeably longer than the others, and to use real API credits.

## Running the eval harness

```bash
python tests/test_routing.py
```

Scores routing accuracy against the four labeled requests above. This
also makes real LLM calls (it runs the actual graph), so it needs
`OPENAI_API_KEY` set too.

## What's simplified here, and how to go further

The LLM is real everywhere it should be. What's simplified is the
plumbing *around* it — each one is called out in code comments, and the
swap-in path is one or two files, never a rewrite:

| Simplified to... | Production version | Where to change it |
|---|---|---|
| MCP tool called in-process (`server.call_tool(...)`) | MCP server as its own process, called over stdio/HTTP | `app/nodes/specialists.py`'s `mcp_tool_node` — swap for a real `ClientSession` + transport. |
| A2A vendor agent simulated in-process (`app/a2a/vendor_agent.py`) | A real HTTP service with a published Agent Card | `call_external_agent()` — same call signature, swap the body for an `httpx` call. |
| Chroma, in-memory, local embeddings | Pinecone, namespace-isolated, OpenAI embeddings | `app/rag/vector_store.py` — same `VectorStore` interface. |
| In-memory LangGraph checkpointer | Postgres-backed checkpointer | `app/graph.py`'s `build_graph()` — swap `MemorySaver()` for `PostgresSaver(...)`. |
| Single queued follow-up (`next_category`) | A generic multi-hop queue | `app/state.py` / `app/nodes/orchestrator.py` — turn the field back into a list if you need more than one follow-up. |

## Repo layout

```
app/
  config.py            env/config - requires OPENAI_API_KEY, loads .env
  state.py             OrchestratorState — the LangGraph shared state
  llm.py               get_model() — the one place every node gets an LLM from
  graph.py             the StateGraph wiring — start here
  main.py              FastAPI app (the "single chat interface")
  nodes/
    planner.py           Planner Agent (real classification call)
    orchestrator.py       Orchestrator Agent (routing + role gate + hop guard, deterministic)
    specialists.py         intake, RAG (real synthesis call), coder, MCP, external agent, finalize
  crew/coder_crew.py    the nested CrewAI crew — real kickoff()
  mcp_servers/          two real MCP servers (billing, build-status)
  a2a/vendor_agent.py   the simulated A2A counterpart
  rag/                  vector store (Chroma, local embeddings) + seed data
tests/test_routing.py   routing-accuracy eval harness (makes real LLM calls)
frontend/index.html     single-file chat UI with a live routing-trace panel
docs/WALKTHROUGH.docx   step-by-step code-flow walkthrough
```

## A note on framing this in an interview

This models the architecture pattern from a real enterprise reference doc
(a three-audience platform, LangGraph + CrewAI + MCP + A2A) — it's a
capstone project built to learn and demonstrate that pattern, not a claim
of having shipped it in production for a specific employer. Speak to it
as "a platform I built to work through this architecture," and you can
defend every line, because you wrote and ran all of it.
