# Multi-Agent Orchestrator — Capstone Reference Build

A single chat interface serving **three audiences** — developers, clients,
and customers — routed by a **LangGraph** orchestrator to the right
capability behind the scenes: a **CrewAI**-backed coding crew, real
**MCP** tool servers, Pinecone/Chroma-backed **RAG**, or an **A2A**
external-agent boundary.

This is a portfolio/capstone build modeled on a documented enterprise
reference architecture (three-audience chat platform, four-technology
stack). It's built to be genuinely run, read, and defended in an
interview — not a slide-deck description of one.

Companion document: `docs/WALKTHROUGH.md` (or the delivered `.docx`)
explains the code flow section by section.

## Why these four frameworks, and only these

| Technology | Where it lives | Why |
|---|---|---|
| **LangGraph** | The whole graph (every node) | Control plane. Routing = a conditional edge, shared state flows natively. |
| **CrewAI** | Nested inside *one* node (`coder`) | Role-based collaboration where it earns its keep; not a second router. |
| **MCP** | Any deterministic backend system | Default integration path for tools/data the platform doesn't own. |
| **A2A** | One boundary node (`external_agent`) | Only for delegating to another team's/vendor's *independently-run, reasoning* agent. |

## Quickstart (zero external accounts needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # DEMO_MODE=true by default — no API keys needed
uvicorn app.main:app --reload
```

The app auto-launches the mock external vendor agent (the A2A
counterpart) in DEMO_MODE, and seeds the Chroma-backed RAG index on
startup. Open `frontend/index.html` directly in a browser (it talks to
`http://localhost:8000` by default), or drive it with curl:

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id":"s1","user_role":"customer","message":"Does the TrailPro backpack come in a larger size?"}'
```

Try the other worked examples from the design doc (Section 7):

```bash
# client -> MCP (billing/build-status server)
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id":"client-4471","user_role":"client","message":"What is the build status on our latest deployment?"}'

# developer -> multi-hop: RAG doc lookup, then CrewAI-backed coder agent
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id":"s3","user_role":"developer","message":"Show me how to call the internal auth API, and check if PR #482 covers ticket JIRA-1190."}'

# client -> A2A external vendor agent
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id":"s4","user_role":"client","message":"Can you check the status of my return, order ord-7712, with the logistics partner?"}'
```

## Running the eval harness

```bash
python tests/test_routing.py
```

Scores routing accuracy against the four labeled requests above — this is
the "evaluation harness" pattern the design doc calls for in its
Observability section, kept small and legible rather than exhaustive.

## Going live (real LLM + real Pinecone)

```
DEMO_MODE=false
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
```

Nothing else changes: `app/llm.py` and `app/rag/vector_store.py` are the
only two files that know about demo vs. live, and every node talks to
their small interface (`get_chat_model()`, `get_vector_store()`) rather
than to OpenAI or Pinecone directly.

## Repo layout

```
app/
  config.py            single source of truth for env/config
  state.py             OrchestratorState — the LangGraph shared state
  llm.py               model provider (demo rule-based / real OpenAI)
  graph.py             the StateGraph wiring — start here
  main.py              FastAPI app (the "single chat interface")
  nodes/               one file per graph node
  crew/coder_crew.py   the nested CrewAI crew (Section 4.7)
  mcp_servers/         two real MCP servers (billing, build-status)
  a2a/                 the A2A boundary: client + mock vendor agent
  rag/                 vector store (Pinecone/Chroma) + seed data
tests/test_routing.py  routing-accuracy eval harness
frontend/index.html    single-file chat UI with a live routing-trace panel
docs/WALKTHROUGH.md    step-by-step code-flow walkthrough
```

## A note on framing this in an interview

This models the architecture pattern from a real enterprise reference doc
(a three-audience platform, LangGraph + CrewAI + MCP + A2A) — it is a
capstone project built to learn and demonstrate that pattern, not a claim
of having shipped it in production for a specific employer. Speak to it as
"a platform I built to work through this architecture" and you can defend
every line of it, because you can.
