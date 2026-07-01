"""
main.py — FastAPI entry point for the Walmart Multi-Agent AI Platform.

Endpoints:
  POST /chat          — Main chat endpoint (runs full pipeline)
  GET  /health        — Health check (includes vector DB mode)
  GET  /personas      — Returns available user roles
  GET  /agent-map     — Returns the routing architecture description
  GET  /db-info       — Returns PGVector schema and connection info
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    ChatRequest, ChatResponse, UserRole,
    DEMO_MODE, PGVECTOR_CONNECTION_STRING,
)
from backend.pipeline import run_pipeline

app = FastAPI(
    title="Walmart Multi-Agent AI Platform",
    description=(
        "A multi-agent orchestrator chatbot serving three personas: "
        "Customer, Client (vendor/supplier), and Developer. "
        "Routes queries to RAG, Tool-Call, Coder, or MCP agents. "
        "Vector store: PGVector (prod) / FAISS (demo)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    vector_db = (
        "FAISS in-memory (DEMO_MODE=true)"
        if DEMO_MODE
        else "PGVector — PostgreSQL + pgvector extension"
    )
    return {
        "status": "ok",
        "demo_mode": DEMO_MODE,
        "vector_db": vector_db,
        "version": "2.0.0",
        "platform": "Walmart Multi-Agent AI Platform",
    }


@app.get("/db-info")
def db_info():
    """
    Returns PGVector schema and connection details.
    Useful for interview demos to explain the DB layer.
    """
    conn_display = (
        "Not applicable (DEMO_MODE=true, using in-memory FAISS)"
        if DEMO_MODE
        else PGVECTOR_CONNECTION_STRING.split("@")[-1]  # host/db only, no credentials
    )
    return {
        "vector_db": "pgvector" if not DEMO_MODE else "faiss (demo)",
        "connection": conn_display,
        "table": "walmart_documents",
        "schema": {
            "id":         "TEXT PRIMARY KEY",
            "persona":    "TEXT  -- 'customer' | 'client' | 'developer'",
            "category":   "TEXT",
            "content":    "TEXT",
            "embedding":  "vector(1536)  -- pgvector column",
            "created_at": "TIMESTAMPTZ DEFAULT NOW()",
        },
        "index": {
            "type":      "IVFFlat",
            "operator":  "vector_cosine_ops",
            "lists":     10,
            "note":      "Use HNSW for >1M vectors (pgvector >=0.5.0)",
        },
        "similarity_operator": "<=>  (cosine distance — lower = more similar)",
        "persona_isolation": "WHERE persona = $1 in every query — enforced at SQL layer",
        "docker_quickstart": (
            "docker run -d --name pgvector-dev "
            "-e POSTGRES_PASSWORD=postgres "
            "-e POSTGRES_DB=walmart_ai "
            "-p 5432:5432 "
            "pgvector/pgvector:pg16"
        ),
    }


@app.get("/personas")
def get_personas():
    return {
        "personas": [
            {
                "id": UserRole.CUSTOMER,
                "label": "Customer",
                "description": "Walmart shoppers — product Q&A, returns, deals, Walmart+",
                "color": "#0071CE",
                "icon": "🛒",
                "example_questions": [
                    "What is Walmart's return policy for electronics?",
                    "What are the Black Friday 2024 TV deals?",
                    "How much does Walmart+ cost?",
                    "Calculate the discount on a $1299 TV at 62% off",
                ],
            },
            {
                "id": UserRole.CLIENT,
                "label": "Client / Vendor",
                "description": "Walmart suppliers — billing, onboarding, compliance",
                "color": "#FFC220",
                "icon": "🏭",
                "example_questions": [
                    "What are the NET-30 payment terms?",
                    "What documents do I need for supplier onboarding?",
                    "Show me the billing summary for November",
                    "What are the vendor compliance fines?",
                    "Check MCP server for supplier contact details",
                ],
            },
            {
                "id": UserRole.DEVELOPER,
                "label": "Developer",
                "description": "Internal engineers — API docs, code gen, DevOps",
                "color": "#007DC6",
                "icon": "💻",
                "example_questions": [
                    "How does OAuth 2.0 work for Walmart APIs?",
                    "Write code to call the Walmart Item API",
                    "What are the webhook retry policies?",
                    "Check if AIRPODS-PRO is in stock",
                    "Open a ticket for the payment service incident",
                ],
            },
        ]
    }


@app.get("/agent-map")
def get_agent_map():
    return {
        "architecture": {
            "entry": "User Message",
            "agents": [
                {
                    "name": "PlannerAgent",
                    "role": "Analyse user intent, produce structured plan",
                    "position": 1,
                },
                {
                    "name": "OrchestratorAgent",
                    "role": "Routing decision — selects specialist based on intent",
                    "position": 2,
                    "routes_to": ["RAGAgent", "ToolAgent", "CoderAgent", "MCPAgent"],
                },
                {
                    "name": "RAGAgent",
                    "role": (
                        "Cosine similarity search against PGVector (prod) or FAISS (demo), "
                        "then grounded answer synthesis"
                    ),
                    "trigger": "Factual Q&A, policy questions, document lookup",
                    "sql": "SELECT ... FROM walmart_documents WHERE persona=$1 ORDER BY embedding <=> $2 LIMIT 3",
                },
                {
                    "name": "ToolAgent",
                    "role": "Deterministic tool calls (pricing, inventory, billing)",
                    "trigger": "Calculations, structured data lookups",
                },
                {
                    "name": "CoderAgent",
                    "role": "Code generation and technical explanation",
                    "trigger": "Write/explain/debug code requests",
                },
                {
                    "name": "MCPAgent",
                    "role": "External system integration via MCP protocol",
                    "trigger": "CRM, ticketing, calendar, order management",
                },
            ],
            "vector_db": (
                "FAISS in-memory (DEMO_MODE=true)"
                if DEMO_MODE
                else "PGVector — PostgreSQL + pgvector extension (DEMO_MODE=false)"
            ),
            "vector_db_table":    "walmart_documents",
            "embedding_dim":      1536,
            "similarity_metric":  "cosine (<=> operator)",
            "persona_isolation":  "SQL-layer WHERE persona = $1",
            "llm": "gpt-4o-mini",
        }
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        response = run_pipeline(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
