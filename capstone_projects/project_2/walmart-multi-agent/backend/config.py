"""
config.py — Shared configuration and Pydantic models
"""
import os
from enum import Enum
from typing import Optional, List, Any, Dict
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ─── Settings ────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# PGVector — primary vector store (production)
PGVECTOR_CONNECTION_STRING = os.getenv(
    "PGVECTOR_CONNECTION_STRING",
    "postgresql://postgres:postgres@localhost:5432/walmart_ai",
)

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

LLM_MODEL = "gpt-4o-mini"

# ─── Enums ───────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    CUSTOMER = "customer"
    CLIENT = "client"
    DEVELOPER = "developer"

class RouteDecision(str, Enum):
    RAG = "rag"
    TOOL_CALL = "tool_call"
    CODER = "coder"
    MCP = "mcp"

# ─── Request / Response Models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    user_role: UserRole
    session_id: Optional[str] = "default"
    conversation_history: Optional[List[Dict[str, str]]] = []

class AgentStep(BaseModel):
    agent: str
    input: str
    output: str
    metadata: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    answer: str
    route_taken: RouteDecision
    steps: List[AgentStep]
    user_role: UserRole
    sources: Optional[List[str]] = []
