"""
rag_agent.py — Retrieval-Augmented Generation Agent.

Responsibility:
  1. Embed the user query
  2. Retrieve top-k documents from the vector store
     - DEMO_MODE=true  → SimpleVectorStore (in-memory FAISS)
     - DEMO_MODE=false → PGVectorStore (PostgreSQL + pgvector)
  3. Synthesise a grounded answer using the retrieved context

PGVector query executed (simplified):
    SELECT id, persona, category, content,
           1 - (embedding <=> $query_vec) AS score
    FROM   walmart_documents
    WHERE  persona = $persona
    ORDER  BY embedding <=> $query_vec
    LIMIT  3;

The <=> operator is pgvector's cosine distance operator.
Persona filtering happens at the SQL layer — not in application code —
so cross-persona data leakage is impossible regardless of query phrasing.
"""
from typing import List
from backend.config import AgentStep, UserRole, DEMO_MODE
from backend.utils.vector_store import get_vector_store
from backend.utils.llm_client import run_rag_agent


class RAGAgent:
    """
    RAGAgent retrieves persona-scoped documents from the vector store
    and synthesises a grounded answer.
    """

    name = "RAGAgent"

    def run(self, user_message: str, user_role: UserRole) -> AgentStep:
        persona = user_role.value

        # ── Retrieval ──────────────────────────────────────────────────────
        store = get_vector_store()
        retrieved_docs = store.query(
            query_text=user_message,
            top_k=3,
            filter_persona=persona,
        )

        source_texts: List[str] = [doc["text"][:80] + "..." for doc in retrieved_docs]

        # ── Generation ────────────────────────────────────────────────────
        answer = run_rag_agent(user_message, retrieved_docs, persona)

        vector_db_label = (
            "FAISS in-memory (DEMO_MODE=true)"
            if DEMO_MODE
            else "PGVector — PostgreSQL + pgvector extension (DEMO_MODE=false)"
        )

        return AgentStep(
            agent=self.name,
            input=user_message,
            output=answer,
            metadata={
                "retrieved_doc_count": len(retrieved_docs),
                "sources": source_texts,
                "persona_filter": persona,
                "vector_db": vector_db_label,
                "similarity_metric": "cosine (<=> operator)",
            },
        )
