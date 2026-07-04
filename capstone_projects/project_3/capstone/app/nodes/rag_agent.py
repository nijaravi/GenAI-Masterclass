"""
Section 4.6: dedicated RAG Agent, kept separate from the general tool layer
because retrieval has its own lifecycle. Section 8.2: embed the query with
the ingestion-time embedding model, retrieve, then generate a retrieval-only
answer with citations.

Namespace selection is role-based (Section 8.1 / 10): customers only ever
see 'customer-product', developers only ever see 'developer-docs'. This is
the same role-based scoping rule enforced elsewhere, just applied at the
retrieval layer instead of the routing layer - defense in depth, matching
Section 10's point that routing restrictions are a UX control, not the only
security boundary.
"""
from __future__ import annotations

from app.llm import get_chat_model
from app.rag.vector_store import get_vector_store
from app.state import OrchestratorState

_NAMESPACE_BY_ROLE = {
    "customer": "customer-product",
    "developer": "developer-docs",
    # clients don't have a RAG namespace in this design - their questions
    # are account/status lookups that route to MCP instead (Section 7.2).
}


async def rag_agent_node(state: OrchestratorState) -> dict:
    role = state["user_role"]
    namespace = _NAMESPACE_BY_ROLE.get(role, "customer-product")

    store = get_vector_store()
    chunks = store.query(namespace, state["user_query"], top_k=4)

    model = get_chat_model()
    context_block = "\n".join(f"- ({c['source']}) {c['text']}" for c in chunks)
    prompt = (
        "Answer using only the retrieved context below. Cite the source "
        f"file for each claim.\n\nQuestion: {state['user_query']}\n\n"
        f"Retrieved context:\n{context_block}"
    )
    synthesis = model.invoke(prompt)

    if synthesis.startswith("(demo synthesis)"):
        if chunks:
            bullets = "\n".join(f"- {c['text']} [{c['source']}]" for c in chunks)
            synthesis = f"Based on the product/reference data:\n{bullets}"
        else:
            synthesis = "No matching passages were found in the retrieval index for this query."

    return {
        "retrieved_context": chunks,
        "agent_output": synthesis,
        "collected_outputs": state.get("collected_outputs", []) + [f"[Docs/Product lookup]\n{synthesis}"],
        "node_path": state.get("node_path", []) + ["rag_agent"],
    }
