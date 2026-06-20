"""Streamlit frontend: a chat interface plus an admin dashboard.

  * Pick one of the 5 onboarded users in the sidebar (this sets the API key sent
    to the backend).
  * Chat tab: ask questions, see answers + the sources the answer cited.
  * Admin tab: only shown for the admin user (user 1) — live metrics, per-user
    usage, and per-user cost pulled from the backend.

Run (after the backend is up):
    streamlit run frontend/app.py
"""
import os

import streamlit as st

from api_client import BackendClient

DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")

# The 5 onboarded users. (Mirrors common/users.py — kept here so the frontend
# has no backend import dependency.)
USERS = {
    "Aisha (Admin)": "key-aisha-001",
    "Ben": "key-ben-002",
    "Carlos": "key-carlos-003",
    "Diya": "key-diya-004",
    "Erik": "key-erik-005",
}

st.set_page_config(page_title="Meridian Knowledge Assistant", page_icon="📚")

# ---- sidebar: who am I + backend URL ----
st.sidebar.title("Meridian RAG")
who = st.sidebar.selectbox("Sign in as", list(USERS.keys()))
base_url = st.sidebar.text_input("Backend URL", DEFAULT_BACKEND)
api_key = USERS[who]
is_admin = who.endswith("(Admin)")
client = BackendClient(api_key=api_key, base_url=base_url)
st.sidebar.caption(f"API key: `{api_key}`")

# ---- tabs ----
tabs = ["💬 Chat", "🛠️ Admin"] if is_admin else ["💬 Chat"]
selected = st.tabs(tabs)

# ===== Chat tab =====
with selected[0]:
    st.header("Ask the knowledge base")
    st.caption("Try: *How many days of annual leave do I get?* or "
               "*How do I reset my password?*")

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("Sources"):
                    for cit in msg["citations"]:
                        st.markdown(f"**{cit['title']}** — `{cit['source']}`\n\n"
                                    f"> {cit['snippet']}")

    question = st.chat_input("Type your question...")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = client.ask(question)
            answer = resp.get("answer", "(no answer)")
            st.markdown(answer)
            meta = (f"model: `{resp.get('model')}` · cached: {resp.get('cached')} "
                    f"· blocked: {resp.get('blocked')}")
            st.caption(meta)
            if resp.get("guardrail_notes"):
                st.warning(" / ".join(resp["guardrail_notes"]))
            st.session_state.history.append({
                "role": "assistant", "content": answer,
                "citations": resp.get("citations", [])})

# ===== Admin tab =====
if is_admin:
    with selected[1]:
        st.header("Admin dashboard")
        if st.button("Refresh"):
            st.rerun()

        data = client.admin_metrics()
        m = data.get("metrics", {})
        alerts = data.get("alerts", [])

        if alerts:
            for a in alerts:
                st.error(f"⚠️ {a}")

        st.subheader("Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total requests", m.get("total_requests", 0))
        c2.metric("p95 latency (ms)", m.get("latency_ms", {}).get("p95", 0))
        c3.metric("Cache hit rate", f"{m.get('cache_hit_rate', 0):.0%}")
        c4, c5, c6 = st.columns(3)
        c4.metric("Block rate", f"{m.get('block_rate', 0):.0%}")
        c5.metric("Avg relevance", m.get("avg_relevance") or "—")
        c6.metric("Avg faithfulness", m.get("avg_faithfulness") or "—")

        st.subheader("Usage by user")
        st.dataframe(client.admin_usage().get("usage", []), use_container_width=True)

        st.subheader("Cost by user")
        cost = client.admin_cost()
        st.dataframe(cost.get("per_user", []), use_container_width=True)
        st.metric("Total cost (USD)", f"${cost.get('total_cost_usd', 0):.4f}")
