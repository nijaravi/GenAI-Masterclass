"""
app.py — A Streamlit chat app with multi-model support.

GenAI Decoded by Nij — Section 3: From Notebooks to Apps

Run:
    streamlit run app.py
"""

import streamlit as st
import json
from datetime import datetime
from openai import OpenAI

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="GenAI Chat — by Nij", page_icon="🤖", layout="wide")

# ============================================================
# MODELS & PRICING
# ============================================================
MODELS = {
    "gpt-4o-mini": {"label": "GPT-4o Mini (fast, cheap)", "input": 0.15, "output": 0.60},
    "gpt-4o": {"label": "GPT-4o (powerful)", "input": 2.50, "output": 10.00},
}

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Settings")
    
    model = st.selectbox("Model", options=list(MODELS.keys()),
                         format_func=lambda m: MODELS[m]["label"])
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    system_prompt = st.text_area("System Prompt",
        value="You are a helpful assistant. Be concise and direct.", height=100)
    max_tokens = st.slider("Max response tokens", 100, 4000, 1000, 100)
    
    st.divider()
    st.subheader("📊 Session Stats")
    st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    st.metric("Estimated Cost", f"${st.session_state.total_cost:.4f}")
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.total_cost = 0.0
            st.rerun()
    with col2:
        if st.session_state.messages:
            st.download_button("💾 Export", use_container_width=True,
                data=json.dumps({"model": model, "system_prompt": system_prompt,
                    "messages": st.session_state.messages,
                    "stats": {"tokens": st.session_state.total_tokens,
                              "cost": st.session_state.total_cost}}, indent=2),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json")

# ============================================================
# MAIN CHAT
# ============================================================
st.title("🤖 GenAI Chat")
st.caption(f"Model: {MODELS[model]['label']} | Temp: {temperature}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    api_messages = []
    if system_prompt.strip():
        api_messages.append({"role": "system", "content": system_prompt})
    api_messages.extend(st.session_state.messages)
    
    with st.chat_message("assistant"):
        try:
            client = OpenAI()
            stream = client.chat.completions.create(
                model=model, messages=api_messages,
                temperature=temperature, max_tokens=max_tokens, stream=True)
            
            response_text = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Estimate tokens (~4 chars per token)
            est_in = sum(len(m["content"]) for m in api_messages) // 4
            est_out = len(response_text) // 4
            est_cost = (est_in * MODELS[model]["input"] + est_out * MODELS[model]["output"]) / 1e6
            st.session_state.total_tokens += est_in + est_out
            st.session_state.total_cost += est_cost
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            st.info("Check your OPENAI_API_KEY environment variable.")
