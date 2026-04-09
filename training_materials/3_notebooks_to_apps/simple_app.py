import streamlit as st
from openai import OpenAI

client = OpenAI()
st.title("My GenAI App")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages
        )
        answer = response.choices[0].message.content
        st.write(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})