# import streamlit as st
# st.title("Step 1 — just a title")

#--------------------------------
# import streamlit as st
# st.title("Step 2 — add chat input")

# if prompt := st.chat_input("Type here..."):
#     st.write(f"You typed: {prompt}")

#--------------------------------
import streamlit as st
st.title("Step 3 — add memory")
print(st.session_state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show what's in memory
st.write("Messages in memory:", st.session_state.messages)

if prompt := st.chat_input("user"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.write("Messages after adding yours:", st.session_state.messages)

