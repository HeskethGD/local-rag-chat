import streamlit as st
import requests

BASE_URL = "http://localhost:8000"
# URL_EXT = "/chat"
URL_EXT = "/rag"
URL = f"{BASE_URL}{URL_EXT}"

def chat_response(messages):
    with requests.post(URL, json={"messages": messages}, stream=True) as r:
        for chunk in r.iter_content(None, decode_unicode=True):
            if chunk:
                yield chunk

st.title('Ask the data a question')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        response = st.write_stream(chat_response(messages))
    st.session_state.messages.append({"role": "assistant", "content": response})