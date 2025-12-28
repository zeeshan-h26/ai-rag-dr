import streamlit as st
from utils.api import ask_question


def render_chat():

    # ---------- CHAT STYLING ----------
    st.markdown("""
    <style>
    .chat-bubble-user {
        background-color:#d9e9ff;
        padding:12px;
        border-radius:12px;
        margin:6px 0;
        color:#000;
        font-size:16px;
    }

    .chat-bubble-assistant {
        background-color:#ffffff;
        padding:12px;
        border-radius:12px;
        margin:6px 0;
        border:1px solid #d6e4ff;
        color:#000;
        font-size:16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- SESSION MEMORY ----------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---------- RENDER CHAT ----------
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-bubble-user'>🙋‍♂️ {msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-bubble-assistant'>🤖 {msg['content']}</div>",
                    unsafe_allow_html=True
                )

    # ---------- USER INPUT ----------
    user_input = st.chat_input("Type your question....")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Assistant is thinking..."):
            response = ask_question(user_input)

        if response.status_code == 200:
            data = response.json()
            answer = data["response"]

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        else:
            st.error(f"Error: {response.text}")
