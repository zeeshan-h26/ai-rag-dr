import streamlit as st

def render_history_download():
    # Check if chat history exists
    if "messages" in st.session_state and st.session_state.messages:
        
        # Format chat messages
        chat_text = "\n\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.messages
        ])

        st.download_button(
            label="📥 Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )
    else:
        st.info("No chat history available yet.")
