import streamlit as st
from components.upload import render_uploader
from components.history_download import render_history_download
from components.chatUI import render_chat

st.set_page_config(
    page_title="PatientMate",
    layout="wide"
)

# ===== GLOBAL STYLE =====

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #EBF4FF, #F8FBFF);
}

/* Remove Streamlit top white bar completely */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Remove extra padding created by Streamlit */
div[data-testid="stAppViewContainer"] > .main {
    padding-top: 0rem !important;
}

.center-title {
    text-align:center;
    color:#0E4D92;
    font-size:45px;
    font-weight:900;
}

</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("<h1 class='center-title'>🩺 PatientMate</h1>", unsafe_allow_html=True)


# ===== SIDEBAR =====
with st.sidebar:
    st.title("📂 Upload Medical Documents")
    render_uploader()

    st.markdown("---")
    st.title("📥 Download History")
    render_history_download()


# ===== MAIN — CHAT ONLY =====
st.markdown("### 💬 Chat with your Assistant")
render_chat()
