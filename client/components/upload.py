import streamlit as st
from utils.api import upload_pdfs_api


def render_uploader():

    # ---------- Card Styling ----------
    st.markdown("""
        <style>
        .upload-card {
            background: #ffffff;
            padding: 18px;
            border-radius: 14px;
            border: 2px dashed #6aa9ff;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)

    st.subheader("📄 Upload Medical Documents")

    uploaded_files = st.file_uploader(
        "Upload multiple medical PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("**Limit:** 20MB per file • PDF Only")

    # ---------- Helpful Info ----------
    with st.expander("📚 Documents you can upload"):
        st.markdown("""
        **Supported Documents**
        - 🧾 Prescription  
        - 🧪 Lab Reports  
        - 🏥 Discharge Summary  
        - 💊 Medicine / Treatment PDFs  
        """)

    # ---------- Show selected file names ----------
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected")
        for f in uploaded_files:
            st.write(f"• {f.name}")

    # ---------- Upload Button ----------
    if st.button("Upload"):
        if uploaded_files:
            with st.spinner("Uploading documents..."):
                response = upload_pdfs_api(uploaded_files)

            if response.status_code == 200:
                st.success("✅ Uploaded successfully")
            else:
                st.error(f"❌ Error: {response.text}")
        else:
            st.warning("⚠️ Please upload files first")

    st.markdown("</div>", unsafe_allow_html=True)
