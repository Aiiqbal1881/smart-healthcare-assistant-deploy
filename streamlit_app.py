import streamlit as st
from chat import chat_response, pdf_chat_response, image_safe_response

st.set_page_config(page_title="Smart Healthcare Assistant", layout="wide")

st.title("🏥 Smart Healthcare Assistant")
st.caption("AI-based medical guidance (non-prescriptive)")

# ------------------ MODE ------------------
mode = st.radio(
    "Choose interaction mode:",
    ["💬 Chat", "📄 PDF", "🖼 Image"],
    horizontal=True
)

# ================== CHAT ==================
if mode == "💬 Chat":
    user_input = st.text_input("Ask your health question:")

    if st.button("Ask") and user_input:
        with st.spinner("Thinking..."):
            response = chat_response(user_input)
        st.success(response)

# ================== PDF ==================
elif mode == "📄 PDF":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    question = st.text_input("Ask question from PDF:")

    if uploaded_file and question:
        with st.spinner("Analyzing PDF..."):
            answer = pdf_chat_response(uploaded_file, question)
        st.success(answer)

# ================== IMAGE ==================
elif mode == "🖼 Image":
    uploaded_image = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        st.image(uploaded_image)
        st.info(image_safe_response())