import streamlit as st
from chat import chat_response, pdf_chat_response, image_analysis_response

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Smart Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

# ------------------ SESSION ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("🕘 Chat History")

    if not st.session_state.chat_history:
        st.caption("No conversations yet")
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"• {msg['content'][:40]}")

# ------------------ MAIN ------------------
st.title("🏥 Smart Healthcare Assistant")
st.caption("Educational use only. Not a doctor.")

mode = st.radio(
    "Choose Mode",
    ["💬 Chat", "📄 PDF", "🖼 Image"],
    horizontal=True
)

# ================= CHAT =================
if mode == "💬 Chat":
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask your health question:",
            placeholder="e.g. What are symptoms of asthma?"
        )
        submit = st.form_submit_button("Ask")

    if submit and user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        with st.spinner("Analyzing..."):
            response = chat_response(user_input)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

    # Display chat
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:**\n{msg['content']}")

# ================= PDF =================
elif mode == "📄 PDF":
    uploaded_pdf = st.file_uploader(
        "Upload a medical PDF",
        type=["pdf"]
    )

    pdf_question = st.text_input("Ask question from PDF")

    if uploaded_pdf and pdf_question:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_pdf.read())

        with st.spinner("Reading document..."):
            answer = pdf_chat_response("temp.pdf", pdf_question)

        st.markdown(answer)

# ================= IMAGE =================
elif mode == "🖼 Image":
    uploaded_image = st.file_uploader(
        "Upload medical image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image:
        st.image(uploaded_image)
        st.warning("⚠️ Handwritten prescriptions may not be readable")

        result = image_analysis_response(uploaded_image)
        st.markdown(result)