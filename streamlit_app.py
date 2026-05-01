import streamlit as st
from chat import chat_response, pdf_chat_response, image_safe_response

# ------------------ PAGE ------------------
st.set_page_config(
    page_title="Smart Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

# ------------------ STATE ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("🕘 Chat History")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.caption(msg["content"][:40])

    st.markdown("---")
    st.markdown("""
**Why RAG?**
- Reduces hallucination  
- Uses documents  

**Safety**
- No diagnosis  
- Educational only  
""")

# ------------------ MAIN ------------------
st.title("🏥 Smart Healthcare Assistant")

mode = st.radio(
    "Choose Mode:",
    ["💬 Chat", "📄 PDF", "🖼 Image"],
    horizontal=True
)

# ================= CHAT =================
if mode == "💬 Chat":

    user_input = st.text_input("Ask your question:")

    if st.button("Ask") and user_input:

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        with st.spinner("Thinking..."):
            response = chat_response(user_input)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

    # DISPLAY
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:**\n{msg['content']}")
            st.info("Educational use only")

# ================= PDF =================
elif mode == "📄 PDF":

    file = st.file_uploader("Upload PDF", type=["pdf"])
    question = st.text_input("Ask from PDF")

    if file and question:
        with open("temp.pdf", "wb") as f:
            f.write(file.read())

        with st.spinner("Reading PDF..."):
            answer = pdf_chat_response("temp.pdf", question)

        st.markdown(answer)

# ================= IMAGE =================
elif mode == "🖼 Image":

    image = st.file_uploader("Upload Image", type=["jpg", "png"])

    if image:
        st.image(image)
        st.info(image_safe_response())