import streamlit as st
from chat import chat_response, pdf_chat_response, image_safe_response

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Smart Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.chat-user {
    background-color: #1f2933;
    padding: 14px;
    border-radius: 10px;
    margin-bottom: 8px;
}
.chat-assistant {
    background-color: #0f3d2e;
    padding: 14px;
    border-radius: 10px;
    margin-bottom: 18px;
}
.sidebar-title {
    font-size: 18px;
    font-weight: bold;
}
.disclaimer {
    background-color: #102a43;
    padding: 10px;
    border-radius: 8px;
    color: #9fb3c8;
    font-size: 14px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SESSION STATE ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_index" not in st.session_state:
    st.session_state.active_index = None

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🕘 Chat History</div>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.caption("No conversations yet")
    else:
        for i in range(0, len(st.session_state.chat_history), 2):
            q = st.session_state.chat_history[i]["content"]
            if st.button(q[:40] + "...", key=f"history_{i}"):
                st.session_state.active_index = i

    st.markdown("---")
    st.markdown("""
**Why RAG in Healthcare?**
- Reduces hallucinations  
- Uses verified medical documents  

**Safety**
- Non-prescriptive  
- Emergency detection  
- Doctor disclaimer  
""")

# ------------------ MAIN UI ------------------
st.title("🏥 Smart Healthcare Assistant")
st.caption("Informational use only. Not a replacement for doctors.")

# ------------------ MODE SELECTOR ------------------
mode = st.radio(
    "Choose interaction mode:",
    ["💬 Chat", "📄 PDF", "🖼 Image"],
    horizontal=True
)

# ================== CHAT MODE ==================
if mode == "💬 Chat":
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask your health question:",
            placeholder="e.g. What are symptoms of asthma?"
        )
        submitted = st.form_submit_button("Ask")

    if submitted and user_input.strip():
        st.session_state.active_index = None

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        with st.spinner("Analyzing medical information..."):
            response = chat_response(user_input)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

# ================== PDF MODE ==================
elif mode == "📄 PDF":
    uploaded_pdf = st.file_uploader(
        "Upload a medical PDF (reports, WHO guidelines, etc.)",
        type=["pdf"]
    )

    pdf_question = st.text_input(
        "Ask a question from this PDF:"
    )

    if uploaded_pdf and pdf_question.strip():
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_pdf.read())

        with st.spinner("Reading uploaded document..."):
            pdf_answer = pdf_chat_response(
                "temp_uploaded.pdf",
                pdf_question
            )

        st.markdown(pdf_answer, unsafe_allow_html=True)

# ================== IMAGE MODE ==================
elif mode == "🖼 Image":
    uploaded_image = st.file_uploader(
        "Upload a medical-related image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image")
        st.info(image_safe_response())

# ------------------ CHAT RENDER FUNCTION ------------------
def render_chat(upto=None):
    history = st.session_state.chat_history
    if upto is not None:
        history = history[:upto + 2]

    for msg in history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
            <b>🧑 You</b><br>{msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-assistant">
            <b>🤖 Assistant</b><br>
            {msg["content"]}
            <div class="disclaimer">
            ℹ️ Educational use only. Consult a qualified healthcare professional.
            </div>
            </div>
            """, unsafe_allow_html=True)

    # Auto-scroll to bottom (ChatGPT-like)
    st.markdown("<div id='bottom'></div>", unsafe_allow_html=True)
    st.markdown("""
    <script>
    document.getElementById("bottom").scrollIntoView({behavior: "smooth"});
    </script>
    """, unsafe_allow_html=True)

# ------------------ DISPLAY CHAT (CHAT MODE ONLY) ------------------
if mode == "💬 Chat":
    render_chat(st.session_state.active_index)
