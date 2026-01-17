import os
import re
from dotenv import load_dotenv
load_dotenv()

# =========================
# LANGCHAIN IMPORTS
# =========================
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =========================
# CONFIG
# =========================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store"
LLM_MODEL = "llama-3.1-8b-instant"  # Groq supported

# =========================
# LOAD EMBEDDINGS
# =========================
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL
)

# =========================
# LOAD LLM (CLOUD SAFE)
# =========================
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0.2
)

# =========================
# VECTOR DB (LAZY + SAFE)
# =========================
_db = None

def load_vector_db():
    """
    Load FAISS only if it exists (local).
    Cloud-safe: returns None if missing.
    """
    global _db

    if _db is not None:
        return _db

    if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        return None

    try:
        _db = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return _db
    except Exception as e:
        print("FAISS not available:", e)
        return None

# =========================
# MEDICAL SCOPE CHECK
# =========================
MEDICAL_KEYWORDS = [
    "symptom", "disease", "fever", "pain", "infection", "asthma",
    "diabetes", "cancer", "covid", "health", "treatment", "medicine",
    "injury", "blood", "pressure", "mental", "depression", "anxiety",
    "headache", "heart", "lung", "kidney", "stomach"
]

def is_medical_query(query: str) -> bool:
    query = query.lower()
    return any(word in query for word in MEDICAL_KEYWORDS)

# =========================
# FORMAT RESPONSE INTO POINTS
# =========================
def format_points(text: str) -> str:
    lines = re.split(r"\n|\d+\.", text)
    points = [line.strip("-• ") for line in lines if len(line.strip()) > 10]

    formatted = ""
    for i, point in enumerate(points, 1):
        formatted += f"{i}. {point}\n"

    return formatted.strip()

# =========================
# MAIN CHAT FUNCTION
# =========================
def chat_response(user_query: str) -> str:
    """
    Used by both app.py (local) and streamlit_app.py (cloud)
    """

    # ❌ Non-medical queries
    if not is_medical_query(user_query):
        return (
            "⚠️ **This assistant is designed only for medical and health-related questions.**\n\n"
            "Please ask about symptoms, diseases, medicines, or healthcare topics.\n\n"
            "For non-medical questions, use a general-purpose assistant."
        )

    # =========================
    # TRY RAG (LOCAL ONLY)
    # =========================
    db = load_vector_db()

    if db:
        try:
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_query)

            if docs:
                context = "\n\n".join([d.page_content for d in docs])
                prompt = f"""
You are a medical information assistant.
Rules:
- Educational information only
- No diagnosis
- No prescriptions
- Respond in numbered points
- End with doctor disclaimer

Context:
{context}

Question:
{user_query}
"""
                response = llm.invoke(prompt).content
                return (
                    "**Based on verified medical sources:**\n\n"
                    + format_points(response)
                )
        except Exception as e:
            print("RAG failed:", e)

    # =========================
    # FALLBACK (CLOUD SAFE)
    # =========================
    general_prompt = f"""
You are a medical information assistant.
Rules:
- Educational information only
- No diagnosis
- No prescriptions
- Always respond in numbered points
- End with a doctor disclaimer

Question:
{user_query}
"""

    response = llm.invoke(general_prompt).content
    return format_points(response)

# =========================
# PDF UPLOAD RAG
# =========================
def pdf_chat_response(pdf_path: str, question: str) -> str:
    """
    Answer ONLY from uploaded PDF
    """

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    temp_db = FAISS.from_documents(chunks, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=temp_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

    answer = qa.run(question)

    return (
        "📄 **Answer based on uploaded medical document:**\n\n"
        + format_points(answer) +
        "\n\n⚠️ This information is for educational purposes only."
    )

# =========================
# IMAGE SAFE RESPONSE
# =========================
def image_safe_response() -> str:
    return (
        "🖼️ **Image received**\n\n"
        "I cannot diagnose medical conditions from images.\n\n"
        "I can help by:\n"
        "1. Describing visible features (color, shape, pattern)\n"
        "2. Explaining general medical possibilities\n"
        "3. Suggesting when to consult a doctor\n\n"
        "⚠️ Please consult a qualified healthcare professional for diagnosis."
    )
