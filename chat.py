import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =========================
# CONFIG
# =========================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store"
LLM_MODEL = "llama-3.1-8b-instant"

# =========================
# LOAD LLM
# =========================
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0.2
)

# =========================
# LOAD EMBEDDINGS
# =========================
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL
)

# =========================
# TRY LOAD VECTOR STORE (SAFE)
# =========================
qa_chain = None
try:
    db = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
except Exception:
    qa_chain = None  # cloud-safe fallback

# =========================
# MEDICAL SCOPE FILTER
# =========================
MEDICAL_KEYWORDS = [
    "symptom", "disease", "fever", "pain", "infection", "asthma",
    "diabetes", "cancer", "covid", "health", "treatment", "medicine",
    "injury", "blood", "pressure", "mental", "depression", "anxiety",
    "headache", "cold", "flu", "memory", "period"
]

def is_medical_query(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in MEDICAL_KEYWORDS)

# =========================
# UNIVERSAL POINT FORMATTER (FIX)
# =========================
def format_points(text: str) -> str:
    """
    Converts ANY answer into clean numbered points.
    """

    # Remove markdown, bullets, numbering
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\n+", "\n", text)

    # Split into sentences
    sentences = re.split(r"\.\s+|\n", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    formatted = ""
    for i, sentence in enumerate(sentences, 1):
        formatted += f"{i}. {sentence}\n"

    return formatted.strip()

# =========================
# MAIN CHAT FUNCTION
# =========================
def chat_response(user_query: str) -> str:

    if not is_medical_query(user_query):
        return (
            "⚠️ **This assistant is designed only for medical and health-related questions.**\n\n"
            "If you have a health concern, symptom, or disease-related question, feel free to ask.\n\n"
            "For non-medical topics, please use a general-purpose assistant."
        )

    # ---------- RAG PATH ----------
    if qa_chain:
        try:
            answer = qa_chain.run(user_query)
            if answer and len(answer.strip()) > 50:
                return (
                    "**Based on verified medical sources:**\n\n"
                    + format_points(answer)
                )
        except Exception:
            pass

    # ---------- FALLBACK LLM ----------
    prompt = f"""
You are a medical information assistant.

Rules:
- Educational information only
- No diagnosis
- No prescriptions
- Always answer in clear bullet-like statements
- End with a doctor disclaimer

Question:
{user_query}
"""

    response = llm.invoke(prompt).content

    return (
        "**Based on general medical knowledge:**\n\n"
        + format_points(response)
        + "\n\n⚠️ Educational use only. Consult a qualified healthcare professional."
    )

# =========================
# PDF CHAT (UPLOAD)
# =========================
def pdf_chat_response(pdf_path: str, question: str) -> str:

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
        + format_points(answer)
        + "\n\n⚠️ Educational use only."
    )

# =========================
# IMAGE SAFE RESPONSE
# =========================
def image_safe_response() -> str:
    return (
        "🖼️ **Image received**\n\n"
        "I cannot diagnose medical conditions from images.\n\n"
        "I can help by:\n"
        "1. Describing visible features\n"
        "2. Explaining general medical possibilities\n"
        "3. Advising when to consult a doctor\n\n"
        "⚠️ Educational use only."
    )
