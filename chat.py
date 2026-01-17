import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ======================================================
# CONFIG
# ======================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store"
LLM_MODEL = "llama-3.1-8b-instant"

# ======================================================
# EMBEDDINGS
# ======================================================
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

# ======================================================
# VECTOR STORE (SAFE FALLBACK FOR CLOUD)
# ======================================================
retriever = None
try:
    db = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
except Exception:
    retriever = None

# ======================================================
# LLM
# ======================================================
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0.3
)

qa_chain = None
if retriever:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

# ======================================================
# MEDICAL FILTER
# ======================================================
MEDICAL_KEYWORDS = [
    "symptom", "disease", "fever", "pain", "infection", "asthma",
    "diabetes", "cancer", "covid", "health", "treatment", "medicine",
    "cold", "flu", "headache", "vomiting", "diarrhea"
]

def is_medical_query(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in MEDICAL_KEYWORDS)

# ======================================================
# LONG ANSWER DETECTION
# ======================================================
def wants_long_answer(query: str) -> bool:
    return any(
        word in query.lower()
        for word in ["explain", "detail", "why", "how", "causes", "effects"]
    )

# ======================================================
# 🔥 BULLETPROOF POINT FORMATTER (FIX)
# ======================================================
def format_points(text: str, max_points: int) -> str:
    """
    Converts ANY model output into clean vertical numbered points
    """

    # Remove markdown & inline numbering
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\d+\.", ".", text)

    # Split by sentences
    sentences = re.split(r"[.\n]+", text)

    points = [
        s.strip()
        for s in sentences
        if len(s.strip()) > 30
    ]

    points = points[:max_points]

    formatted = ""
    for i, point in enumerate(points, 1):
        formatted += f"{i}. {point}\n"

    return formatted.strip()

# ======================================================
# CHAT RESPONSE
# ======================================================
def chat_response(user_query: str) -> str:

    if not is_medical_query(user_query):
        return (
            "⚠️ **This assistant is designed only for medical and health-related questions.**\n\n"
            "Please ask about symptoms, diseases, or health concerns."
        )

    long_mode = wants_long_answer(user_query)
    max_points = 12 if long_mode else 6

    # -------- RAG FIRST --------
    if qa_chain:
        try:
            rag_answer = qa_chain.run(user_query)
            if rag_answer and len(rag_answer.strip()) > 40:
                return (
                    "**Based on verified medical sources:**\n\n"
                    + format_points(rag_answer, max_points)
                )
        except Exception:
            pass

    # -------- FALLBACK LLM --------
    prompt = f"""
You are a medical information assistant.

Rules:
- Educational only
- No diagnosis
- No prescriptions
- Write SHORT, CLEAR sentences
- EACH point must be ONE idea
- DO NOT write paragraphs

Question:
{user_query}
"""

    response = llm.invoke(prompt).content

    return (
        format_points(response, max_points)
        + "\n\n⚠️ Educational use only. Consult a healthcare professional."
    )

# ======================================================
# PDF RAG
# ======================================================
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
        + format_points(answer, 12)
        + "\n\n⚠️ Educational use only."
    )

# ======================================================
# IMAGE SAFE RESPONSE
# ======================================================
def image_safe_response() -> str:
    return (
        "🖼️ **Image received**\n\n"
        "I cannot diagnose medical conditions from images.\n\n"
        "I can describe visible features and suggest when to consult a doctor.\n\n"
        "⚠️ Consult a healthcare professional for diagnosis."
    )
