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
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ======================================================
# LOAD VECTOR STORE (SAFE FALLBACK)
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
    retriever = None  # Cloud-safe fallback

# ======================================================
# LOAD LLM
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
# MEDICAL SCOPE CHECK
# ======================================================
MEDICAL_KEYWORDS = [
    "symptom", "disease", "fever", "pain", "infection", "asthma",
    "diabetes", "cancer", "covid", "health", "treatment", "medicine",
    "injury", "blood", "pressure", "mental", "depression", "anxiety",
    "cold", "flu", "headache", "vomiting", "diarrhea"
]

def is_medical_query(query: str) -> bool:
    query = query.lower()
    return any(word in query for word in MEDICAL_KEYWORDS)

# ======================================================
# AUTO-DETECT LONG ANSWER NEED
# ======================================================
def wants_long_answer(query: str) -> bool:
    long_keywords = [
        "explain", "detail", "detailed", "how", "why",
        "causes", "effects", "complications",
        "treatment", "management", "mechanism"
    ]
    query = query.lower()
    return any(word in query for word in long_keywords)

# ======================================================
# FORMAT RESPONSE INTO CLEAN POINTS
# ======================================================
def format_points(text: str, max_points: int) -> str:
    # Normalize text
    text = text.replace("•", "\n").replace("-", "\n")
    chunks = re.split(r"\n|\d+\.", text)

    points = [
        chunk.strip()
        for chunk in chunks
        if len(chunk.strip()) > 20
    ]

    points = points[:max_points]

    formatted = ""
    for i, point in enumerate(points, 1):
        formatted += f"{i}. {point}\n"

    return formatted.strip()

# ======================================================
# MAIN CHAT FUNCTION
# ======================================================
def chat_response(user_query: str) -> str:
    """
    Main chat response (Chat mode)
    """

    # ❌ Non-medical queries
    if not is_medical_query(user_query):
        return (
            "⚠️ **This assistant is designed only for medical and health-related questions.**\n\n"
            "If you have a health concern, symptom, or disease-related question, feel free to ask.\n\n"
            "For non-medical topics, please use a general-purpose assistant."
        )

    long_mode = wants_long_answer(user_query)
    max_points = 15 if long_mode else 6

    # ==================================================
    # TRY RAG FIRST (if vector store exists)
    # ==================================================
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

    # ==================================================
    # FALLBACK: GENERAL MEDICAL LLM
    # ==================================================
    general_prompt = f"""
You are a medical information assistant.

Rules:
- Educational information only
- No diagnosis
- No prescriptions
- Answer ONLY in numbered points
- Be clear and factual
- End with a doctor disclaimer

Number of points allowed: {max_points}

Question:
{user_query}
"""

    response = llm.invoke(general_prompt).content

    return format_points(response, max_points)

# ======================================================
# PDF RAG (USER UPLOAD)
# ======================================================
def pdf_chat_response(pdf_path: str, question: str) -> str:
    """
    Answer questions ONLY from uploaded PDF
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
        + format_points(answer, max_points=12)
        + "\n\n⚠️ Educational use only. Consult a healthcare professional."
    )

# ======================================================
# IMAGE SAFE RESPONSE
# ======================================================
def image_safe_response() -> str:
    return (
        "🖼️ **Image received**\n\n"
        "I cannot diagnose medical conditions from images.\n\n"
        "However, I can help by:\n"
        "1. Describing visible features\n"
        "2. Explaining general medical possibilities\n"
        "3. Suggesting when to consult a doctor\n\n"
        "⚠️ Please consult a qualified healthcare professional."
    )
