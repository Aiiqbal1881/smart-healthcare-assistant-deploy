import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

# ---------------- CONFIG ----------------
LLM_MODEL = "llama-3.1-8b-instant"

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0.3
)

# ---------------- MEDICAL CHECK ----------------
MEDICAL_KEYWORDS = [
    "symptom", "disease", "fever", "pain", "infection", "asthma",
    "diabetes", "cancer", "covid", "health", "treatment", "medicine",
    "injury", "blood", "pressure", "mental", "depression", "anxiety"
]

def is_medical_query(query: str) -> bool:
    query = query.lower()
    return any(word in query for word in MEDICAL_KEYWORDS)

# ---------------- FORMAT POINTS ----------------
def force_points(text: str, max_points=10) -> str:
    lines = re.split(r"\n|\d+\.", text)
    clean = [l.strip("-• ") for l in lines if len(l.strip()) > 20]

    points = clean[:max_points]

    formatted = ""
    for i, p in enumerate(points, 1):
        formatted += f"{i}. {p}\n"

    formatted += "\n⚠️ Educational use only. Consult a healthcare professional."
    return formatted.strip()

# ---------------- CHAT RESPONSE ----------------
def chat_response(user_query: str) -> str:

    if not is_medical_query(user_query):
        return (
            "⚠️ This assistant answers **medical questions only**.\n\n"
            "Please ask a health-related question."
        )

    prompt = f"""
You are a medical information assistant.

Rules:
- Educational use only
- No diagnosis
- No prescriptions
- ALWAYS respond in numbered points
- Maximum 10 points
- Clear, simple language

Question:
{user_query}
"""

    response = llm.invoke(prompt).content
    return force_points(response)

# ---------------- PDF MODE ----------------
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def pdf_chat_response(pdf_path: str, question: str) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)

    answer = llm.invoke(
        f"Answer strictly from this document in numbered points:\n{question}"
    ).content

    return force_points(answer)

# ---------------- IMAGE MODE ----------------
def image_safe_response() -> str:
    return (
        "🖼️ Image received.\n\n"
        "I cannot diagnose from images.\n\n"
        "I can:\n"
        "1. Describe visible features\n"
        "2. Explain general medical info\n"
        "3. Suggest when to see a doctor\n\n"
        "⚠️ Educational use only."
    )