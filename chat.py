import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =========================
# CONFIG
# =========================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store"
LLM_MODEL = "llama-3.1-8b-instant"

# =========================
# EMBEDDINGS
# =========================
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL
)

# =========================
# VECTOR STORE (OPTIONAL)
# =========================
retriever = None
if os.path.exists(VECTOR_STORE_PATH):
    try:
        db = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})
    except Exception:
        retriever = None

# =========================
# LLM
# =========================
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0.2
)

qa_chain = None
if retriever:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

# =========================
# MEDICAL FILTER
# =========================
MEDICAL_KEYWORDS = [
    "symptom","disease","fever","pain","infection","asthma","diabetes",
    "cancer","covid","health","treatment","medicine","blood","pressure",
    "mental","depression","anxiety"
]

def is_medical_query(q: str) -> bool:
    q = q.lower()
    return any(word in q for word in MEDICAL_KEYWORDS)

# =========================
# 🔥 HARD FORMAT POINTS (FINAL FIX)
# =========================
def format_points(text: str, max_points: int = 6) -> str:
    """
    Forces each numbered point onto a new line.
    Prevents inline point collapse.
    """

    # Remove markdown
    text = re.sub(r"\*\*", "", text)

    # Force newline before numbered points
    text = re.sub(r"(?<!\n)(\d+\.)", r"\n\1", text)

    # Split into points
    parts = re.findall(r"\d+\.\s+[^.\n]+(?:\.[^.\n]+)?", text)

    if not parts:
        sentences = re.split(r"\.\s+", text)
        parts = sentences

    clean = []
    for p in parts:
        p = p.strip()
        if len(p) > 25:
            clean.append(p)

    clean = clean[:max_points]

    output = ""
    for i, point in enumerate(clean, 1):
        point = re.sub(r"^\d+\.\s*", "", point)
        output += f"{i}. {point.strip()}\n\n"

    return output.strip()

# =========================
# CHAT RESPONSE
# =========================
def chat_response(user_query: str) -> str:

    if not is_medical_query(user_query):
        return (
            "⚠️ **Medical questions only**\n\n"
            "Please ask about symptoms, diseases, or health concerns."
        )

    # Try RAG first
    if qa_chain:
        try:
            rag_answer = qa_chain.run(user_query)
            if rag_answer and len(rag_answer) > 40:
                return (
                    "**Based on verified medical sources:**\n\n"
                    + format_points(rag_answer)
                )
        except Exception:
            pass

    # Fallback
    prompt = f"""
You are a medical information assistant.

Rules:
- Educational use only
- No diagnosis
- No prescriptions
- Answer in short numbered points (max 6)
- Each point on a new line
- End with doctor disclaimer

Question:
{user_query}
"""

    response = llm.invoke(prompt).content
    return format_points(response)

# =========================
# PDF CHAT
# =========================
def pdf_chat_response(pdf_path: str, question: str) -> str:

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    temp_db = FAISS.from_documents(chunks, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=temp_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

    answer = qa.run(question)

    return (
        "📄 **Answer based on uploaded medical document:**\n\n"
        + format_points(answer, max_points=8)
        + "\n\n⚠️ Educational use only."
    )

# =========================
# IMAGE SAFETY
# =========================
def image_safe_response() -> str:
    return (
        "🖼️ **Image received**\n\n"
        "I cannot diagnose from images.\n\n"
        "1. I can describe visible features\n"
        "2. Explain general medical possibilities\n"
        "3. Suggest when to consult a doctor\n\n"
        "⚠️ Consult a healthcare professional."
    )
