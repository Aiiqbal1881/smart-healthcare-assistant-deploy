import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------ CONFIG ------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store"
LLM_MODEL = "llama-3.1-8b-instant"  # Groq supported

# ------------------ EMBEDDINGS ------------------
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ------------------ LOAD LLM ------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0.2
)

# ------------------ MEDICAL SCOPE CHECK ------------------
MEDICAL_KEYWORDS = [
    "symptom", "disease", "fever", "pain", "infection", "asthma",
    "diabetes", "cancer", "covid", "health", "treatment", "medicine",
    "injury", "blood", "pressure", "mental", "depression", "anxiety",
    "headache", "cold", "flu"
]

def is_medical_query(query: str) -> bool:
    return any(word in query.lower() for word in MEDICAL_KEYWORDS)

# ------------------ FORMAT INTO POINTS ------------------
def format_points(text: str) -> str:
    """
    Forces clean numbered bullet points even if the LLM
    returns everything in a single paragraph.
    """

    # Step 1: Normalize spacing
    text = text.replace("\n", " ").strip()

    # Step 2: Split on numbered patterns like "1. ", "2. "
    parts = re.split(r"(?=\d+\.\s)", text)

    points = []
    for part in parts:
        clean = part.strip()
        if len(clean) > 20:
            clean = re.sub(r"^\d+\.\s*", "", clean)
            points.append(clean)

    # Step 3: Rebuild numbered list
    if not points:
        return text

    return "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1))

# ------------------ MAIN CHAT FUNCTION ------------------
def chat_response(user_query: str) -> str:

    if not is_medical_query(user_query):
        return (
            "⚠️ **This assistant is designed only for medical and health-related questions.**\n\n"
            "If you have a health concern, symptom, or disease-related question, feel free to ask.\n\n"
            "For non-medical topics, please use a general-purpose assistant."
        )

    # ---------- TRY RAG ----------
    try:
        if os.path.exists(VECTOR_STORE_PATH):
            db = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = db.as_retriever(search_kwargs={"k": 3})

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )

            rag_answer = qa_chain.run(user_query)

            if rag_answer and len(rag_answer.strip()) > 30:
                return (
                    "**Based on verified medical sources:**\n\n"
                    + format_points(rag_answer)
                    + "\n\n⚠️ Educational use only. Consult a healthcare professional."
                )
    except Exception:
        pass

    # ---------- FALLBACK ----------
    prompt = f"""
You are a medical information assistant.

Rules:
- Educational information only
- No diagnosis
- No prescriptions
- Respond in numbered points
- End with doctor disclaimer

Question:
{user_query}
"""

    response = llm.invoke(prompt).content

    return (
        format_points(response)
        + "\n\n⚠️ Educational use only. Consult a healthcare professional."
    )

# =========================
# PDF QUESTION ANSWERING
# =========================
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        retriever=temp_db.as_retriever(search_kwargs={"k": 3})
    )

    answer = qa.run(question)

    return (
        "📄 **Answer based on uploaded medical document:**\n\n"
        + format_points(answer)
        + "\n\n⚠️ Educational use only. Consult a healthcare professional."
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
        "3. Suggesting when to consult a doctor\n\n"
        "⚠️ Always consult a healthcare professional."
    )
