# import os
import os
from dotenv import load_dotenv
load_dotenv()

import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# ------------------ CONFIG ------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store"

LLM_MODEL = "llama-3.1-8b-instant"  # ✅ supported Groq model

# ------------------ LOAD EMBEDDINGS ------------------
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ------------------ LOAD VECTOR STORE ------------------
db = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ------------------ LOAD LLM ------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0.2
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# ------------------ MEDICAL SCOPE CHECK ------------------
MEDICAL_KEYWORDS = [
    "symptom", "disease", "fever", "pain", "infection", "asthma",
    "diabetes", "cancer", "covid", "health", "treatment", "medicine",
    "injury", "blood", "pressure", "mental", "depression", "anxiety"
]

def is_medical_query(query: str) -> bool:
    query = query.lower()
    return any(word in query for word in MEDICAL_KEYWORDS)

# ------------------ FORMAT RESPONSE INTO POINTS ------------------
def format_points(text: str) -> str:
    lines = re.split(r"\n|\d+\.", text)
    points = [line.strip("-• ") for line in lines if len(line.strip()) > 10]

    formatted = ""
    for i, point in enumerate(points, 1):
        formatted += f"{i}. {point}\n"

    return formatted.strip()

# ------------------ MAIN FUNCTION ------------------
def chat_response(user_query: str) -> str:
    """
    Main function used by app.py
    """

    # ❌ Non-medical queries
    if not is_medical_query(user_query):
        return (
            "⚠️ **This assistant is designed only for medical and health-related questions.**\n\n"
            "If you have a health concern, symptom, or disease-related question, feel free to ask.\n\n"
            "For non-medical topics, please use a general-purpose assistant."
        )

    # ✅ Try RAG first
    try:
        rag_answer = qa_chain.run(user_query)

        if rag_answer and len(rag_answer.strip()) > 30:
            return (
                "**Based on verified medical sources:**\n\n"
                + format_points(rag_answer)
            )

    except Exception:
        pass  # fallback to general medical LLM

    # 🔁 Fallback (General medical info, still safe)
    general_prompt = f"""
You are a medical information assistant.
Rules:
- Educational information only
- No diagnosis
- No prescriptions
- Always respond in numbered points
- End with doctor disclaimer

Question:
{user_query}
"""

    response = llm.invoke(general_prompt).content

    return format_points(response)



# =========================
# PDF RAG (USER UPLOAD)
# =========================
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        + answer +
        "\n\n⚠️ This information is for educational purposes only."
    )



# =========================
# IMAGE SAFE ANALYSIS
# =========================
def image_safe_response() -> str:
    return (
        "🖼️ **Image received**\n\n"
        "I cannot diagnose medical conditions from images.\n\n"
        "However, I can help by:\n"
        "1. Describing visible features (color, shape, pattern)\n"
        "2. Explaining general medical possibilities\n"
        "3. Suggesting when to consult a doctor\n\n"
        "⚠️ Please consult a qualified healthcare professional for diagnosis."
    )
