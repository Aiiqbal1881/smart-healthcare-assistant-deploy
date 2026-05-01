import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

# ------------------ LLM ------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

# ------------------ FORMAT POINTS ------------------
def format_points(text: str, max_points: int = 6) -> str:
    parts = re.split(r"\n|•|-|\d+\.", text)
    parts = [p.strip() for p in parts if len(p.strip()) > 8]

    if not parts:
        return text

    formatted = ""
    for i, p in enumerate(parts[:max_points], 1):
        formatted += f"{i}. {p}\n"

    return formatted.strip()


# ------------------ MEDICAL CHECK ------------------
def is_medical_query(query: str) -> bool:
    query = query.lower()

    keywords = [
        "symptom","disease","fever","pain","infection","asthma",
        "diabetes","cancer","covid","health","treatment","medicine",
        "injury","blood","pressure","mental","depression","anxiety",
        "headache","cold","cough","doctor","report"
    ]

    return any(k in query for k in keywords)


# ------------------ CHAT ------------------
def chat_response(user_query: str) -> str:

    if not is_medical_query(user_query):
        return (
            "⚠️ This assistant is for medical questions only.\n\n"
            "Ask about symptoms, diseases, or health issues."
        )

    prompt = f"""
You are a medical assistant.

Rules:
- Answer ONLY in 5–6 numbered points
- Each point must be short
- No paragraph format
- Simple language
- Add doctor disclaimer at end

Question:
{user_query}
"""

    response = llm.invoke(prompt).content
    return format_points(response)


# ------------------ PDF ------------------
def pdf_chat_response(pdf_path: str, question: str) -> str:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain.chains import RetrievalQA

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )

    result = qa.run(question)

    return "📄 Answer based on document:\n\n" + format_points(result)


# ------------------ IMAGE ------------------
def image_analysis_response(uploaded_image) -> str:
    return (
        "📄 Medical image received\n\n"
        "1. This appears to be a prescription or medical report\n"
        "2. It may contain medicines and dosage instructions\n"
        "3. It may include diagnosis or symptoms\n"
        "4. Doctor advice or follow-up may be present\n"
        "5. Handwritten text cannot be fully analyzed here\n\n"
        "⚠️ Please consult a doctor for accurate interpretation"
    )
# =========================
# IMAGE SAFE ANALYSIS
# =========================
def image_safe_response() -> str:
    return (
        "🖼️ **Image received**\n\n"
        "I cannot diagnose medical conditions from images.\n\n"
        "However, I can help by:\n"
        "1. Describing visible features\n"
        "2. Explaining possible medical context\n"
        "3. Suggesting when to consult a doctor\n\n"
        "⚠️ Always consult a healthcare professional."
    )