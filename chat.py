import os
from dotenv import load_dotenv
load_dotenv()

import re

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# PDF
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# ------------------ LLM ------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

# ------------------ MEDICAL CHECK ------------------
MEDICAL_TERMS = [
    "fever", "pain", "asthma", "diabetes", "cancer", "covid",
    "infection", "blood", "pressure", "anxiety", "depression",
    "heart", "chest", "breathing", "symptom", "treatment"
]

def is_medical_query(query: str) -> bool:
    query = query.lower()
    return any(word in query for word in MEDICAL_TERMS)

# ------------------ FORMAT POINTS ------------------
def format_points(text: str) -> str:
    lines = re.split(r"\n|\d+\.", text)
    points = [l.strip("-• ") for l in lines if len(l.strip()) > 15]

    if not points:
        return text

    return "\n".join([f"{i+1}. {p}" for i, p in enumerate(points[:8])])

# ------------------ CHAT ------------------
def chat_response(user_query: str) -> str:

    if not is_medical_query(user_query):
        return (
            "⚠️ **This assistant is for medical questions only.**\n\n"
            "Ask about symptoms, diseases, or health issues."
        )

    prompt = f"""
You are a medical assistant.

Rules:
- Answer in numbered points (max 6–8 points)
- Keep answer clear and short
- Give practical advice
- Add when to see a doctor

Question:
{user_query}
"""

    try:
        response = llm.invoke(prompt).content
        return format_points(response)

    except Exception:
        return "⚠️ Error generating response."

# ------------------ PDF ------------------
def pdf_chat_response(pdf_path: str, question: str) -> str:

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        db = FAISS.from_documents(
            chunks,
            FakeEmbeddings(size=384)
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 3})
        )

        answer = qa.run(question)

        return format_points(answer)

    except Exception as e:
        return f"⚠️ PDF processing error: {str(e)}"

# ------------------ IMAGE ------------------
def image_safe_response():
    return (
        "🖼️ Image received\n\n"
        "I cannot diagnose from images.\n\n"
        "1. I can describe visible features\n"
        "2. Suggest possible conditions\n"
        "3. Recommend doctor consultation\n\n"
        "⚠️ Always consult a healthcare professional."
    )