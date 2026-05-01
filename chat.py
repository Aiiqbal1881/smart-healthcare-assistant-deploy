import os
from dotenv import load_dotenv
load_dotenv()

import re
from langchain_groq import ChatGroq

# PDF
from chat import chat_response, pdf_chat_response, image_safe_response
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# ---------------- LLM ----------------
llm = ChatGroq(
    temperature=0.3,
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ---------------- MEDICAL DETECTION ----------------
MEDICAL_KEYWORDS = [
    "symptom", "disease", "fever", "pain", "infection", "asthma",
    "diabetes", "cancer", "covid", "health", "treatment", "medicine",
    "injury", "blood", "pressure", "mental", "depression", "anxiety",
    "headache", "vomiting", "cold", "cough", "chest", "burning"
]

def is_medical_query(query: str) -> bool:
    query = query.lower()
    return any(word in query for word in MEDICAL_KEYWORDS)

# ---------------- FORMAT RESPONSE ----------------
def format_points(text):
    lines = text.split("\n")
    points = []
    count = 1

    for line in lines:
        line = line.strip()
        if line:
            points.append(f"{count}. {line}")
            count += 1

    return "\n".join(points[:10])  # limit to 10 points

# ---------------- CHAT RESPONSE ----------------
def chat_response(query):

    if not is_medical_query(query):
        return (
            "⚠️ This assistant is for medical questions only.\n\n"
            "Please ask about symptoms, diseases, or health issues."
        )

    prompt = f"""
You are a medical assistant.

User query: {query}

Give answer in:
- Simple language
- Bullet points
- Maximum 10 points
- Include: symptoms, causes, and basic advice
"""

    result = llm.invoke(prompt)
    return format_points(result.content)

# ---------------- PDF RESPONSE ----------------
def pdf_chat_response(pdf_path, query):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        return "❌ Could not read PDF properly."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        return "❌ No readable content found in PDF."

    embeddings = FakeEmbeddings(size=384)
    db = FAISS.from_documents(chunks, embeddings)

    docs = db.similarity_search(query, k=3)

    if not docs:
        return "❌ No relevant info found in document."

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a medical assistant.

Based on this document:
{context}

Question: {query}

Give answer in bullet points (max 10).
"""

    result = llm.invoke(prompt)
    return format_points(result.content)

# ---------------- IMAGE RESPONSE ----------------
def image_safe_response():
    return """
📷 Image received

I cannot diagnose medical conditions from images.

However, I can help by:
1. Describing visible features
2. Explaining possible medical context
3. Suggesting when to consult a doctor

⚠️ Always consult a healthcare professional.
"""