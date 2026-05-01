import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

# -----------------------------
# LLM SETUP
# -----------------------------
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.3,
        max_tokens=1024
    )

# -----------------------------
# CHAT RESPONSE (MEDICAL)
# -----------------------------
def chat_response(user_input):
    llm = get_llm()

    prompt = f"""
You are a professional healthcare assistant.

User query:
{user_input}

Give answer in this format:

1. Possible causes
2. Common symptoms
3. General advice (safe, non-prescriptive)
4. When to consult a doctor

Keep it clear and simple.
"""

    return llm.invoke(prompt).content


# -----------------------------
# PDF RESPONSE (FIXED)
# -----------------------------
def pdf_chat_response(file, question):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    # ✅ FIX: handle both string path & uploaded file
    if isinstance(file, str):
        file_path = file
    else:
        file_path = "temp.pdf"
        with open(file_path, "wb") as f:
            f.write(file.read())

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        return "❌ No readable content found in PDF."

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if not chunks:
        return "❌ PDF has no usable text."

    embeddings = FakeEmbeddings(size=384)
    db = FAISS.from_documents(chunks, embeddings)

    results = db.similarity_search(question, k=2)

    context = "\n".join([doc.page_content for doc in results])

    llm = get_llm()

    prompt = f"""
You are analyzing a medical report.

Report Content:
{context}

Question:
{question}

Give a clear explanation like a doctor.
"""

    return llm.invoke(prompt).content


# -----------------------------
# IMAGE RESPONSE (SAFE)
# -----------------------------
def image_safe_response(image=None):
    return """
⚠️ I cannot diagnose medical conditions from images.

However, I can help with:
1. General observations
2. Possible explanations
3. When to consult a doctor

Please describe your symptoms for better assistance.
"""