import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

# -----------------------------
# LLM Setup
# -----------------------------
def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )

# -----------------------------
# CHAT RESPONSE (Medical)
# -----------------------------
def chat_response(user_input):
    llm = get_llm()

    prompt = f"""
You are a helpful medical assistant.

User symptoms:
{user_input}

Give:
1. Possible common causes
2. General advice (non-prescriptive)
3. When to consult a doctor

Keep it simple and safe.
"""

    return llm.invoke(prompt).content


# -----------------------------
# PDF RESPONSE (SAFE VERSION)
# -----------------------------
def pdf_chat_response(file, question):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FakeEmbeddings

    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    if len(docs) == 0:
        return "No content found in PDF."

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if len(chunks) == 0:
        return "PDF has no readable content."

    # Use FakeEmbeddings (NO TORCH ISSUE)
    embeddings = FakeEmbeddings(size=384)

    db = FAISS.from_documents(chunks, embeddings)

    results = db.similarity_search(question, k=2)

    context = "\n".join([doc.page_content for doc in results])

    llm = get_llm()

    prompt = f"""
Answer based on the PDF:

{context}

Question: {question}
"""

    return llm.invoke(prompt).content


# -----------------------------
# IMAGE RESPONSE (SAFE)
# -----------------------------
def image_safe_response(image):
    return """
I cannot diagnose medical conditions from images.

However, I can help with:
1. General observations
2. Possible explanations
3. When to consult a doctor

Please describe your symptoms for better help.
"""