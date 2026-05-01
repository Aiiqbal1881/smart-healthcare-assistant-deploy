import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ------------------ CONFIG ------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store"
LLM_MODEL = "llama-3.1-8b-instant"

# ------------------ EMBEDDINGS ------------------
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

# ------------------ LOAD VECTOR STORE (SAFE) ------------------
try:
    db = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
except:
    retriever = None

# ------------------ LOAD LLM ------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0.3
)

# ------------------ MEDICAL CHECK ------------------
def is_medical_query(query: str) -> bool:
    query = query.lower()

    medical_terms = [
        "pain","fever","burning","chest","heartburn","acid","reflux",
        "cough","breathing","infection","disease","symptoms",
        "treatment","doctor","medicine","anxiety","depression",
        "vomit","nausea","headache","stomach","pressure"
    ]

    return any(term in query for term in medical_terms)

# ------------------ FORMAT ------------------
def format_points(text: str, max_points=6) -> str:
    lines = re.split(r"\n|\d+\.", text)
    points = [line.strip("-• ") for line in lines if len(line.strip()) > 15]

    points = points[:max_points]

    return "\n".join([f"{i+1}. {p}" for i, p in enumerate(points)])

# ------------------ MAIN CHAT ------------------
def chat_response(user_query: str) -> str:

    if not is_medical_query(user_query):
        return (
            "⚠️ This assistant is for medical questions only.\n\n"
            "Please ask about symptoms, diseases, or health issues."
        )

    # -------- TRY RAG --------
    if retriever:
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever
            )

            rag_answer = qa_chain.run(user_query)

            if rag_answer:
                return "**Based on medical knowledge:**\n\n" + format_points(rag_answer)

        except:
            pass

    # -------- FALLBACK (DOCTOR STYLE) --------
    prompt = f"""
You are a professional medical assistant.

Instructions:
- Understand patient symptoms
- Suggest possible conditions (not diagnosis)
- Give practical advice
- Keep response SHORT and CLEAR
- Use 5-6 bullet points ONLY

Question:
{user_query}
"""

    response = llm.invoke(prompt).content
    return format_points(response)

# =========================
# PDF CHAT (SAFE)
# =========================
def pdf_chat_response(pdf_path: str, question: str) -> str:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        return "⚠️ No readable content found in PDF."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        return "⚠️ This PDF may be scanned or empty."

    db = FAISS.from_documents(chunks, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )

    answer = qa.run(question)

    return "📄 **Answer based on document:**\n\n" + format_points(answer)

# =========================
# IMAGE RESPONSE
# =========================
def image_safe_response() -> str:
    return (
        "🖼️ Image received\n\n"
        "This appears to be a medical document or prescription.\n\n"
        "⚠️ Currently, text extraction from images is not supported here.\n\n"
        "👉 Please upload as PDF or type details manually."
    )