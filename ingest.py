import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

DATA_DIRS = [
    "data/medical_pdfs/who",
    "data/medicine_info",
    "data/hospital_faqs"
]

documents = []

for folder in DATA_DIRS:
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)
db.save_local("vector_store")

print("✅ Vector database created successfully")
