import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

VECTORSTORE_FILENAME = "faiss_store.pkl"
TEXT_FILE = "teaching-in-a-digital-age.txt"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_or_build_vectorstore():
    if os.path.exists(VECTORSTORE_FILENAME):
        with open(VECTORSTORE_FILENAME, "rb") as f:
            return pickle.load(f)

    # ✅ Read and validate text file
    if not os.path.exists(TEXT_FILE):
        raise FileNotFoundError(f"{TEXT_FILE} not found in working directory.")

    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if not raw_text.strip():
        raise ValueError(f"{TEXT_FILE} is empty or unreadable.")

    # ✅ Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents([Document(page_content=raw_text)])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(VECTORSTORE_FILENAME, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

def load_system_prompt():
    try:
        with open("initial_prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are an assistant that answers questions based on the textbook 'Teaching in a Digital Age'."
