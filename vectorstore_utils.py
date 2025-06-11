import pickle
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_or_build_vectorstore():
    if os.path.exists("faiss_store.pkl"):
        with open("faiss_store.pkl", "rb") as f:
            return pickle.load(f)

    # ✅ FIX: Explicitly specify UTF-8 encoding
    loader = TextLoader("teaching-in-a-digital-age.txt", encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(splits, embeddings)

    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)

    return store

def load_system_prompt():
    with open("initial_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()
