import os
import pickle
import chardet
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

def load_or_build_vectorstore():
    if os.path.exists("faiss_store.pkl"):
        with open("faiss_store.pkl", "rb") as f:
            return pickle.load(f)

    # Detect encoding
    with open("teaching-in-a-digital-age.txt", "rb") as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result["encoding"]

    # Use detected encoding in TextLoader
    loader = TextLoader("teaching-in-a-digital-age.txt", encoding=encoding)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore
