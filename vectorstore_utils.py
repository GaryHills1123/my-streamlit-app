import os
import pickle
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

VECTORSTORE_FILENAME = "faiss_store.pkl"

def load_or_build_vectorstore(text_path="teaching-in-a-digital-age.txt"):
    if os.path.exists(VECTORSTORE_FILENAME):
        with open(VECTORSTORE_FILENAME, "rb") as f:
            return pickle.load(f)

    loader = TextLoader(text_path, encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

    with open(VECTORSTORE_FILENAME, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

def load_system_prompt(prompt_path="initial_prompt.txt"):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
