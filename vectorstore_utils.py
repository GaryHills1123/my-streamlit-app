# vectorstore_utils.py

import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_or_build_vectorstore(
    txt_path="teaching-in-a-digital-age.txt", 
    faiss_dir="faiss_index"
):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing file: {txt_path}")

    if os.path.exists(faiss_dir):
        return FAISS.load_local(faiss_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    with open(txt_path, "r", encoding="latin-1") as f:
        raw_text = f.read()

    docs = [Document(page_content=raw_text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(faiss_dir)

    return store
