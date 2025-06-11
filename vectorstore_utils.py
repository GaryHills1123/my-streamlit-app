import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter

def load_or_build_vectorstore():
    faiss_index_path = "faiss_index"
    if os.path.exists(faiss_index_path):
        return FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # ✅ Load with fallback for encoding issues
    with open("teaching-in-a-digital-age.txt", "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    docs = [Document(page_content=raw_text)]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())
    vectorstore.save_local(faiss_index_path)
    return vectorstore
