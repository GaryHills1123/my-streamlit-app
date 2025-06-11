import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

def load_or_build_vectorstore():
    faiss_index_path = "faiss_index"
    if os.path.exists(faiss_index_path):
        return FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    loader = TextLoader("teaching-in-a-digital-age.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())
    vectorstore.save_local(faiss_index_path)
    return vectorstore
