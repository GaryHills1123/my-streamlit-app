import os
import streamlit as st
import openai

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Streamlit page setup
st.set_page_config(
    page_title="Ask the Textbook",
    page_icon="📘",
    layout="centered"
)

# Load API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set. Please add it in your Render environment variables.")
    st.stop()

openai.api_key = api_key  # Required by LangChain/OpenAI under the hood

# Load textbook text file
@st.cache_data
def load_text():
    with open("teaching-in-a-digital-age.txt", "r", encoding="utf-8", errors="ignore") as file:
        return file.read()

# Set up the vector store and retrieval chain
@st.cache_resource
def setup_qa():
    full_text = load_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([full_text])

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3, openai_api_key=api_key)

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# UI layout
st.markdown("<h1 style='text-align: center;'>📘 Ask the Textbook</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>"
    "Explore Tony Bates' insights on digital teaching — one question at a time."
    "</p>", 
    unsafe_allow_html=True
)
st.divider()

query = st.text_input("💬 What would you like to ask?")
qa = setup_qa()

if query:
    with st.spinner("Thinking..."):
        answer = qa.run(query)
        st.markdown("### 📖 Answer")
        st.write(answer)

