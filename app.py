import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import openai

# Set page config
st.set_page_config(
    page_title="Ask Tony: Digital Teaching Chatbot",
    page_icon="📘",
    layout="centered"
)

# Set OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set.")
    st.stop()

openai.api_key = api_key

# Load the book
@st.cache_data
def load_text():
    with open("teaching-in-a-digital-age.txt", "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# Build QA system
@st.cache_resource
def setup_qa():
    text = load_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3, openai_api_key=api_key)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# UI
st.markdown("<h1 style='text-align: center;'>📘 Ask Tony</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Your friendly guide to <i>Teaching in a Digital Age</i> by Tony Bates</p>", unsafe_allow_html=True)

st.divider()

query = st.text_input("💬 Ask a question based on the book:")
qa = setup_qa()

if query:
    with st.spinner("Thinking..."):
        answer = qa.run(query)
        st.markdown("### 📖 Answer")
        st.write(answer)
