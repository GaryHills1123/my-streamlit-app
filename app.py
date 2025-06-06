import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import openai

# ✅ Step 1: Set your OpenAI API key from Render environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set in environment variables.")
    st.stop()

openai.api_key = api_key  # LangChain needs this set

# ✅ Step 2: Load the text file
@st.cache_data
def load_text():
    with open("teaching-in-a-digital-age.txt", "r", encoding="utf-8", errors="ignore") as file:
        return file.read()

# ✅ Step 3: Create the QA chain
@st.cache_resource
def setup_qa():
    full_text = load_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([full_text])

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3, openai_api_key=api_key)

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ✅ Streamlit UI
st.title("📘 Teaching in a Digital Age Bot")
st.markdown("Ask a question based on Tony Bates' open textbook about digital pedagogy:")

qa = setup_qa()
query = st.text_input("Your question")

if query:
    st.markdown("### Answer")
    st.write(qa.run(query))
