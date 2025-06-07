import os
import streamlit as st
import openai

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Streamlit config
st.set_page_config(
    page_title="Ask the Textbook",
    page_icon="📘",
    layout="centered"
)

# Load OpenAI key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set. Please add it in your Render environment variables.")
    st.stop()

openai.api_key = api_key

# Load the book
@st.cache_data
def load_text():
    with open("teaching-in-a-digital-age.txt", "r", encoding="utf-8", errors="ignore") as file:
        return file.read()

# Set up chain
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

# Input and response UI only
query = st.text_input("💬 What would you like to ask?")
qa = setup_qa()

if query:
    with st.spinner("Thinking..."):
        response = qa.invoke(query)

        answer_text = response["result"] if isinstance(response, dict) and "result" in response else str(response)
        clean_answer = answer_text.replace("\n", "<br>").replace("**", "<b>").replace("*", "•")

        st.markdown("### 📖 Answer", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: #f0f4f8; padding: 1em; border-radius: 8px; font-size: 16px; line-height: 1.6;">
            {clean_answer}
        </div>
        """, unsafe_allow_html=True)
