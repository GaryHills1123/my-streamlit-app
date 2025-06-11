import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from vectorstore_utils import load_or_build_vectorstore

st.set_page_config(page_title="📘 Ask the Textbook", page_icon="📘")

st.title("📘 Ask the Textbook")
st.markdown("Ask a grounded, pedagogically trained AI assistant—modeled after Tony Bates—for clear, textbook-based answers, 
explained step by step, formatted for learning, and limited strictly to content from Teaching in a Digital Age.")

query = st.text_input("💬 Ask a question:")

vectorstore = load_or_build_vectorstore()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

if query:
    result = qa_chain.run(query)
    st.write(result)
