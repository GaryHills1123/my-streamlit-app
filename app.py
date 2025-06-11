import streamlit as st

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from vectorstore_utils import load_or_build_vectorstore

st.set_page_config(page_title="📘 Ask the Textbook", page_icon="📘")

col1, col2 = st.columns([1, 12])  # Adjust ratio for spacing
with col1:
    st.image("assets/animated-question.gif", width=28)
with col2:
    st.markdown("### Ask the Textbook")
    st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

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
