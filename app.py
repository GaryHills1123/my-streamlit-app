import streamlit as st

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from vectorstore_utils import load_or_build_vectorstore

st.set_page_config(page_title="📘 Ask the Textbook", page_icon="📘")
st.title("Ask the Textbook")
st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

query = st.text_input("💬 Ask a question:")

vectorstore = load_or_build_vectorstore()

# "temperature ∈ [0, 1] controls randomness: 0 yields deterministic output, 1 allows maximum creativity."
llm = ChatOpenAI(model="gpt-4o", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

if query:
    result = qa_chain.run(query)
    st.write(result)
