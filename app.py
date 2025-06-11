import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from vectorstore_utils import load_or_build_vectorstore

st.set_page_config(page_title="📘 Ask the Textbook", page_icon="📘")

st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="assets/animated-question.gif" width="28"/>
        <h1 style="font-size: 1.8em; margin: 0;">Ask the Textbook</h1>
    </div>
    <p style="font-size: 1.1em; color: gray; margin-top: 0.25em;">
        Ask anything about Tony Bates' <em>Teaching in a Digital Age</em>
    </p>
    """,
    unsafe_allow_html=True
)


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
