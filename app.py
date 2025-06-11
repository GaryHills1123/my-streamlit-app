import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vectorstore_utils import load_or_build_vectorstore

# Load the initial system prompt from file
def load_initial_prompt(path="initial_prompt.txt"):
    with open(path, "r") as f:
        return f.read()

# Set up LLM with system prompt
def setup_llm(system_prompt):
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        model_kwargs={"system_prompt": system_prompt}
    )

# Build the RetrievalQA chain
def build_qa_chain(llm, vectorstore):
    prompt_template = """
You are Tony Bates, author of *Teaching in a Digital Age*. Answer the question using the following context:

{context}

Question: {question}
Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template.strip(),
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# Streamlit UI
st.set_page_config(page_title="📘 Ask the Textbook", layout="wide")
st.title("📘 Ask the Textbook")
st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

# Build vectorstore from text
vectorstore = load_or_build_vectorstore()

# Set up LLM and chain
system_prompt = load_initial_prompt()
llm = setup_llm(system_prompt)
qa_chain = build_qa_chain(llm, vectorstore)

# Input field
query = st.text_input("💬 Ask a question:", placeholder="e.g., What are affordances in online learning?")

# Handle query
if query:
    with st.spinner("Thinking like Tony Bates..."):
        result = qa_chain({"query": query})
        st.markdown("### 🧠 Answer")
        st.write(result["result"])
