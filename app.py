import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from vectorstore_utils import load_or_build_vectorstore

# Load vectorstore from file or build it
vectorstore = load_or_build_vectorstore()

# Load system prompt from file
with open("initial_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Setup chat model
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# Prompt template with system role
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{query}")
])

# Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# --- Streamlit UI ---

st.title("📘 Ask the Textbook")
st.markdown("Ask anything about Tony Bates' *Teaching in a Digital Age*")

query = st.text_input("💬 Ask a question:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.run({"query": query})
    st.write(result)
