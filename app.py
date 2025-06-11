import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from vectorstore_utils import load_or_build_vectorstore, load_system_prompt

# Load environment variable for OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OpenAI API key in the environment.")
    st.stop()

# Page configuration
st.set_page_config(page_title="Ask the Textbook", layout="wide")
st.title("📘 Ask the Textbook")
st.subheader("Ask anything about Tony Bates' *Teaching in a Digital Age*")

# Debugging output
with st.expander("📂 Current working directory:", expanded=False):
    st.code(f"{os.getcwd()}")
with st.expander("📄 Files found in this directory:", expanded=False):
    st.code(os.listdir())

# Load vectorstore
with st.spinner("Loading knowledge base..."):
    vectorstore = load_or_build_vectorstore()

# Load system prompt
system_prompt = load_system_prompt()

# Initialize the LLM and RetrievalQA chain
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"document_variable_name": "context"},
    return_source_documents=True
)

# Ask a question
query = st.text_input("💬 Ask a question:", placeholder="e.g. What is online learning according to Tony Bates?")
if query:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": query})
        st.markdown("### 📎 Answer")
        st.write(result["result"])

        # Show source docs (optional)
        with st.expander("🔍 Source Documents"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**{i+1}.** {doc.metadata.get('source', 'N/A')}")
                st.write(doc.page_content[:500] + "...")
