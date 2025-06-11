import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pickle
import os

# Load or build FAISS vectorstore
@st.cache_resource
def load_or_build_vectorstore():
    txt_path = os.path.join(os.getcwd(), "teaching-in-a-digital-age.txt")

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"❌ File not found at expected path: {txt_path}")

    if os.path.exists("faiss_store.pkl"):
        with open("faiss_store.pkl", "rb") as f:
            return pickle.load(f)
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        docs = [Document(page_content=raw_text)]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        store = FAISS.from_documents(chunks, embeddings)

        with open("faiss_store.pkl", "wb") as f:
            pickle.dump(store, f)

        return store

# Load system prompt
def load_system_prompt():
    with open("initial_prompt.txt", "r", encoding="utf-8") as file:
        return file.read()

# Streamlit UI
st.set_page_config(page_title="Ask the Textbook", page_icon="📘")
st.title("📘 Ask the Textbook")
st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

# Debug info
st.write("📂 Current working directory:", os.getcwd())
st.write("📄 Files found in this directory:", os.listdir("."))

# Load system prompt and vectorstore
system_prompt = load_system_prompt()
vectorstore = load_or_build_vectorstore()

# Prepare prompt template
prompt_template = PromptTemplate.from_template(system_prompt)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# User input
if user_input := st.chat_input("Ask a question about the textbook..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        response = qa_chain.run(user_input)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
