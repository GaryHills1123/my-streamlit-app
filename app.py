import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import pickle

# Load vectorstore (FAISS)
@st.cache_resource
def load_vectorstore():
    with open("faiss_store.pkl", "rb") as f:
        return pickle.load(f)

# Load system prompt from file
def load_system_prompt():
    with open("initial_prompt.txt", "r", encoding="utf-8") as file:
        return file.read()

# Set up the page
st.set_page_config(page_title="Ask the Textbook", page_icon="📘")
st.title("📘 Ask the Textbook")
st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

# Load system prompt + vectorstore
vectorstore = load_vectorstore()
system_prompt = load_system_prompt()

# Create LangChain-style prompt template
prompt_template = PromptTemplate.from_template(system_prompt)

# Initialize OpenAI Chat model
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Set up the QA chain with the system prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template}
)

# Chat history handling
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Input field
if user_input := st.chat_input("Ask a question about the textbook..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        response = qa_chain.run(user_input)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
