import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vectorstore_utils import load_or_build_vectorstore

# Load system prompt
def load_system_prompt(path="initial_prompt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# App config
st.set_page_config(page_title="Ask the Textbook", page_icon="📘")
st.title("📘 Ask the Textbook")
st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

# Load vectorstore and system prompt
system_prompt = load_system_prompt()
vectorstore = load_or_build_vectorstore()

# Prompt and model setup
prompt = PromptTemplate.from_template(system_prompt)
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# Conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask a question about the textbook..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        response = qa_chain.run(user_input)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
