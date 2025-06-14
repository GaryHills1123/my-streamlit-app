import os
import streamlit as st
from elevenlabs.client import ElevenLabs
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from vectorstore_utils import load_or_build_vectorstore

st.set_page_config(page_title="📘 Ask the Textbook", page_icon="📘")
st.title("Ask the Textbook")
st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

query = st.text_input("💬 Ask a question:")

vectorstore = load_or_build_vectorstore()
llm = ChatOpenAI(model="gpt-4o", temperature=1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

if query:
    result = qa_chain.run(query)
    st.write(result)

    # ElevenLabs TTS (modern SDK)
    try:
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        audio = client.generate(
            text=result,
            voice="Adam",
            model="eleven_monolingual_v1"
        )
        st.audio(audio, format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
