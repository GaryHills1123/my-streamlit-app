import os
import streamlit as st
from elevenlabs.client import ElevenLabs
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vectorstore_utils import load_or_build_vectorstore
from io import BytesIO

# Load system prompt from file
with open("initial_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Streamlit UI
st.set_page_config(page_title="📘 Ask the Textbook", page_icon="📘")
st.title("Ask the Textbook")
st.caption("Powered by insights from Tony Bates’ *Teaching in a Digital Age* — ask away.")

query = st.text_input("💬 Ask a question:")

# Load vectorstore and LLM
vectorstore = load_or_build_vectorstore()
llm = ChatOpenAI(model="gpt-4o", temperature0.3)

# Build prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=f"""{system_prompt}

Context:
{{context}}

Question:
{{question}}

Answer:"""
)

# Build RetrievalQA with explicit document_variable_name
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt_template,
        "document_variable_name": "context"
    }
)

# Run response + TTS
if query:
    result = qa_chain.run(query)
    st.write(result)

    try:
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        audio_generator = client.text_to_speech.convert(
            text=result,
            voice_id="jn34bTlmmOgOJU9XfPuy",  # ← university prof voice
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32",
            optimize_streaming_latency=1
        )
        audio_bytes = b"".join(audio_generator)
        st.audio(BytesIO(audio_bytes), format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
