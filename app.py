import os
import streamlit as st
from elevenlabs.client import ElevenLabs
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from vectorstore_utils import load_or_build_vectorstore

# Set Streamlit page config
st.set_page_config(page_title="📘 Ask the Textbook", page_icon="📘")
st.title("Ask the Textbook")
st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

# Get user query
query = st.text_input("💬 Ask a question:")

# Load vector store and LLM
vectorstore = load_or_build_vectorstore()
llm = ChatOpenAI(model="gpt-4o", temperature=1)

# Set up Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Handle user query
if query:
    result = qa_chain.run(query)
    st.write(result)

    # Generate voice using ElevenLabs
    try:
        # Authenticate
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

        # Generate audio (returns generator)
        audio_generator = client.text_to_speech.convert(
            text=result,
            voice_id="21m00Tcm4TlvDq8ikWAM",        # Rachel's voice ID
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32",
            optimize_streaming_latency=1  # Required kwarg; range: 0 (best latency) to 4 (best quality)
        )

        # Convert to byte stream for playback
        audio_bytes = b"".join(audio_generator)

        # Play audio in Streamlit
        from io import BytesIO
        st.audio(BytesIO(audio_bytes), format="audio/mp3")
        
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
