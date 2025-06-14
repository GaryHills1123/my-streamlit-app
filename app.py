# import streamlit.web.server.server_util as server_util
# server_util._add_header = lambda *args, **kwargs: None

import streamlit as st
from elevenlabs.client import ElevenLabs
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
audio = client.generate(
    text=result,
    voice="Rachel",
    model="eleven_monolingual_v1"
)
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from vectorstore_utils import load_or_build_vectorstore

st.set_page_config(page_title="📘 Ask the Textbook", page_icon="📘")
st.title("Ask the Textbook")
st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

query = st.text_input("💬 Ask a question:")

vectorstore = load_or_build_vectorstore()

# temperature ∈ [0, 1] controls randomness: 0 yields deterministic output, 1 allows maximum creativity.
llm = ChatOpenAI(model="gpt-4o", temperature=1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

if query:
    result = qa_chain.run(query)
    st.write(result)

    # ElevenLabs TTS (optional playback)
    try:
        from elevenlabs import generate, set_api_key
        import os

        # Set API key from env var
        set_api_key(os.getenv("ELEVENLABS_API_KEY"))

        # Generate speech from response text
        audio = generate(
            text=result,
            voice="Adam",  # Try "Bella", "Adam", "Antoni", etc.
            model="eleven_monolingual_v1"
        )

        # Play audio in Streamlit
        st.audio(audio, format="audio/mp3")

    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
