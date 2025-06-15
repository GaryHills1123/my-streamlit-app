import os
import streamlit as st
from elevenlabs.client import ElevenLabs
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vectorstore_utils import load_or_build_vectorstore
from io import BytesIO

# Load system prompt at runtime
with open("initial_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Set Streamlit page config
st.set_page_config(page_title="📘 Ask the Textbook", page_icon="📘")
st.title("Ask the Textbook")
st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

# Get user query
query = st.text_input("💬 Ask a question:")

# Load vector store and LLM
vectorstore = load_or_build_vectorstore()
llm = ChatOpenAI(model="gpt-4o", temperature=1)

# Correct prompt: declare only 'question', but include {context} in template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=f"""{system_prompt}

Context:
{{context}}

Question:
{{question}}

Answer:"""
)

# Create RetrievalQA chain using the corrected prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

# Run query and return result
if query:
    result = qa_chain.run(query)
    st.write(result)

    try:
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        audio_generator = client.text_to_speech.convert(
            text=result,
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32",
            optimize_streaming_latency=1
        )
        audio_bytes = b"".join(audio_generator)
        st.audio(BytesIO(audio_bytes), format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
