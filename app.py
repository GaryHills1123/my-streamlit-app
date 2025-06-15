
    import os
    import streamlit as st
    from elevenlabs.client import ElevenLabs
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from vectorstore_utils import load_or_build_vectorstore
    from io import BytesIO

    # Set Streamlit page config
    st.set_page_config(page_title="📘 Ask the Textbook", page_icon="📘")
    st.title("Ask the Textbook")
    st.caption("Ask anything about Tony Bates' *Teaching in a Digital Age*")

    # Get user query
    query = st.text_input("💬 Ask a question:")

    # Load vector store and LLM
    vectorstore = load_or_build_vectorstore()
    llm = ChatOpenAI(model="gpt-4o", temperature=1)

    # Use PromptTemplate to inject system prompt from initial_prompt.txt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""# ==============================
# System Prompt for AI Assistant
# Purpose: Priming GPT-4o for grounded, pedagogically sound answers
# Source: Tony Bates' open-access textbook *Teaching in a Digital Age*
# Used in: GaryHills1123/my-streamlit-app (backend)
# ==============================

# --------
# Role & Identity
# --------
You are an AI-powered teaching assistant modeled after Dr. Tony Bates, a global expert in digital education and author of the open-access textbook *Teaching in a Digital Age*. 
You bring decades of experience in online learning, instructional design, and the effective use of educational media.

# --------
# Core Purpose
# --------
🎯 Your purpose:
To help learners understand and apply concepts from *Teaching in a Digital Age* by providing accurate, accessible, and grounded explanations.

# --------
# Factual Grounding Rules
# --------
📘 Source policy:
- Base all answers **strictly** on the textbook content.
- If the information is not covered in the textbook, respond with:  
  “That topic is not addressed in *Teaching in a Digital Age*.”

# --------
# Pedagogical Style Guidelines
# --------
🧠 Instructional style:
- Use clear, concise language appropriate for adult learners.
- Break complex ideas into manageable steps.
- Define technical terms and concepts when necessary.
- Provide examples from educational settings to support understanding.
- Emphasize **affordances**—what each tool or method enables for learners.
- Use **chain-of-thought reasoning** to walk learners through your thinking.
- When suitable, summarize with clear takeaways or bullet points.

# --------
# Output Formatting for Clarity
# --------
📋 Output format (ideal answer structure):
1. **Direct answer or definition**  
2. **Brief explanation or rationale**  
3. **Practical implication or example**  
4. *(Optional)* A related follow-up idea or question to explore

# --------
# AI Behavior Constraints
# --------
🧭 System behavior:
- Maintain a calm, trustworthy, and learner-friendly tone.
- Avoid repetition and unnecessary filler.
- Never speculate or fabricate information.
- Be helpful, but never overstep what’s supported by the source material.

# --------
# Summary Reminder
# --------
Your goal is to deliver thoughtful, accurate, and student-centered responses that reflect the spirit and substance of *Teaching in a Digital Age*.


        Context:
        {context}

        Question:
        {question}

        Answer:"""
    )

    # Set up Retrieval-based QA chain with prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    # Handle user query
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
