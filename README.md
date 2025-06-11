# 📘 Ask the Textbook

An AI-powered learning assistant trained on Tony Bates' *Teaching in a Digital Age* — built with Streamlit, LangChain, and GPT-4o, and embedded in an Articulate Rise course.

## 🔍 What It Does

This app allows learners to:

- Ask natural-language questions about the textbook
- Receive clear, contextual responses grounded in the text
- Interact with a responsive teaching assistant modeled on Tony Bates
- Explore course design, media selection, assessment, and more

## 🎯 Audience

This project is designed for:

- Instructional designers
- Online educators
- Learning technology teams exploring AI integration
- Educators using *Teaching in a Digital Age* in PD or course design workshops

## 🌐 Live Demo

▶️ [Launch the chatbot](https://my-streamlit-app-yj1z.onrender.com)  
💬 *Also available as an embedded module in Rise (coming soon)*

## 🛠 Tech Stack

- **Streamlit** – lightweight web app framework
- **LangChain** – document chunking and vector retrieval
- **OpenAI GPT-4o** – for natural-language response generation
- **FAISS** – fast semantic search across textbook chunks
- **Render** – app hosting platform
- **Articulate Rise + WordPress** – optional front-end delivery

## 📚 Why This Book?

- Authored by Tony Bates, a leading expert in digital learning
- Openly licensed under CC BY
- Modular, pedagogically rich, and widely adopted
- Ideal for exploring the affordances of technology in education

## 📂 Project Files

- `app.py` – Main Streamlit app logic
- `teaching-in-a-digital-age.txt` – Source textbook content (used for embedding & retrieval)
- `initial_prompt.txt` – System prompt that defines assistant behavior, tone, pedagogy, and constraints
- `requirements.txt` – Python dependencies
- `Dockerfile` + `nginx.conf` – Render-compatible deployment setup

## 🧠 Prompt Design Philosophy

This assistant doesn’t just respond—it teaches.

The behavior of the AI assistant is defined by `initial_prompt.txt`, a modular system prompt that guides how the model thinks, speaks, and reasons. It’s designed to reflect the tone and structure of *Teaching in a Digital Age*, emphasizing:

- Clarity and accuracy over creativity
- Instructional reasoning (affordances, chain-of-thought)
- Structured answers (definition → explanation → implication → follow-up)
- Transparency (never fabricates; says "not addressed in the textbook" when needed)

This makes the tool not just an AI chatbot, but a reliable micro-tutor grounded in educational integrity.

## 🚀 How to Run Locally

```bash
git clone https://github.com/GaryHills1123/my-streamlit-app.git
cd my-streamlit-app
pip install -r requirements.txt
streamlit run app.py
