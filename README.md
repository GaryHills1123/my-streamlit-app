# 📘 Ask the Textbook – AI-Powered Learning Assistant

*Turn a textbook into a tutor. This tool lets learners query Tony Bates’* Teaching in a Digital Age *using natural language—receiving structured, contextual, and accurate answers.*  
Built with Streamlit, LangChain, FAISS, GPT-4o — and embedded inside an Articulate Rise course.

---

## 🔍 What It Does

This assistant lets users:

- Ask natural-language questions about the textbook
- Get clear, contextual answers grounded in the text (no guessing or hallucinations)
- Interact with a tutor modeled on Tony Bates’ tone and reasoning
- Explore course design, media selection, assessment, and more

---

## 🎯 Who It’s For

Ideal for:

- Instructional designers and curriculum developers
- Online educators and PD facilitators
- Learning tech teams exploring AI integration
- Anyone using *Teaching in a Digital Age* for instructional planning

---

## 🌐 Try It Out

📺 [Launch the chatbot](https://my-streamlit-app-yj1z.onrender.com)  
📺 [Embedded in an interactive Rise course](https://www.garyhills.dev/ai/)

---

## 🛠 Tech Stack

- **Streamlit** – rapid web app development
- **LangChain** – document chunking & retrieval
- **OpenAI GPT-4o** – natural-language reasoning engine
- **FAISS** – fast semantic search
- **Render** – cloud hosting
- **Articulate Rise + WordPress** – optional front-end delivery

---

## 📚 Why This Textbook?

- Tony Bates is a respected voice in digital education
- Openly licensed under CC BY
- Modular and well-structured—ideal for vector search
- Broadly used in higher ed, PD, and online course design

---

## 📂 Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit app logic — handles UI, query input, and response rendering |
| `vectorstore_utils.py` | Utility functions for chunking, embedding, and FAISS retrieval |
| `teaching-in-a-digital-age.txt` | Full textbook content used to generate embeddings |
| `initial_prompt.txt` | System prompt defining assistant behavior, tone, and response structure |
| `requirements.txt` | Python dependencies for local or hosted deployment |
| `Dockerfile` | Container setup for deployment on Render |
| `nginx.conf` | Nginx configuration for routing and static assets (used in Docker image) |
| `README.md` | Project overview, usage instructions, and design philosophy |

---

## 🧠 Prompt Design Philosophy

This assistant doesn’t just respond—it teaches.

The `initial_prompt.txt` defines how the AI speaks, reasons, and responds, with a focus on:

- **Clarity over creativity**  
- **Instructional logic** (definition → explanation → implication → optional follow-up)  
- **Groundedness** (won’t fabricate—admits when something isn’t in the text)  
- **Tone matching** Bates' educational style  

This isn’t just an AI chatbot—it’s a focused teaching tool aligned with digital learning principles.

---

## 🚀 Local Setup

```bash
git clone https://github.com/GaryHills1123/my-streamlit-app.git
cd my-streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🤝 Contribute or Connect

Got feedback, ideas, or want to collaborate? Reach out via [garyhills.dev](https://garyhills.dev) or fork the repo and start experimenting.
