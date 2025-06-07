# 📘 Ask the Textbook

An AI-powered learning assistant trained on Tony Bates' *Teaching in a Digital Age* — built with Streamlit, LangChain, and OpenAI, and embedded in a Rise course.

## 🔍 What It Does

This app allows learners to:

- Ask natural-language questions about the book
- Get context-rich answers powered by GPT-4o
- Interact directly inside an eLearning module or website

## 🎯 Audience

This project is designed for:

- Instructional designers
- Online educators
- Learning technology teams exploring AI integration

## 🌐 Live Demo

▶️ [Launch the chatbot](https://my-streamlit-app-yj1z.onrender.com)

💬 Embedded in Rise (optional link): *Coming soon or hosted elsewhere*

## 🛠 Tech Stack

- **Streamlit** – web app framework
- **LangChain** – for document chunking and retrieval
- **OpenAI GPT-4o** – language model
- **FAISS** – vector search
- **Render** – deployment platform
- **Rise + WordPress** – course authoring and delivery

## 📂 Files

- `app.py` – Main app logic
- `teaching-in-a-digital-age.txt` – Source content
- `requirements.txt` – Python dependencies
- `Dockerfile` + `nginx.conf` – Render-compatible deploy

## 📚 Why This Book?

- Written by Tony Bates, a global authority on online learning
- Openly licensed (CC BY)
- Modular, searchable, and educationally rich
- Widely used in higher ed and instructional design contexts

## 🚀 How to Run Locally

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py
