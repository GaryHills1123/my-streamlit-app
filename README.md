📘 Ask the Textbook – AI-Powered Learning Assistant
Turn a textbook into a tutor. This tool lets learners query Tony Bates’ Teaching in a Digital Age using natural language—receiving structured, contextual, and accurate answers.
Built with Streamlit, LangChain, FAISS, GPT-4o — and embedded inside an Articulate Rise course.

🔍 What It Does
This assistant lets users:
Ask natural-language questions about the textbook
Get clear, grounded answers based on actual content (no hallucinations)
Interact with a tutor modeled on Tony Bates’ tone and reasoning
Explore topics like course design, media selection, and online assessment

🎯 Who It’s For
Designed for:
Instructional designers and curriculum developers
Online educators and PD facilitators
EdTech teams exploring AI integration
Anyone using Teaching in a Digital Age in practice or training

🌐 Try It Out
🔗 Launch the assistant
🔗 See it embedded in a Rise course

🛠 Tech Stack
Streamlit – interactive UI and app runtime
LangChain – document parsing & retrieval
OpenAI GPT-4o – language model interface
FAISS – semantic search over textbook content
Render – fast, free-tier cloud hosting

📚 Why This Textbook?
Authored by Tony Bates, a leading voice in digital education
Openly licensed (CC BY)
Structured and modular — perfect for chunk-based retrieval
Widely used in instructional design courses and PD programs

📂 Project Structure
File	Purpose
app.py	Main app logic (Streamlit UI, user query flow)
vectorstore_utils.py	FAISS setup and embedding logic
teaching-in-a-digital-age.txt	Source document
initial_prompt.txt	System prompt shaping assistant behavior
requirements.txt	App dependencies
README.md	Project overview and usage guide

🧠 Prompt Design Philosophy
The assistant aims to teach, not just chat.
The prompt defines tone and logic with:
Clear instructional structure (definition → explanation → implication)
Minimal creativity, maximum groundedness
Educational tone aligned with Bates’ style

🚀 Local Setup (Optional)
bash
Copy
Edit
git clone https://github.com/YOUR_USERNAME/your-repo.git
cd your-repo
pip install -r requirements.txt
streamlit run app.py
🤝 Contribute or Connect
Got ideas or feedback? Fork the repo or connect via garyhills.dev.
