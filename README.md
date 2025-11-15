Study Buddy AI — Personalized Study Planner Agent

A multi-agent LLM-powered study planner that generates personalized learning plans, follow-up questions, quizzes, resource recommendations, memory persistence, and full evaluation logs.

This project is built for the Kaggle x Google AI Agents Intensive – Capstone Project (Nov 2025).

⭐ Features

✔ Multi-agent pipeline (Planning Agent, Resource Agent, Quiz Agent)
✔ LLM-powered using Groq (LLaMA 3.1 8B Instant)
✔ In-memory session system
✔ Long-term memory (memory.json)
✔ Observability: structured logs (logs/log.jsonl)
✔ Safe tool calling pattern
✔ Evaluation using custom test framework
✔ Extensible search tool (SerpAPI optional)
✔ Clean code structure & easy to run

📁 Folder Structure
Study-Buddy/
│─ agent.py
│─ tools.py
│─ memory_manager.py
│─ logger.py
│─ demo_runner.py
│─ evaluator.py
│─ requirements.txt
│─ README.md
│
├── screenshots/
│   ├── terminal_output.png
│   ├── folder_structure.png
│   ├── evaluation_result.png
│
└── logs/ (ignored in git)

🚀 Running the Project (Local Setup)
1. Clone the repo
git clone https://github.com/<your-username>/study-buddy-agent
cd study-buddy-agent

2. Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
python -m pip install -r requirements.txt

4. Export your GROQ API Key
export GROQ_API_KEY="sk-groq-xxxx"

5. Choose LLM model
export STUDY_BUDDY_MODEL="llama-3.1-8b-instant"

6. Run the Demo
python demo_runner.py


Output will generate:

Personalized study plan

Quiz

Memory snapshot (memory.json)

Logs (logs/log.jsonl)

🧪 Running Evaluation
python evaluator.py


Creates:

evaluation_result_<timestamp>.json

🧰 Optional: Enable real search (SerpAPI)
python -m pip install google-search-results
export SERPAPI_KEY="your-serpapi-key"

🏆 Kaggle Capstone Track

Track: Concierge Agents
Category: Personalized Productivity Agents
