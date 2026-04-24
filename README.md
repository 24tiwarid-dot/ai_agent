# 🤖 AI Tool Comparison Guide Builder

A multi-agent AI app that automatically researches, compares, and generates a buying guide for any AI tool category.

🔗 **Live App:** https://aiagent-production-af54.up.railway.app

---

## What It Does

Enter any AI tool category (e.g. "coding assistants", "writing tools", "AI search") and the app will:

1. 🔍 **Agent 1 — Tool Researcher** — Searches the web for the top tools in that category
2. ⚖️ **Agent 2 — Comparison Agent** — Builds a feature-by-feature comparison matrix
3. ✍️ **Agent 3 — Guide Writer** — Writes a detailed buying guide in Markdown
4. 🧑‍⚖️ **Agent 4 — LLM Judge** — Scores the guide for quality, bias, and accuracy

---

## Tech Stack

- [Streamlit](https://streamlit.io) — Web UI
- [Groq](https://groq.com) — LLM inference (llama-3.1-8b-instant)
- [Tavily](https://tavily.com) — Real-time web search
- [Railway](https://railway.app) — Deployment

---

## Run Locally

### 1. Clone the repo
```bash
GROQ_API_KEY
TAVILY_API_KEY
### 4. Run the app
```bash
streamlit run app.py
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Get from [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | Get from [tavily.com](https://tavily.com) |

---

## Project Structure
ai_agent/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── railway.toml            # Railway deployment config
├── .gitignore
└── agents/
├── tool_researcher.py
├── comparison_agent.py
├── guide_writer.py
├── judge_agent.py
└── groq_utils.py