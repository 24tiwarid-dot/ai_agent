# AI Tool Comparison Guide Builder

A Streamlit multi-agent system that takes an AI tool category as input and generates:

1. Tool landscape research (top tools)
2. Structured comparison matrix (pricing, features, reviews)
3. Markdown buying guide with recommendation
4. LLM-as-Judge quality evaluation with rubric scoring

## Architecture

- `app.py`: Streamlit UI and pipeline orchestration
- `agents/tool_researcher.py`: Agent 1 (Tavily-powered tool discovery)
- `agents/comparison_agent.py`: Agent 2 (per-tool pricing/features/reviews)
- `agents/guide_writer.py`: Agent 3 (Markdown guide generation)
- `agents/judge_agent.py`: Agent 4 (LLM-as-Judge rubric scoring)
- `agents/agent_loop.py`: Shared agentic tool-calling loop

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL, enter a category (e.g. `coding assistants`), and click **Generate Buying Guide**.

## Notes

- The agentic loop allows the model to decide when to call tools and when to stop.
- Judge is a separate API call to reduce self-bias.
- If Tavily/Groq keys are missing, the app shows clear error messages.
