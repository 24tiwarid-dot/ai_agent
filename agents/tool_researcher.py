from __future__ import annotations

import os
from typing import Any

from tavily import TavilyClient

from agents.agent_loop import run_agent_loop


TOOL_RESEARCHER_SYSTEM_PROMPT = """
You are Agent 1: Tool Landscape Researcher.

Goal:
- Discover the most relevant AI tools for the provided category.
- Use tavily_search_tool as needed to verify rankings and market relevance.
- Produce a ranked list of 5-8 tools with concise descriptions.

Rules:
- Prefer well-known and currently active tools.
- Avoid duplicates and obscure tools with weak evidence.
- Keep output clean and structured.
- Output only the final ranked list.
- No intro, no disclaimer, no commentary.
- Format: one line per tool -> "N. Tool Name - short description".
""".strip()


tavily_decl = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": (
            "Search the web for top AI tools in a given category, including product "
            "names, rankings, and key features."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to fetch",
                },
            },
            "required": ["query"],
        },
    },
}


def _run_tavily(query: str, max_results: int = 5) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Missing TAVILY_API_KEY in environment."

    import time
    time.sleep(1.0) # Rate limit protection for Tavily
    tavily_client = TavilyClient(api_key=api_key)
    try:
        response = tavily_client.search(
            query=query,
            max_results=max(1, min(max_results, 10)),
            search_depth="advanced",
        )
    except Exception as exc:  # noqa: BLE001
        return f"Tavily request failed: {exc}"

    lines: list[str] = []
    for item in response.get("results", []):
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        content = item.get("content", "").strip()[:320]
        lines.append(f"- {title}\n  URL: {url}\n  Snippet: {content}")
    joined = "\n".join(lines) if lines else "No results."
    return joined[:3000]


def tool_researcher_agent(
    client: Any,
    category: str,
    model: str = "llama-3.1-8b-instant",
    max_turns: int = 12,
    log_step=None,
) -> str:
    initial_prompt = (
        f"Identify the top 5-8 AI tools in the category: {category}.\n"
        "Return a ranked list with short descriptions."
    )

    def executor(name: str, args: dict[str, Any]) -> str:
        if name != "tavily_search_tool":
            return f"Unknown tool: {name}"
        query = str(args.get("query", "")).strip()
        max_results = int(args.get("max_results", 5))
        return _run_tavily(query=query, max_results=max_results)

    result = run_agent_loop(
        client=client,
        system_prompt=TOOL_RESEARCHER_SYSTEM_PROMPT,
        initial_prompt=initial_prompt,
        tools=[tavily_decl],
        executor=executor,
        max_turns=max_turns,
        model=model,
        log_step=log_step,
    )
    if _is_failed_research(result):
        return _fallback_landscape(category)
    return _clean_research_output(result)


def _is_failed_research(text: str) -> bool:
    lower = text.lower()
    return (
        "tavily request failed" in lower
        or "unable to complete the search" in lower
        or "rate limit" in lower
        or "quota" in lower
        or len(text.strip()) < 40
    )


def _fallback_landscape(category: str) -> str:
    lower = category.lower()
    if "code" in lower or "coding" in lower or "developer" in lower:
        tools = [
            "GitHub Copilot - Widely used AI coding assistant integrated with major IDEs.",
            "Cursor - AI-first code editor with project-aware editing and chat.",
            "Tabnine - Code completion assistant focused on privacy and enterprise controls.",
            "Codeium - Free-first coding assistant with broad language support.",
            "Amazon Q Developer - AWS-focused coding assistant for cloud workflows.",
            "Google Gemini Code Assist - Google ecosystem coding assistant for IDEs.",
        ]
    else:
        tools = [
            "ChatGPT - General-purpose AI assistant used across many workflows.",
            "Claude - Strong long-context assistant for analysis and writing.",
            "Gemini - Multi-modal assistant integrated with Google tools.",
            "Perplexity - AI answer engine focused on web-grounded responses.",
            "Microsoft Copilot - Productivity and enterprise-focused AI assistant.",
        ]
    lines = [f"{idx}. {item}" for idx, item in enumerate(tools, start=1)]
    return "\n".join(lines)


def _clean_research_output(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned: list[str] = []
    for line in lines:
        candidate = line.replace("**", "").strip()
        if len(candidate) < 4:
            continue
        if candidate.lower().startswith(("based on", "note:", "please note", "here is")):
            continue
        if candidate[0].isdigit() and "." in candidate[:4]:
            cleaned.append(candidate)
            continue
        if candidate.startswith(("-", "*")):
            item = candidate[1:].strip()
            cleaned.append(f"{len(cleaned) + 1}. {item}")
    if cleaned:
        return "\n".join(cleaned[:8])
    return text
