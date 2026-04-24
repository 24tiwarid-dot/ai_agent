from __future__ import annotations

import json
import os
from typing import Any

from tavily import TavilyClient

from agents.agent_loop import run_agent_loop
from agents.groq_utils import (
    groq_chat_with_retry,
    is_rate_limit_error,
    rate_limit_message,
    safe_content,
)


COMPARISON_AGENT_SYSTEM_PROMPT = """
You are Agent 2: Comparison Agent.

Goal:
- Build a structured comparison matrix for the tool list.
- For each tool, gather PRICING, FEATURES, REVIEWS, and OVERVIEW evidence.
- Use fetch_tool_data repeatedly as needed.

Output requirements:
- Return ONLY valid JSON.
- JSON schema:
{
  "tools": [
    {
      "tool": "Tool name",
      "pricing": "concise pricing summary",
      "features": ["f1", "f2", "f3"],
      "review_summary": "balanced sentiment summary",
      "rating": "x.x/5 or N/A",
      "sources": ["url1", "url2"]
    }
  ],
  "missing_data_warnings": ["..."]
}
""".strip()


fetch_tool_data_decl = {
    "type": "function",
    "function": {
        "name": "fetch_tool_data",
        "description": (
            "Fetch pricing, features, and user review summary for a specific AI "
            "tool using live web search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the AI tool to research",
                },
                "data_type": {
                    "type": "string",
                    "description": "Type of data: PRICING, FEATURES, REVIEWS, OVERVIEW",
                },
            },
            "required": ["tool_name", "data_type"],
        },
    },
}


def _run_fetch_tool_data(tool_name: str, data_type: str) -> dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "tool": tool_name,
            "data_type": data_type,
            "error": "Missing TAVILY_API_KEY in environment.",
        }

    query = f"{tool_name} AI tool {data_type.lower()} 2026"
    import time
    time.sleep(1.0) # Rate limit protection for Tavily
    tavily_client = TavilyClient(api_key=api_key)
    try:
        response = tavily_client.search(
            query=query,
            max_results=5,
            search_depth="advanced",
        )
    except Exception as exc:  # noqa: BLE001
        return {"tool": tool_name, "data_type": data_type, "error": str(exc)}

    snippets: list[str] = []
    sources: list[str] = []
    for item in response.get("results", []):
        sources.append(item.get("url", "").strip())
        snippets.append(item.get("content", "").strip()[:320])

    return {
        "tool": tool_name,
        "data_type": data_type,
        "query": query,
        "sources": [url for url in sources if url],
        "content": ("\n".join([s for s in snippets if s]).strip() or "No content.")[:2500],
    }


def _extract_tools_from_text(tool_landscape_text: str) -> list[str]:
    lines = [ln.strip() for ln in tool_landscape_text.splitlines() if ln.strip()]
    extracted: list[str] = []
    for line in lines:
        cleaned = line
        # Skip empty lines
        if not cleaned:
            continue
        # Handle numbered formats: "1. Tool", "1) Tool", etc.
        if len(cleaned) >= 2 and cleaned[0].isdigit():
            for sep in [".", ")", ":"]:
                idx = cleaned.find(sep)
                if 0 < idx <= 3:
                    cleaned = cleaned[idx + 1:].strip()
                    break
        if cleaned.startswith(("-", "*", "•")):
            cleaned = cleaned[1:].strip()
        token = cleaned.replace("**", "").strip()
        token = token.split(" - ", 1)[0].split(":", 1)[0].strip()
        # Filter headings/sentences that are not tool names.
        if not token or "ranked list" in token.lower() or len(token.split()) > 5:
            continue
        if token.lower() not in {t.lower() for t in extracted}:
            extracted.append(token)
    return extracted[:8]


def comparison_agent(
    client: Any,
    tool_landscape_text: str,
    category: str,
    model: str = "llama-3.1-8b-instant",
    max_turns: int = 12,
    log_step=None,
) -> dict[str, Any]:
    tools = _extract_tools_from_text(tool_landscape_text)
    tools_hint = ", ".join(tools) if tools else "Infer tools from landscape text."
    initial_prompt = (
        f"Category: {category}\n"
        f"Candidate tools: {tools_hint}\n"
        "Build a comparison matrix with pricing, features, reviews, and rating. "
        "Call fetch_tool_data as needed and return strict JSON only."
    )

    def executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name != "fetch_tool_data":
            return {"error": f"Unknown tool: {name}"}
        tool_name = str(args.get("tool_name", "")).strip()
        data_type = str(args.get("data_type", "OVERVIEW")).strip().upper()
        return _run_fetch_tool_data(tool_name=tool_name, data_type=data_type)

    raw = run_agent_loop(
        client=client,
        system_prompt=COMPARISON_AGENT_SYSTEM_PROMPT,
        initial_prompt=initial_prompt,
        tools=[fetch_tool_data_decl],
        executor=executor,
        max_turns=max_turns,
        model=model,
        log_step=log_step,
    )

    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict) and parsed.get("tools"):
            return parsed
    except json.JSONDecodeError:
        # Try extracting the first JSON object from mixed text.
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = clean[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and parsed.get("tools"):
                    return parsed
            except json.JSONDecodeError:
                pass

    # Fallback: ask model to convert its own output to strict JSON.
    repair_prompt = (
        "Convert the following text into valid JSON only with this schema:\n"
        '{"tools":[{"tool":"...","pricing":"...","features":["..."],'
        '"review_summary":"...","rating":"...","sources":["..."]}],'
        '"missing_data_warnings":["..."]}\n\n'
        f"Text:\n{raw}"
    )
    try:
        repair = groq_chat_with_retry(
            client,
            model=model,
            messages=[{"role": "user", "content": repair_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_completion_tokens=1500,
        )
        repaired_text = safe_content(repair) or "{}"
        repaired = json.loads(repaired_text)
        if isinstance(repaired, dict) and repaired.get("tools"):
            return repaired
    except Exception as exc:  # noqa: BLE001
        if is_rate_limit_error(exc):
            raise

    fallback = _fallback_comparison_matrix(client, tools, category, model)
    if fallback.get("tools"):
        return fallback

    return {
        "tools": [],
        "missing_data_warnings": ["Failed to parse comparison JSON output."],
        "raw": raw,
    }


def _fallback_comparison_matrix(
    client: Any,
    tools: list[str],
    category: str,
    model: str,
) -> dict[str, Any]:
    """Deterministic backup path when agent loop does not finalize."""
    chosen = tools[:5]
    if not chosen:
        return {"tools": [], "missing_data_warnings": ["No tools discovered for fallback."]}

    evidence_chunks: list[str] = []
    for tool in chosen:
        overview = _run_fetch_tool_data(tool, "OVERVIEW")
        pricing = _run_fetch_tool_data(tool, "PRICING")
        features = _run_fetch_tool_data(tool, "FEATURES")
        reviews = _run_fetch_tool_data(tool, "REVIEWS")
        evidence_chunks.append(
            f"Tool: {tool}\n"
            f"Overview: {overview.get('content','')}\n"
            f"Pricing: {pricing.get('content','')}\n"
            f"Features: {features.get('content','')}\n"
            f"Reviews: {reviews.get('content','')}\n"
            f"Sources: {overview.get('sources',[])} {pricing.get('sources',[])} "
            f"{features.get('sources',[])} {reviews.get('sources',[])}\n"
        )

    prompt = (
        f"Category: {category}\n"
        f"Tools: {', '.join(chosen)}\n\n"
        "Build a strict JSON matrix:\n"
        '{"tools":[{"tool":"...","pricing":"...","features":["..."],'
        '"review_summary":"...","rating":"x.x/5 or N/A","sources":["..."]}],'
        '"missing_data_warnings":["..."]}\n\n'
        "Use only this evidence:\n"
        + "\n---\n".join(evidence_chunks)[:12000]
    )

    try:
        response = groq_chat_with_retry(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_completion_tokens=2048,
        )
        parsed = json.loads(safe_content(response) or "{}")
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:  # noqa: BLE001
        if is_rate_limit_error(exc):
            raise

    return {
        "tools": [],
        "missing_data_warnings": ["Fallback comparison matrix generation failed."],
    }
