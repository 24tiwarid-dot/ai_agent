from __future__ import annotations

import json
from typing import Any

from agents.groq_utils import (
    groq_chat_with_retry,
    is_rate_limit_error,
    rate_limit_message,
    safe_content,
)

GUIDE_WRITER_SYSTEM_PROMPT = """
You are Agent 3: Guide Writer.

Write a polished, balanced, structured buying guide in Markdown.
Required sections:
1) Title + short intro
2) Tool profiles
3) Side-by-side comparison table
4) Use-case recommendations
5) Final recommendation with justification
6) Notes on missing/uncertain data (if any)

Tone: neutral, practical, evidence-based.
""".strip()


def guide_writer_agent(
    client: Any,
    comparison_matrix: dict[str, Any],
    category: str,
    model: str = "llama-3.1-8b-instant",
) -> str:
    matrix_json = json.dumps(comparison_matrix, ensure_ascii=True, indent=2)
    prompt = f"""
Create a complete buying guide for AI tools in category: {category}

Use this comparison matrix JSON:
{matrix_json}

Output Markdown only.
""".strip()

    response = groq_chat_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": GUIDE_WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        max_completion_tokens=2048,
    )
    return safe_content(response).strip()
