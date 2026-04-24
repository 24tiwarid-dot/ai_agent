from __future__ import annotations

import json
from typing import Any

from agents.groq_utils import (
    groq_chat_with_retry,
    is_rate_limit_error,
    rate_limit_message,
    safe_content,
)

JUDGE_SYSTEM_PROMPT = """
You are a senior technology analyst and content quality evaluator.
Your job is to assess AI-generated tool comparison buying guides against a structured rubric.

RULES:
- Score each criterion independently on a scale of 1 to 5.
- Be strict but fair. Score 3 = competent. Score 5 = exceptional. Score 1 = fails.
- Base scores ONLY on the provided guide. Do not infer or assume anything absent.
- Provide specific, actionable reasoning for each score.
- Return ONLY valid JSON. No markdown, no preamble, no commentary outside JSON.
""".strip()


JUDGE_EVAL_PROMPT = """
Evaluate this AI-generated tool comparison buying guide.

=== CATEGORY INFO ===
Tool Category: {category}
Tools Compared: {tools_list}

=== GENERATED BUYING GUIDE ===
{guide_content}

=== RUBRIC - Score each 1-5 ===
1. COMPARISON BALANCE (1-5)
   1: Obvious bias toward one tool
   3: Mostly balanced but uneven depth
   5: All tools assessed with equal rigour and evidence

2. PRICING ACCURACY (1-5)
   1: No pricing or clearly incorrect
   3: Some pricing, incomplete
   5: All current pricing tiers accurately represented

3. FEATURE COVERAGE (1-5)
   1: Vague or missing key differentiators
   3: Core features covered, differentiators not highlighted
   5: Specific feature comparisons with differentiators called out

4. RECOMMENDATION CLARITY (1-5)
   1: No recommendation or completely unjustified
   3: Generic recommendation, weakly justified
   5: Clear use-case-specific recommendations with evidence

5. READABILITY & STRUCTURE (1-5)
   1: Unstructured, hard to compare tools
   3: Readable but inconsistent
   5: Well-structured, professional, easy to scan

=== REQUIRED OUTPUT - return ONLY this JSON ===
{{
  "scores": {{
    "comparison_balance":     {{"score": N, "reasoning": "..."}},
    "pricing_accuracy":       {{"score": N, "reasoning": "..."}},
    "feature_coverage":       {{"score": N, "reasoning": "..."}},
    "recommendation_clarity": {{"score": N, "reasoning": "..."}},
    "readability_structure":  {{"score": N, "reasoning": "..."}}
  }},
  "overall_score": N,
  "summary": "One paragraph overall assessment",
  "top_strength": "Best aspect of this guide",
  "top_improvement": "Most important thing to improve"
}}
""".strip()


def judge_agent(
    client: Any,
    guide_content: str,
    category: str,
    tools_list: list[str],
    model: str = "llama-3.1-8b-instant",
    log_step=None,
) -> dict[str, Any]:
    if log_step:
        log_step("Judge: evaluating guide against rubric...")

    eval_prompt = JUDGE_EVAL_PROMPT.format(
        category=category,
        tools_list=", ".join(tools_list),
        guide_content=guide_content,
    )

    response = groq_chat_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": eval_prompt},
        ],
        temperature=0.1,
        max_completion_tokens=2048,
        response_format={"type": "json_object"},
    )

    try:
        raw_text = safe_content(response)
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
    except json.JSONDecodeError:
        # Try extracting JSON object from mixed output.
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(clean[start : end + 1])
            except json.JSONDecodeError:
                result = {"error": "JSON parse failed", "raw": raw_text}
        else:
            result = {"error": "JSON parse failed", "raw": raw_text}

    if "error" in result:
        repair_prompt = (
            "Return valid JSON only for this judge schema:\n"
            '{"scores":{"comparison_balance":{"score":0,"reasoning":"..."},'
            '"pricing_accuracy":{"score":0,"reasoning":"..."},'
            '"feature_coverage":{"score":0,"reasoning":"..."},'
            '"recommendation_clarity":{"score":0,"reasoning":"..."},'
            '"readability_structure":{"score":0,"reasoning":"..."}},'
            '"overall_score":0,"summary":"...","top_strength":"...",'
            '"top_improvement":"..."}\n\n'
            f"Text:\n{raw_text}"
        )
        try:
            repaired = groq_chat_with_retry(
                client,
                model=model,
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0,
                response_format={"type": "json_object"},
                max_completion_tokens=1500,
            )
            result = json.loads(repaired.choices[0].message.content or "{}")
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit_error(exc):
                raise

    if log_step:
        overall = result.get("overall_score", "?")
        log_step(f"Judge: overall score = {overall}/5")
    return result
