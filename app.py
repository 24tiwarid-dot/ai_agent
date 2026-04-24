from __future__ import annotations

import json
import os
import time
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from agents.comparison_agent import comparison_agent
from agents.guide_writer import guide_writer_agent
from agents.judge_agent import judge_agent
from agents.groq_utils import is_rate_limit_error, rate_limit_message
from agents.tool_researcher import tool_researcher_agent


def init_client() -> Groq:
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is missing in environment.")
    return Groq(api_key=groq_key)


def extract_tool_names(comparison_matrix: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    for item in comparison_matrix.get("tools", []):
        name = str(item.get("tool", "")).strip()
        if name and name.lower() not in {t.lower() for t in tools}:
            tools.append(name)
    return tools


def main() -> None:
    st.set_page_config(
        page_title="AI Tool Comparison Guide Builder",
        page_icon="🤖",
        layout="wide",
    )
    st.title("🤖 AI Tool Comparison Guide Builder")
    st.caption("Minimal output mode")

    with st.sidebar:
        st.header("Configuration")
        model_name = st.text_input("Groq Model", value="llama-3.1-8b-instant")
        max_turns = st.slider("Agent max turns", min_value=3, max_value=12, value=12)
        st.markdown(
            "Required env vars in `.env`:\n\n"
            "- `GROQ_API_KEY`\n"
            "- `TAVILY_API_KEY`"
        )

    category = st.text_input(
        "Enter AI tool category",
        placeholder="e.g. coding assistants, writing tools, AI search",
    )
    run_pipeline = st.button("Generate Buying Guide", type="primary")

    if not run_pipeline:
        return

    if not category.strip():
        st.error("Please enter a category before running the pipeline.")
        return

    try:
        client = init_client()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to initialize Groq client: {exc}")
        return

    def log_step(msg: str) -> None:
        _ = msg

    try:
        with st.spinner("Running Agent 1: Tool Landscape Researcher..."):
            tool_landscape = tool_researcher_agent(
                client=client,
                category=category,
                model=model_name,
                max_turns=max_turns,
                log_step=log_step,
            )

        log_step("Agent 1 complete")
        time.sleep(5)

        with st.spinner("Running Agent 2: Comparison Agent..."):
            matrix = comparison_agent(
                client=client,
                tool_landscape_text=tool_landscape,
                category=category,
                model=model_name,
                max_turns=max_turns,
                log_step=log_step,
            )

        log_step("Agent 2 complete")
        time.sleep(5)

        with st.spinner("Running Agent 3: Guide Writer..."):
            guide_md = guide_writer_agent(
                client=client,
                comparison_matrix=matrix,
                category=category,
                model=model_name,
            )

        st.markdown(guide_md or "_No output_")
        if guide_md:
            st.download_button(
                "Download Guide (.md)",
                data=guide_md.encode("utf-8"),
                file_name=f"{category.strip().replace(' ', '_')}_buying_guide.md",
                mime="text/markdown",
            )

        log_step("Agent 3 complete")
        time.sleep(5)

        with st.spinner("Running Agent 4: LLM-as-Judge..."):
            tools_list = extract_tool_names(matrix)
            judge_result = judge_agent(
                client=client,
                guide_content=guide_md,
                category=category,
                tools_list=tools_list,
                model=model_name,
                log_step=log_step,
            )

        if "error" in judge_result:
            st.caption("Judge unavailable")
        else:
            st.caption(f"Overall Score: {judge_result.get('overall_score', '?')}/5")

    except Exception as exc:  # noqa: BLE001
        if is_rate_limit_error(exc):
            st.error(rate_limit_message(exc))
        else:
            st.error(f"Pipeline failed: {exc}")


if __name__ == "__main__":
    main()
