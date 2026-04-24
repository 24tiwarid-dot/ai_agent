from __future__ import annotations

import json
import time
from typing import Any, Callable

from agents.groq_utils import (
    groq_chat_with_retry,
    is_rate_limit_error,
    is_tool_validation_error,
    rate_limit_message,
)

ExecutorFn = Callable[[str, dict[str, Any]], Any]
LoggerFn = Callable[[str], None]
MAX_TOOL_CONTENT_CHARS = 2000
MAX_HISTORY_CHARS = 12000

# Small inter-turn delay (seconds) to stay under Groq RPM limits.
INTER_TURN_DELAY = 5.0


def _truncate_for_transport(value: Any, max_chars: int = MAX_TOOL_CONTENT_CHARS) -> Any:
    """Keep tool responses compact to avoid oversized LLM requests."""
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"
    if isinstance(value, list):
        trimmed: list[Any] = []
        used = 0
        for item in value:
            compact_item = _truncate_for_transport(item, max_chars=max_chars)
            text = json.dumps(compact_item, ensure_ascii=True)
            if used + len(text) > max_chars:
                break
            trimmed.append(compact_item)
            used += len(text)
        return trimmed
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        used = 0
        for key, item in value.items():
            compact_item = _truncate_for_transport(item, max_chars=max_chars)
            text = json.dumps({key: compact_item}, ensure_ascii=True)
            if used + len(text) > max_chars:
                compact["truncated"] = True
                break
            compact[key] = compact_item
            used += len(text)
        return compact
    return value


def _prune_messages(messages: list[dict[str, Any]], max_chars: int = MAX_HISTORY_CHARS) -> list[dict[str, Any]]:
    """Keep recent context while dropping oldest tool chatter safely."""
    if len(messages) <= 2:
        return messages

    compact = list(messages)
    while len(json.dumps(compact, ensure_ascii=True)) > max_chars and len(compact) > 4:
        # Remove the oldest non-system/non-initial user message.
        del compact[2]
        # Also remove any subsequent 'tool' role messages to avoid orphaning them.
        while len(compact) > 2 and compact[2].get("role") == "tool":
            del compact[2]
    return compact


def run_agent_loop(
    client: Any,
    system_prompt: str,
    initial_prompt: str,
    tools: list[dict[str, Any]],
    executor: ExecutorFn,
    max_turns: int = 12,
    model: str = "llama-3.1-8b-instant",
    log_step: LoggerFn | None = None,
) -> str:
    """Generic Groq tool-calling loop used by research/comparison agents."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_prompt},
    ]
    last_assistant_text = ""

    for turn in range(1, max_turns + 1):
        if log_step:
            log_step(f"Turn {turn}/{max_turns}: generating model response...")

        # Rate-limit-safe delay between turns.
        if turn > 1:
            time.sleep(INTER_TURN_DELAY)

        pruned_messages = _prune_messages(messages)
        try:
            response = groq_chat_with_retry(
                client,
                model=model,
                messages=pruned_messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_completion_tokens=1024,
            )
        except Exception as exc:  # noqa: BLE001
            if is_tool_validation_error(exc):
                if log_step:
                    log_step("Tool validation mismatch detected. Falling back to text-only turn.")
                text_only = groq_chat_with_retry(
                    client,
                    model=model,
                    messages=pruned_messages
                    + [
                        {
                            "role": "user",
                            "content": (
                                "Do not call any tool. Continue using only prior tool outputs "
                                "already present in this chat and provide the best final answer."
                            ),
                        }
                    ],
                    temperature=0.2,
                    max_completion_tokens=1024,
                )
                content = (text_only.choices[0].message.content or "").strip()
                return content or "Partial output."
            raise

        assistant_message = response.choices[0].message
        if assistant_message.content:
            last_assistant_text = assistant_message.content
        function_calls = assistant_message.tool_calls or []

        # Prevent token explosion by limiting concurrent tool calls
        if len(function_calls) > 4:
            if log_step:
                log_step(f"Model requested {len(function_calls)} tools. Truncating to 4 to prevent rate limits.")
            function_calls = function_calls[:4]

        if not function_calls:
            if log_step:
                log_step("Model stopped requesting tools and returned final output.")
            return (assistant_message.content or "").strip()

        # Preserve the model turn in conversation history.
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tc.model_dump() for tc in function_calls],
            }
        )

        for function_call in function_calls:
            try:
                args = json.loads(function_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if log_step:
                log_step(f"Executing tool: {function_call.function.name}({args})")
            result = executor(function_call.function.name, args)
            compact_result = _truncate_for_transport(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": function_call.id,
                    "name": function_call.function.name,
                    "content": json.dumps({"result": compact_result}, ensure_ascii=True),
                }
            )

    if log_step:
        log_step("Max turns reached. Returning partial output.")
    if last_assistant_text:
        return last_assistant_text.strip()
    return "Max turns reached - returning partial result."
