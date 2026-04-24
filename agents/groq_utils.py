from __future__ import annotations

import re
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------

def rate_limit_message(exc: Exception) -> str:
    text = str(exc)
    match = re.search(r"Please try again in ([^\.]+(?:\.[0-9]+s)?)", text)
    wait_hint = f" Try again in {match.group(1)}." if match else ""
    return (
        "Groq rate limit reached (quota exceeded for current plan)."
        f"{wait_hint} You can reduce turns/model size or upgrade Groq tier."
    )


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate_limit_exceeded" in text or "rate limit" in text or "429" in text


def is_tool_validation_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "tool call validation failed" in text or "tool_use_failed" in text


def safe_content(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Retry wrapper for Groq API calls
# ---------------------------------------------------------------------------

def _extract_retry_seconds(exc: Exception) -> float:
    """Try to parse retry-after hint from the error message."""
    text = str(exc)
    # Match patterns like "Please try again in 1m12.345s" or "12.5s" or "854ms"
    m_ms = re.search(r"try again in (\d+(?:\.\d+)?)ms", text, re.IGNORECASE)
    if m_ms:
        return float(m_ms.group(1)) / 1000.0
        
    m = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", text, re.IGNORECASE)
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    return 0.0


def groq_chat_with_retry(
    client: Any,
    *,
    max_retries: int = 5,
    base_delay: float = 20.0,
    **chat_kwargs: Any,
) -> Any:
    """Call client.chat.completions.create with retry-on-rate-limit logic.

    Parameters
    ----------
    client : groq.Groq
        Initialised Groq client.
    max_retries : int
        How many times to retry on rate-limit errors.
    base_delay : float
        Default wait time when the server doesn't tell us how long to wait.
    **chat_kwargs
        All keyword arguments forwarded to ``client.chat.completions.create``.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):  # 1 initial + max_retries
        try:
            return client.chat.completions.create(**chat_kwargs)
        except Exception as exc:  # noqa: BLE001
            if not is_rate_limit_error(exc):
                raise
            last_exc = exc
            if attempt > max_retries:
                raise
            # Wait the server-suggested time, or exponential back-off.
            wait = _extract_retry_seconds(exc) or (base_delay * attempt)
            wait = min(wait, 120.0)  # cap at 2 minutes
            logger.warning(
                "Rate limit hit (attempt %d/%d). Waiting %.1fs before retry...",
                attempt, max_retries + 1, wait,
            )
            time.sleep(wait)

    # Should never reach here, but just in case.
    if last_exc:
        raise last_exc
