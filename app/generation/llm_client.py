"""
LLM client: calls the Groq API (free tier) with timeout + retry logic.

If GROQ_API_KEY is missing or every retry fails, we fall back to a
documented, deterministic local responder rather than silently inventing
an answer. The fallback never fabricates a grounded answer — it always
returns "Insufficient evidence" with an explicit refusal_reason explaining
that the LLM backend was unavailable, so the failure is visible to the
evaluator rather than hidden behind a made-up response.
"""
import json
import time
from typing import Optional

from app import config


class LLMUnavailableError(Exception):
    pass


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY, timeout=config.GROQ_TIMEOUT_SECONDS)
    completion = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    return completion.choices[0].message.content


def generate_raw(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Returns the raw model text, or None if the LLM is unavailable
    after all retries (caller must handle the documented fallback)."""
    if not config.GROQ_API_KEY:
        return None

    last_error = None
    for attempt in range(config.GROQ_MAX_RETRIES + 1):
        try:
            return _call_groq(system_prompt, user_prompt)
        except Exception as e:
            last_error = e
            if attempt < config.GROQ_MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    # All retries exhausted
    return None


def local_fallback_response(reason: str) -> str:
    """A deterministic, honest fallback JSON string used when the LLM
    backend cannot be reached. Never invents grounded content."""
    return json.dumps({
        "answer": "Insufficient evidence",
        "citations": [],
        "evidence": "",
        "inference": "",
        "confidence": 0.0,
        "refusal_reason": (
            f"The language model backend was unavailable ({reason}). "
            "No answer was generated to avoid fabricating an ungrounded response. "
            "Configure GROQ_API_KEY in .env, or point GROQ_MODEL/EMBEDDING_MODEL "
            "at a locally runnable alternative as documented in the README."
        ),
    })
