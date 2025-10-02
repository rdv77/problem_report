from __future__ import annotations

import os
from typing import Dict, Tuple
from openai import OpenAI


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return OpenAI(api_key=api_key)


def chat_complete(model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def chat_complete_ex(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, int]]:
    """Same as chat_complete but also returns usage tokens.

    Returns: (content, {prompt_tokens, completion_tokens, total_tokens})
    """
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    if usage is None:
        usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    else:
        # Some SDKs expose dict-like usage; ensure ints
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = int(getattr(usage, "completion_tokens", 0))
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens))
        usage_dict = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    return content, usage_dict

