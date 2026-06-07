"""DeepSeek API client for CoT distillation (OpenAI-compatible)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from openai import OpenAI

from cot_core import (
    build_system_prompt,
    extract_answer,
    get_user_prompt,
    normalize_question,
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
FAST_MODEL = "deepseek-chat"


def load_api_key(path: Path = Path("api_key.txt")) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    raise FileNotFoundError(
        "DeepSeek API key not found. Set DEEPSEEK_API_KEY or create api_key.txt"
    )


def build_cot_messages(question: str | list, *, compact: bool = True) -> list[dict[str, str]]:
    q = normalize_question(question)
    return [
        {"role": "system", "content": build_system_prompt(q, compact=compact)},
        {"role": "user", "content": get_user_prompt(q, compact=compact)},
    ]


def chat_content(
    question: str | list,
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    compact: bool = True,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    timeout: float = 120.0,
) -> str:
    key = api_key or load_api_key()
    client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
    messages = build_cot_messages(question, compact=compact)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                stream=False,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:
            last_error = exc
            time.sleep(min(retry_delay * (2**attempt), 60.0))

    raise RuntimeError(f"DeepSeek API failed after {max_retries} retries: {last_error}")


def chat_answer(
    question: str | list,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    compact: bool = True,
) -> str:
    return extract_answer(chat_content(question, api_key=api_key, model=model, compact=compact))
