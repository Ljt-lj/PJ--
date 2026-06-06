"""Ollama local inference client for Qwen2.5-0.5B."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal

from cot_core import (
    BEST_PROMPT_MODE,
    BEST_TEMPERATURE,
    BEST_TOP_P,
    build_direct_messages,
    build_system_prompt,
    extract_answer,
    get_user_prompt,
    normalize_question,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:0.5b-instruct"

COT_TEMPERATURE = BEST_TEMPERATURE
COT_TOP_P = BEST_TOP_P
COT_NUM_CTX = 2048
COT_NUM_PREDICT = 96
COT_REPEAT_PENALTY = 1.1
COT_STOP = ["\n\n", "问题：", "示例："]

PromptMode = Literal["direct", "compact", "full"]


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    temperature: float = COT_TEMPERATURE
    top_p: float = COT_TOP_P
    num_ctx: int = COT_NUM_CTX
    repeat_penalty: float = COT_REPEAT_PENALTY
    num_predict: int = COT_NUM_PREDICT
    prompt_mode: PromptMode = BEST_PROMPT_MODE  # type: ignore[assignment]
    compact_prompt: bool = True
    timeout: float = 180.0


def build_options(config: OllamaConfig) -> dict:
    opts = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_ctx": config.num_ctx,
        "repeat_penalty": config.repeat_penalty,
        "num_predict": config.num_predict,
    }
    if COT_STOP:
        opts["stop"] = list(COT_STOP)
    return opts


def build_messages(question: str | list, config: OllamaConfig) -> list[dict[str, str]]:
    q = normalize_question(question)
    if config.prompt_mode == "direct":
        system, user = build_direct_messages(q)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    compact = config.prompt_mode == "compact" or config.compact_prompt
    return [
        {"role": "system", "content": build_system_prompt(q, compact=compact)},
        {"role": "user", "content": get_user_prompt(q, compact=compact)},
    ]


def chat_content(
    question: str | list,
    *,
    config: OllamaConfig,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    payload = {
        "model": config.model,
        "messages": build_messages(question, config),
        "stream": False,
        "options": build_options(config),
    }

    url = f"{config.base_url.rstrip('/')}/api/chat"
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=config.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "")
            return content.strip()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            time.sleep(min(retry_delay * (2**attempt), 30))

    raise RuntimeError(f"Ollama call failed after {max_retries} retries: {last_error}")


def chat(
    question: str | list,
    *,
    config: OllamaConfig,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    content = chat_content(question, config=config, max_retries=max_retries, retry_delay=retry_delay)
    return extract_answer(content)


def list_models(base_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in data.get("models", [])]
