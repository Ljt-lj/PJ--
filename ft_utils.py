"""Shared helpers for LoRA fine-tuning and PEFT inference."""

from __future__ import annotations

import re
from pathlib import Path

from cot_core import format_submit_answer, normalize_question

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_INSTRUCTION = (
    "这是小学数学1-6年级的校内题目，无需进行分析，请直接输出数字答案，不带单位。"
)


def normalize_sample(row: dict) -> dict:
    instruction = str(row.get("instruction") or DEFAULT_INSTRUCTION).strip()
    question = normalize_question(row["question"])
    answer = str(row["answer"]).strip()
    return {
        "id": row.get("id"),
        "instruction": instruction,
        "question": question,
        "answer": answer,
    }


def build_chat_messages(instruction: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]


def build_training_prefix(tokenizer, instruction: str, question: str) -> str:
    messages = build_chat_messages(instruction, question)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


IM_END = "<|" + "im_end|>"
IM_START = "<|" + "im_start|>"

# Model may append a fake next turn after the answer, e.g. "37Human: ...".
ROLE_TAIL = re.compile(
    r"(?:Human|Assistant|用户)[:：].*$|" + re.escape(IM_START) + r".*$",
    re.IGNORECASE,
)


def _strip_role_hallucination(text: str) -> str:
    """Drop hallucinated follow-up chat; keep the leading answer intact."""
    text = ROLE_TAIL.sub("", text).strip()
    # Glued without space: "20000Human: ..."
    text = re.sub(r"Human:.*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"Assistant:.*$", "", text, flags=re.IGNORECASE).strip()
    return text


def format_ft_prediction(raw: str, question: str) -> str:
    """Format direct-answer SFT output (match Math_Solver infer.py)."""
    text = str(raw).strip()
    for stop in (IM_END, "<|endoftext|>", IM_START):
        text = text.split(stop)[0].strip()
    text = _strip_role_hallucination(text)
    text = text.replace("\n", " ").strip()
    if not text:
        return "0"
    return format_submit_answer(question, text)


def find_latest_checkpoint(output_dir: Path) -> Path:
    candidates = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* under {output_dir}")
    return candidates[-1]


def resolve_model_dir(model_id: str, cache_dir: Path) -> Path:
    """Return local path after ModelScope or HuggingFace download."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / model_id.replace("/", "--")
    if local.exists() and any(local.iterdir()):
        return local

    print(f"Downloading model: {model_id}")
    try:
        from modelscope import snapshot_download

        path = snapshot_download(model_id, cache_dir=str(cache_dir), revision="master")
        return Path(path)
    except Exception as exc:
        print(f"ModelScope download failed ({exc}), trying HuggingFace Hub...")
        from huggingface_hub import snapshot_download as hf_download

        path = hf_download(repo_id=model_id, cache_dir=str(cache_dir))
        return Path(path)
