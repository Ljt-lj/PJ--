#!/usr/bin/env python3
"""
Generate submit.csv with a LoRA fine-tuned Qwen2.5-0.5B checkpoint.

Example:
  python ft_infer.py --checkpoint output/qwen-lora/final --test test.json --output submit.csv
  python ft_infer.py --checkpoint output/qwen-lora/merged --dev dev.json --report dev_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_core import answers_equal, normalize_question
from ft_utils import (
    DEFAULT_INSTRUCTION,
    DEFAULT_MODEL_ID,
    build_chat_messages,
    clean_model_output,
    find_latest_checkpoint,
    normalize_sample,
    resolve_model_dir,
)


def require_gpu() -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for PEFT inference.", file=sys.stderr)
        sys.exit(1)


def load_model(base_dir: Path, checkpoint: Path):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True, use_fast=False)
    base = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if (checkpoint / "adapter_config.json").exists() or checkpoint.name.startswith("checkpoint"):
        model = PeftModel.from_pretrained(base, str(checkpoint))
    else:
        model = base
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def predict(model, tokenizer, instruction: str, question: str, max_new_tokens: int) -> str:
    messages = build_chat_messages(instruction, question)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def load_json_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_rows(
    rows: list[dict],
    *,
    model,
    tokenizer,
    max_new_tokens: int,
    has_labels: bool,
) -> tuple[dict[str, str], dict | None]:
    preds: dict[str, str] = {}
    correct = 0
    for row in tqdm(rows, desc="FT inference"):
        sid = str(row.get("id", len(preds)))
        instruction = str(row.get("instruction") or DEFAULT_INSTRUCTION)
        question = normalize_question(row["question"])
        raw = predict(model, tokenizer, instruction, question, max_new_tokens)
        pred = clean_model_output(raw, question)
        preds[sid] = pred
        if has_labels:
            gold = str(row["answer"]).strip()
            if answers_equal(pred, gold):
                correct += 1

    report = None
    if has_labels:
        total = len(rows)
        report = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }
    return preds, report


def write_submit(path: Path, preds: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ret"])
        for sid in sorted(preds, key=lambda x: int(x) if str(x).isdigit() else x):
            w.writerow([sid, preds[sid]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen2.5-0.5B LoRA")
    parser.add_argument("--checkpoint", type=Path, default=None, help="LoRA adapter or merged model dir")
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen-lora"))
    parser.add_argument("--base-model", type=Path, default=None, help="Base model dir (auto-detect if omitted)")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--test", type=Path, default=Path("test.json"))
    parser.add_argument("--dev", type=Path, default=None, help="Optional labeled JSON for accuracy")
    parser.add_argument("--output", type=Path, default=Path("submit.csv"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    require_gpu()

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(args.output_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"Checkpoint: {checkpoint}")

    base_dir = args.base_model or resolve_model_dir(args.model_id, args.cache_dir)
    model, tokenizer = load_model(base_dir, checkpoint)

    if args.dev and args.dev.exists():
        dev_rows = load_json_rows(args.dev)
        _, report = run_rows(
            dev_rows,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            has_labels=True,
        )
        print(f"Dev accuracy: {report['accuracy']:.2%} ({report['correct']}/{report['total']})")
        if args.report:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Report -> {args.report}")

    if args.test.exists():
        test_rows = load_json_rows(args.test)
        preds, _ = run_rows(
            test_rows,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            has_labels=False,
        )
        write_submit(args.output, preds)
        print(f"Submit -> {args.output} ({len(preds)} rows)")
    else:
        print(f"Skip test inference: {args.test} not found")


if __name__ == "__main__":
    main()
