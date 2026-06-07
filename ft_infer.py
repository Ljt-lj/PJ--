#!/usr/bin/env python3
"""Generate submit.csv with fine-tuned Qwen2.5-0.5B (no CSV header by default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_core import answers_equal, normalize_question
from ft_utils import (
    COT_DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_INSTRUCTION,
    DEFAULT_MODEL_ID,
    build_cot_training_prefix,
    build_training_prefix,
    find_latest_checkpoint,
    format_cot_prediction,
    format_ft_prediction,
    resolve_model_dir,
)


def require_gpu() -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for PEFT inference.", file=sys.stderr)
        sys.exit(1)


def load_model(checkpoint: Path, base_dir: Path | None = None):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if (checkpoint / "adapter_config.json").exists():
        if base_dir is None:
            raise ValueError("base_dir required for LoRA adapter checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True, use_fast=False)
        base = AutoModelForCausalLM.from_pretrained(
            base_dir, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, str(checkpoint))
        model.eval()
        return model, tokenizer

    if (checkpoint / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model.eval()
        return model, tokenizer

    raise FileNotFoundError(f"Unrecognized checkpoint: {checkpoint}")


@torch.inference_mode()
def predict(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def build_prompt(tokenizer, mode: str, instruction: str, question: str) -> str:
    if mode == "cot":
        return build_cot_training_prefix(tokenizer, question)
    return build_training_prefix(tokenizer, instruction, question)


def format_prediction(mode: str, raw: str, question: str) -> str:
    if mode == "cot":
        return format_cot_prediction(raw, question)
    return format_ft_prediction(raw, question)


def load_json_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_rows(
    rows: list[dict],
    *,
    model,
    tokenizer,
    mode: str,
    max_new_tokens: int,
    has_labels: bool,
) -> tuple[dict[str, str], dict | None]:
    preds: dict[str, str] = {}
    correct = 0
    for row in tqdm(rows, desc=f"FT inference ({mode})"):
        sid = str(row.get("id", len(preds)))
        instruction = str(row.get("instruction") or DEFAULT_INSTRUCTION)
        question = normalize_question(row["question"])
        prompt = build_prompt(tokenizer, mode, instruction, question)
        raw = predict(model, tokenizer, prompt, max_new_tokens)
        pred = format_prediction(mode, raw, question)
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


def write_submit(path: Path, preds: dict[str, str], *, with_header: bool) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        if with_header:
            f.write("id,ret\n")
        for sid in sorted(preds, key=lambda x: int(x) if str(x).isdigit() else x):
            f.write(f"{sid},{preds[sid]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen2.5-0.5B LoRA")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen-lora"))
    parser.add_argument("--mode", choices=("direct", "cot"), default="direct")
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--test", type=Path, default=Path("test.json"))
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submit.csv"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--with-header", action="store_true", help="Write CSV header row")
    args = parser.parse_args()

    if args.max_new_tokens is None:
        args.max_new_tokens = COT_DEFAULT_MAX_NEW_TOKENS if args.mode == "cot" else 32
    if args.mode == "cot" and args.output_dir == Path("output/qwen-lora"):
        ck = str(args.checkpoint or "")
        if "distill" in ck:
            args.output_dir = Path("output/qwen-distill-lora")
        else:
            args.output_dir = Path("output/qwen-cot-lora")

    require_gpu()

    checkpoint = args.checkpoint
    if checkpoint is None:
        merged = args.output_dir / "merged"
        checkpoint = merged if merged.exists() else args.output_dir / "final"
        if not checkpoint.exists():
            checkpoint = find_latest_checkpoint(args.output_dir)
    print(f"Mode: {args.mode}, max_new_tokens: {args.max_new_tokens}")
    print(f"Checkpoint: {checkpoint}")

    base_dir = args.base_model or resolve_model_dir(args.model_id, args.cache_dir)
    model, tokenizer = load_model(
        checkpoint, base_dir if (checkpoint / "adapter_config.json").exists() else None
    )

    if args.dev and args.dev.exists():
        dev_rows = load_json_rows(args.dev)
        _, report = run_rows(
            dev_rows, model=model, tokenizer=tokenizer, mode=args.mode,
            max_new_tokens=args.max_new_tokens, has_labels=True,
        )
        print(f"Dev accuracy: {report['accuracy']:.2%} ({report['correct']}/{report['total']})")
        if args.report:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Report -> {args.report}")

    if args.test.exists():
        test_rows = load_json_rows(args.test)
        preds, _ = run_rows(
            test_rows, model=model, tokenizer=tokenizer, mode=args.mode,
            max_new_tokens=args.max_new_tokens, has_labels=False,
        )
        write_submit(args.output, preds, with_header=args.with_header)
        print(f"Submit -> {args.output} ({len(preds)} rows, header={args.with_header})")
    else:
        print(f"Skip test inference: {args.test} not found")


if __name__ == "__main__":
    main()
