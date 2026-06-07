#!/usr/bin/env python3
"""LoRA fine-tune - aligned with Math_Solver baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from ft_utils import (
    COT_DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_ID,
    build_cot_training_prefix,
    build_training_prefix,
    cot_target_for_row,
    normalize_sample,
    resolve_model_dir,
)


def require_gpu() -> None:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {name} ({mem_gb:.1f} GB VRAM)")
        return

    print("ERROR: No GPU detected by PyTorch.", file=sys.stderr)
    sys.exit(1)


def load_train_rows(path: Path, dev_size: int, seed: int) -> tuple[list[dict], list[dict] | None]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = [normalize_sample(r) for r in raw]
    if dev_size <= 0:
        return rows, None
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    dev = [rows[i] for i in idx[:dev_size]]
    train = [rows[i] for i in idx[dev_size:]]
    return train, dev


def make_process_func(tokenizer, max_length: int, mode: str):
    def process(example: dict) -> dict | None:
        if mode == "cot":
            prefix = build_cot_training_prefix(tokenizer, example["question"])
            target = cot_target_for_row(example)
        else:
            prefix = build_training_prefix(tokenizer, example["instruction"], example["question"])
            target = example["answer"]

        prefix_ids = tokenizer(prefix, add_special_tokens=False)
        answer_ids = tokenizer(target, add_special_tokens=False)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id or pad_id

        answer_len = len(answer_ids["input_ids"]) + 1
        prefix_len = len(prefix_ids["input_ids"])
        if answer_len >= max_length:
            return None
        if prefix_len + answer_len > max_length:
            budget = max_length - answer_len
            prefix_ids = {k: v[-budget:] for k, v in prefix_ids.items()}
            prefix_len = len(prefix_ids["input_ids"])

        input_ids = prefix_ids["input_ids"] + answer_ids["input_ids"] + [eos_id]
        attention_mask = prefix_ids["attention_mask"] + answer_ids["attention_mask"] + [1]
        labels = [-100] * prefix_len + answer_ids["input_ids"] + [eos_id]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return process


def maybe_add_swanlab_callback(enable: bool):
    if not enable:
        return []
    try:
        from swanlab.integration.huggingface import SwanLabCallback
    except ImportError as exc:
        raise SystemExit("Install swanlab first: pip install swanlab") from exc

    return [
        SwanLabCallback(
            project="PJ2-math-lora",
            experiment_name="Qwen2.5-0.5B-Instruct",
            config={"model": DEFAULT_MODEL_ID},
        )
    ]


def merge_and_save(base_dir: Path, adapter_dir: Path, merged_dir: Path) -> None:
    from peft import PeftModel

    print(f"Merging LoRA -> {merged_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True, use_fast=False)
    base = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved: {merged_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-0.5B on train.json")
    parser.add_argument("--train", type=Path, default=Path("train.json"))
    parser.add_argument("--mode", choices=("direct", "cot"), default="direct")
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen-lora"))
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--swanlab", action="store_true")
    args = parser.parse_args()

    require_gpu()
    if not args.train.exists():
        raise FileNotFoundError(f"Missing {args.train}")

    max_length = args.max_length
    if max_length is None:
        max_length = COT_DEFAULT_MAX_LENGTH if args.mode == "cot" else 512
    if args.mode == "cot" and args.output_dir == Path("output/qwen-lora"):
        stem = args.train.stem.lower()
        if "distill" in stem:
            args.output_dir = Path("output/qwen-distill-lora")
        else:
            args.output_dir = Path("output/qwen-cot-lora")

    train_rows, dev_rows = load_train_rows(args.train, args.dev_size, args.seed)
    print(f"Mode: {args.mode}, max_length: {max_length}")
    print(f"Train samples: {len(train_rows)}")
    if dev_rows is not None:
        print(f"Dev holdout:   {len(dev_rows)}")

    model_dir = resolve_model_dir(args.model_id, args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", torch_dtype=dtype, trust_remote_code=True
    )
    model.enable_input_require_grads()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    process = make_process_func(tokenizer, max_length, args.mode)
    processed = [x for x in (process(r) for r in train_rows) if x is not None]
    skipped = len(train_rows) - len(processed)
    if skipped:
        print(f"Skipped {skipped} samples (too long even after prefix trim)")
    train_ds = Dataset.from_list(processed)

    dev_ds = None
    if dev_rows is not None:
        dev_processed = [x for x in (process(r) for r in dev_rows) if x is not None]
        dev_ds = Dataset.from_list(dev_processed)
        print(f"Dev eval set:  {len(dev_processed)} samples")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if dev_rows is not None:
        (args.output_dir / "dev_holdout.json").write_text(
            json.dumps(dev_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="epoch" if dev_ds is not None else "no",
        per_device_eval_batch_size=args.batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=maybe_add_swanlab_callback(args.swanlab),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    print(f"Training done. Adapter saved: {final_dir}")

    if args.merge_lora:
        merge_and_save(model_dir, final_dir, args.output_dir / "merged")


if __name__ == "__main__":
    main()
