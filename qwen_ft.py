#!/usr/bin/env python3
"""
LoRA fine-tune Qwen2.5-0.5B-Instruct on train.json (competition baseline style).

Requires GPU (NVIDIA CUDA or AMD ROCm). CPU-only training is not supported in practice.

Example (cloud GPU):
  pip install -r requirements-ft.txt
  python qwen_ft.py --train train.json --output-dir output/qwen-lora

Resume / customize:
  python qwen_ft.py --output-dir output/qwen-lora --resume-from-checkpoint output/qwen-lora/checkpoint-1000
  python qwen_ft.py --epochs 3 --batch-size 8 --merge-lora
"""

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

from ft_utils import DEFAULT_MODEL_ID, build_chat_messages, normalize_sample, resolve_model_dir


def require_gpu() -> None:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {name} ({mem_gb:.1f} GB VRAM)")
        return

    print("ERROR: No GPU detected by PyTorch (torch.cuda.is_available()=False).", file=sys.stderr)
    print(f"  torch={torch.__version__}, torch.version.cuda={torch.version.cuda}", file=sys.stderr)
    print(
        "Run:  python check_gpu.py\n"
        "NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "AMD ROCm (魔搭 DSW): do NOT install cu121; use image PyTorch or --system-site-packages venv.\n"
        "      nvidia-smi will fail on AMD — use rocm-smi instead.",
        file=sys.stderr,
    )
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


def make_process_func(tokenizer, max_length: int):
    def process(example: dict) -> dict:
        messages = build_chat_messages(example["instruction"], example["question"])
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        answer = example["answer"]
        prefix_ids = tokenizer(prefix, add_special_tokens=False)
        answer_ids = tokenizer(answer, add_special_tokens=False)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        input_ids = prefix_ids["input_ids"] + answer_ids["input_ids"] + [pad_id]
        attention_mask = prefix_ids["attention_mask"] + answer_ids["attention_mask"] + [1]
        labels = [-100] * len(prefix_ids["input_ids"]) + answer_ids["input_ids"] + [pad_id]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

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
        import swanlab
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
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen-lora"))
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dev-size", type=int, default=0, help="Hold out N samples (not used in loss)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--merge-lora", action="store_true", help="Merge best/latest checkpoint after train")
    parser.add_argument("--swanlab", action="store_true", help="Enable SwanLab experiment logging")
    args = parser.parse_args()

    require_gpu()

    if not args.train.exists():
        raise FileNotFoundError(f"Missing {args.train}")

    train_rows, dev_rows = load_train_rows(args.train, args.dev_size, args.seed)
    print(f"Train samples: {len(train_rows)}")
    if dev_rows is not None:
        print(f"Dev holdout:   {len(dev_rows)} (excluded from training)")

    model_dir = resolve_model_dir(args.model_id, args.cache_dir)
    print(f"Model dir: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
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

    process = make_process_func(tokenizer, args.max_length)
    train_ds = Dataset.from_list([process(r) for r in train_rows])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if dev_rows is not None:
        (args.output_dir / "dev_holdout.json").write_text(
            json.dumps(dev_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
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
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=maybe_add_swanlab_callback(args.swanlab),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    print(f"Training done. Adapter saved: {final_dir}")

    if args.merge_lora:
        merged_dir = args.output_dir / "merged"
        merge_and_save(model_dir, final_dir, merged_dir)


if __name__ == "__main__":
    main()
