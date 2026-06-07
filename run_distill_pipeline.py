#!/usr/bin/env python3
"""
DeepSeek distillation pipeline for Qwen2.5-0.5B:
  1) build_distill_train.py  (DeepSeek API, CPU/any machine)
  2) qwen_ft.py --mode cot    (GPU cloud)
  3) ft_infer.py --mode cot   (GPU cloud)

Quick test (100 samples):
  python run_distill_pipeline.py --distill-limit 100 --merge-lora

Full run:
  python run_distill_pipeline.py --only-correct --merge-lora --workers 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek distill -> Qwen LoRA pipeline")
    parser.add_argument("--train", type=Path, default=Path("train.json"))
    parser.add_argument("--train-distill", type=Path, default=Path("train_distill.json"))
    parser.add_argument("--test", type=Path, default=Path("test.json"))
    parser.add_argument("--dev", type=Path, default=Path("dev.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen-distill-lora"))
    parser.add_argument("--submit", type=Path, default=Path("submit_distill.csv"))
    parser.add_argument("--teacher-model", type=str, default="deepseek-chat")
    parser.add_argument("--skip-distill", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--rebuild-distill", action="store_true")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--distill-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only-correct", action="store_true", help="Keep teacher-correct samples only")
    parser.add_argument("--on-wrong", choices=("template", "skip"), default="template")
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=768)
    args, extra = parser.parse_known_args()

    root = Path(__file__).resolve().parent
    py = sys.executable

    if not args.skip_distill:
        distill_cmd = [
            py,
            str(root / "build_distill_train.py"),
            "--train",
            str(args.train),
            "--output",
            str(args.train_distill),
            "--model",
            args.teacher_model,
            "--workers",
            str(args.workers),
            "--on-wrong",
            args.on_wrong,
        ]
        if args.distill_limit > 0:
            distill_cmd.extend(["--limit", str(args.distill_limit)])
        if args.only_correct:
            distill_cmd.append("--only-correct")
        if args.rebuild_distill:
            distill_cmd.append("--rebuild")
        run(distill_cmd)
    elif not args.train_distill.exists():
        raise FileNotFoundError(f"Missing {args.train_distill}; run distill step first")

    if not args.skip_train:
        train_cmd = [
            py,
            str(root / "qwen_ft.py"),
            "--mode",
            "cot",
            "--train",
            str(args.train_distill),
            "--output-dir",
            str(args.output_dir),
            "--max-length",
            str(args.max_length),
            "--epochs",
            str(args.epochs),
            "--dev-size",
            str(args.dev_size),
        ]
        if args.merge_lora:
            train_cmd.append("--merge-lora")
        train_cmd.extend(extra)
        run(train_cmd)

    ckpt = args.output_dir / "merged"
    if not (args.merge_lora and ckpt.exists()):
        ckpt = args.output_dir / "final"
        if not ckpt.exists():
            from ft_utils import find_latest_checkpoint

            ckpt = find_latest_checkpoint(args.output_dir)

    infer_cmd = [
        py,
        str(root / "ft_infer.py"),
        "--mode",
        "cot",
        "--checkpoint",
        str(ckpt),
        "--output-dir",
        str(args.output_dir),
        "--test",
        str(args.test),
        "--output",
        str(args.submit),
        "--max-new-tokens",
        "128",
    ]
    if args.dev.exists():
        infer_cmd.extend(["--dev", str(args.dev), "--report", str(args.output_dir / "dev_report.json")])
    elif (args.output_dir / "dev_holdout.json").exists():
        infer_cmd.extend([
            "--dev", str(args.output_dir / "dev_holdout.json"),
            "--report", str(args.output_dir / "dev_report.json"),
        ])
    run(infer_cmd)

    print("\nDistill pipeline finished.")
    print(f"  train_distill -> {args.train_distill}")
    print(f"  checkpoint    -> {ckpt}")
    print(f"  submit        -> {args.submit}")
    print(f"  dev report    -> {args.output_dir / 'dev_report.json'}")


if __name__ == "__main__":
    main()
