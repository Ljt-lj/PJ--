#!/usr/bin/env python3
"""
One-click pipeline: LoRA train -> dev eval -> test submit.csv

Usage (on a GPU cloud machine):
  pip install -r requirements-ft.txt
  python run_ft_pipeline.py

Pass through training args:
  python run_ft_pipeline.py --epochs 3 --merge-lora --dev-size 500
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
    parser = argparse.ArgumentParser(description="Full fine-tune pipeline")
    parser.add_argument("--train", type=Path, default=Path("train.json"))
    parser.add_argument("--test", type=Path, default=Path("test.json"))
    parser.add_argument("--dev", type=Path, default=Path("dev.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen-lora"))
    parser.add_argument("--submit", type=Path, default=Path("submit.csv"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--dev-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    args, extra = parser.parse_known_args()

    root = Path(__file__).resolve().parent
    py = sys.executable

    if not args.skip_train:
        train_cmd = [
            py,
            str(root / "qwen_ft.py"),
            "--train",
            str(args.train),
            "--output-dir",
            str(args.output_dir),
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
        "--checkpoint",
        str(ckpt),
        "--output-dir",
        str(args.output_dir),
        "--test",
        str(args.test),
        "--output",
        str(args.submit),
    ]
    if args.dev.exists():
        infer_cmd.extend(["--dev", str(args.dev), "--report", str(args.output_dir / "dev_report.json")])
    run(infer_cmd)

    print("\nPipeline finished.")
    print(f"  submit.csv -> {args.submit}")
    if args.dev.exists():
        print(f"  dev report -> {args.output_dir / 'dev_report.json'}")


if __name__ == "__main__":
    main()
