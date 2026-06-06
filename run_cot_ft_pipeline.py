#!/usr/bin/env python3
"""
CoT LoRA pipeline: build train_cot.json -> train -> dev eval -> submit.csv

Usage (GPU cloud):
  python run_cot_ft_pipeline.py --merge-lora

Fast start (template CoT labels, no Ollama):
  python run_cot_ft_pipeline.py --cot-source template --merge-lora

With Ollama CoT generation (better labels, needs local Ollama):
  python run_cot_ft_pipeline.py --cot-source auto --merge-lora
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
    parser = argparse.ArgumentParser(description="CoT LoRA fine-tune pipeline")
    parser.add_argument("--train", type=Path, default=Path("train.json"))
    parser.add_argument("--train-cot", type=Path, default=Path("train_cot.json"))
    parser.add_argument("--test", type=Path, default=Path("test.json"))
    parser.add_argument("--dev", type=Path, default=Path("dev.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen-cot-lora"))
    parser.add_argument("--submit", type=Path, default=Path("submit.csv"))
    parser.add_argument("--skip-build-cot", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--rebuild-cot", action="store_true")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--cot-source", choices=("auto", "ollama", "template"), default="template")
    parser.add_argument("--cot-limit", type=int, default=0, help="Limit CoT build samples (0=all)")
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    args, extra = parser.parse_known_args()

    root = Path(__file__).resolve().parent
    py = sys.executable

    if not args.skip_build_cot:
        build_cmd = [
            py,
            str(root / "build_cot_train.py"),
            "--train",
            str(args.train),
            "--output",
            str(args.train_cot),
            "--source",
            args.cot_source,
        ]
        if args.cot_limit > 0:
            build_cmd.extend(["--limit", str(args.cot_limit)])
        if args.rebuild_cot:
            build_cmd.append("--rebuild")
        run(build_cmd)
    elif not args.train_cot.exists():
        raise FileNotFoundError(f"Missing {args.train_cot}; run without --skip-build-cot")

    if not args.skip_train:
        train_cmd = [
            py,
            str(root / "qwen_ft.py"),
            "--mode",
            "cot",
            "--train",
            str(args.train_cot),
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
    ]
    if args.dev.exists():
        infer_cmd.extend(["--dev", str(args.dev), "--report", str(args.output_dir / "dev_report.json")])
    elif (args.output_dir / "dev_holdout.json").exists():
        infer_cmd.extend([
            "--dev", str(args.output_dir / "dev_holdout.json"),
            "--report", str(args.output_dir / "dev_report.json"),
        ])
    run(infer_cmd)

    print("\nCoT pipeline finished.")
    print(f"  train_cot  -> {args.train_cot}")
    print(f"  submit.csv -> {args.submit}")
    print(f"  dev report -> {args.output_dir / 'dev_report.json'}")


if __name__ == "__main__":
    main()
