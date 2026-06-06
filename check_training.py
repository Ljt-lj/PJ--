#!/usr/bin/env python3
"""Analyze LoRA training convergence from trainer_state.json and data stats."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_log_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    return state.get("log_history", [])


def summarize_loss(history: list[dict]) -> dict:
    points = [
        (h["step"], h["loss"])
        for h in history
        if "loss" in h and "step" in h
    ]
    if not points:
        return {"error": "no loss entries in log_history"}

    steps, losses = zip(*points)
    n = len(losses)
    first_k = max(1, n // 10)
    last_k = max(1, n // 10)
    start_avg = sum(losses[:first_k]) / first_k
    end_avg = sum(losses[-last_k:]) / last_k
    best = min(losses)
    best_step = steps[losses.index(best)]

    # Simple plateau check: last 20% vs previous 20%
    seg = max(1, n // 5)
    prev_avg = sum(losses[-2 * seg : -seg]) / seg if n >= 2 * seg else end_avg
    tail_avg = sum(losses[-seg:]) / seg

    return {
        "steps_logged": n,
        "first_step": steps[0],
        "last_step": steps[-1],
        "loss_start_avg": round(start_avg, 4),
        "loss_end_avg": round(end_avg, 4),
        "loss_best": round(best, 4),
        "loss_best_step": best_step,
        "loss_drop": round(start_avg - end_avg, 4),
        "tail_vs_prev": round(tail_avg - prev_avg, 4),
        "converged": tail_avg <= prev_avg + 0.02 and end_avg < 1.5,
        "likely_underfit": end_avg > 1.0,
    }


def expected_steps(train_n: int, batch: int, accum: int, epochs: int) -> int:
    per_epoch = math.ceil(train_n / (batch * accum))
    return per_epoch * epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LoRA training convergence")
    parser.add_argument("--output-dir", type=Path, default=Path("output/qwen-lora"))
    parser.add_argument("--train", type=Path, default=Path("train.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--dev-size", type=int, default=0)
    args = parser.parse_args()

    state_path = args.output_dir / "trainer_state.json"
    if not state_path.exists():
        # checkpoints also store trainer_state.json
        cks = sorted(args.output_dir.glob("checkpoint-*"))
        if cks:
            state_path = cks[-1] / "trainer_state.json"

    print("=== Training convergence ===")
    if state_path.exists():
        history = load_log_history(state_path)
        summary = summarize_loss(history)
        for k, v in summary.items():
            print(f"  {k}: {v}")

        if summary.get("likely_underfit"):
            print("\n  [!] Final loss still > 1.0 — model may not have fully learned short answers.")
        if summary.get("converged"):
            print("\n  [OK] Loss plateaued — training ran to completion.")
        elif summary.get("tail_vs_prev", 0) > 0.05:
            print("\n  [!] Loss still rising at end — may need more epochs or lower LR.")
    else:
        print(f"  Missing {state_path}")
        print("  Run on cloud GPU after training, or copy output/qwen-lora/ here.")

    if args.train.exists():
        with args.train.open("r", encoding="utf-8") as f:
            n = len(json.load(f))
        train_n = max(0, n - args.dev_size)
        exp = expected_steps(train_n, args.batch_size, args.grad_accum, args.epochs)
        print("\n=== Schedule ===")
        print(f"  train samples: {train_n}")
        print(f"  expected steps ({args.epochs} epochs): {exp}")
        if state_path.exists():
            last = summarize_loss(load_log_history(state_path)).get("last_step")
            if last:
                pct = 100 * last / exp
                print(f"  last logged step: {last} ({pct:.0f}% of expected)")
                if last < exp * 0.9:
                    print("  [!] Training may have stopped early.")

    dev_report = args.output_dir / "dev_report.json"
    if dev_report.exists():
        rep = json.loads(dev_report.read_text(encoding="utf-8"))
        print("\n=== Dev accuracy (local) ===")
        print(f"  {rep.get('accuracy', '?')} ({rep.get('correct', '?')}/{rep.get('total', '?')})")

    merged = args.output_dir / "merged"
    final = args.output_dir / "final"
    print("\n=== Checkpoints ===")
    print(f"  merged/: {'yes' if merged.exists() else 'no'}")
    print(f"  final/:  {'yes' if final.exists() else 'no'}")
    ckpts = sorted(args.output_dir.glob("checkpoint-*"))
    print(f"  checkpoint-*: {len(ckpts)}" + (f" (latest {ckpts[-1].name})" if ckpts else ""))


if __name__ == "__main__":
    main()
