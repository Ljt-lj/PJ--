"""
Evaluate CoT predictions on a dev split from train.json.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from cot_core import (
    answers_equal,
    classify_question,
    normalize_question,
)
from ollama_client import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaConfig,
    chat,
)
from cot_infer import load_completed_ids, write_submission

TRAIN_URL = "https://github.com/AI-FDU/Math_Solver/raw/main/train.json"


def download_train(path: Path) -> None:
    print(f"Downloading {TRAIN_URL} -> {path}")
    urllib.request.urlretrieve(TRAIN_URL, path)
    print(f"Saved ({path.stat().st_size // 1024} KB)")


def load_train(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def sample_id(row: dict, index: int) -> str:
    return str(row.get("id", index))


def build_dev_split(
    train_data: list[dict],
    *,
    size: int,
    seed: int,
    dev_path: Path,
) -> list[dict]:
    if size <= 0 or size > len(train_data):
        raise ValueError(f"dev-size must be in 1..{len(train_data)}, got {size}")

    rng = random.Random(seed)
    indices = list(range(len(train_data)))
    rng.shuffle(indices)
    dev = [train_data[i] for i in indices[:size]]

    dev_path.parent.mkdir(parents=True, exist_ok=True)
    with dev_path.open("w", encoding="utf-8") as f:
        json.dump(dev, f, ensure_ascii=False, indent=2)
    print(f"Dev split: {size} samples (seed={seed}) -> {dev_path}")
    return dev


def load_dev(dev_path: Path) -> list[dict]:
    with dev_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_inference(
    dev_data: list[dict],
    *,
    config: OllamaConfig,
    predictions_path: Path,
) -> dict[str, str]:
    results: dict[str, str] = {}
    if predictions_path.exists():
        raw = load_completed_ids(predictions_path)
        results = {str(k): v for k, v in raw.items()}

    pending: list[tuple[str, dict]] = []
    for i, row in enumerate(dev_data):
        sid = sample_id(row, i)
        if sid not in results:
            pending.append((sid, row))

    print(f"Inference: total={len(dev_data)}, done={len(results)}, pending={len(pending)}")

    for sid, row in tqdm(pending, desc="Eval inference"):
        question = normalize_question(row["question"])
        pred = chat(question, config=config)
        results[sid] = pred
        int_results = {int(k): v for k, v in results.items() if str(k).isdigit()}
        write_submission(predictions_path, int_results)

    int_results = {int(k): v for k, v in results.items() if str(k).isdigit()}
    write_submission(predictions_path, int_results)
    return {str(k): v for k, v in int_results.items()}


def score_predictions(
    dev_data: list[dict],
    predictions: dict[str, str],
) -> tuple[dict, list[dict]]:
    correct = 0
    by_tag: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    errors: list[dict] = []

    for i, row in enumerate(dev_data):
        sid = sample_id(row, i)
        gold = str(row["answer"]).strip()
        pred = predictions.get(sid, "")
        ok = answers_equal(pred, gold)
        tags = classify_question(row["question"])

        if ok:
            correct += 1
        else:
            errors.append(
                {
                    "id": sid,
                    "question": normalize_question(row["question"]),
                    "gold": gold,
                    "pred": pred,
                    "tags": tags,
                }
            )

        for tag in tags:
            by_tag[tag]["total"] += 1
            if ok:
                by_tag[tag]["correct"] += 1

    total = len(dev_data)
    report = {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "by_tag": {
            tag: {
                **stats,
                "accuracy": round(stats["correct"] / stats["total"], 4)
                if stats["total"]
                else 0.0,
            }
            for tag, stats in sorted(by_tag.items())
        },
    }
    return report, errors


def print_report(report: dict) -> None:
    print("\n=== Evaluation Report ===")
    print(f"Total:    {report['total']}")
    print(f"Correct:  {report['correct']}")
    print(f"Wrong:    {report['wrong']}")
    print(f"Accuracy: {report['accuracy']:.2%}")

    print("\n--- By question type ---")
    for tag, stats in report["by_tag"].items():
        print(
            f"  {tag:14s}  {stats['correct']:4d}/{stats['total']:4d}  "
            f"({stats['accuracy']:.2%})"
        )


def print_errors(errors: list[dict], limit: int) -> None:
    if not errors:
        print("\nNo errors.")
        return
    print(f"\n--- Sample errors (showing {min(limit, len(errors))}/{len(errors)}) ---")
    for item in errors[:limit]:
        tags = ", ".join(item["tags"])
        print(f"\n[id={item['id']}] tags={tags}")
        print(f"Q: {item['question'][:120]}")
        print(f"gold={item['gold']}  pred={item['pred']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CoT on train dev split")
    parser.add_argument("--train", type=Path, default=Path("train.json"))
    parser.add_argument("--dev", type=Path, default=Path("dev.json"))
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictions", type=Path, default=Path("eval_preds.csv"))
    parser.add_argument("--report", type=Path, default=Path("eval_report.json"))
    parser.add_argument("--errors", type=Path, default=Path("eval_errors.jsonl"))
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="compact",
        choices=["direct", "compact", "full"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--rebuild-dev", action="store_true")
    parser.add_argument("--download-train", action="store_true")
    parser.add_argument("--show-errors", type=int, default=10)
    args = parser.parse_args()

    if args.download_train or not args.train.exists():
        if not args.train.exists():
            download_train(args.train)
        elif args.download_train:
            download_train(args.train)

    if args.score_only:
        if not args.dev.exists():
            raise FileNotFoundError(f"Missing dev split: {args.dev}")
        dev_data = load_dev(args.dev)
    else:
        train_data = load_train(args.train)
        if args.rebuild_dev or not args.dev.exists():
            build_dev_split(
                train_data,
                size=args.dev_size,
                seed=args.seed,
                dev_path=args.dev,
            )
        dev_data = load_dev(args.dev)

    if args.limit > 0:
        dev_data = dev_data[: args.limit]

    if args.score_only:
        if not args.predictions.exists():
            raise FileNotFoundError(f"Missing predictions: {args.predictions}")
        raw = load_completed_ids(args.predictions)
        predictions = {str(k): v for k, v in raw.items()}
    else:
        cfg_kwargs: dict = {
            "base_url": args.ollama_url,
            "model": args.model,
            "prompt_mode": args.prompt_mode,
        }
        if args.temperature is not None:
            cfg_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            cfg_kwargs["top_p"] = args.top_p
        if args.num_ctx is not None:
            cfg_kwargs["num_ctx"] = args.num_ctx
        if args.repeat_penalty is not None:
            cfg_kwargs["repeat_penalty"] = args.repeat_penalty
        if args.num_predict is not None:
            cfg_kwargs["num_predict"] = args.num_predict
        config = OllamaConfig(**cfg_kwargs)  # type: ignore[arg-type]
        print(
            f"Ollama: {config.base_url}, model: {config.model}, mode: {config.prompt_mode}, "
            f"T={config.temperature}, top_p={config.top_p}, ctx={config.num_ctx}, "
            f"repeat={config.repeat_penalty}, predict={config.num_predict}"
        )
        predictions = run_inference(
            dev_data,
            config=config,
            predictions_path=args.predictions,
        )

    report, errors = score_predictions(dev_data, predictions)

    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.errors.open("w", encoding="utf-8") as f:
        for item in errors:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print_report(report)
    print_errors(errors, args.show_errors)
    print(f"\nReport -> {args.report}")
    print(f"Errors -> {args.errors} ({len(errors)} items)")
    print(f"Predictions -> {args.predictions}")


if __name__ == "__main__":
    main()
