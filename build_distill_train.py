#!/usr/bin/env python3
"""
Generate distillation training data with DeepSeek teacher -> train_distill.json

Usage:
  export DEEPSEEK_API_KEY=sk-...   # or api_key.txt
  python build_distill_train.py --limit 100          # smoke test
  python build_distill_train.py                      # full ~12k (API cost + time)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from cot_core import answers_equal, extract_answer
from deepseek_client import DEFAULT_MODEL, chat_content, load_api_key
from ft_utils import normalize_cot_response, normalize_sample, template_cot_response


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[str(row["id"])] = row
    return done


class CheckpointWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def append(self, row: dict) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def distill_one(
    row: dict,
    *,
    model: str,
    api_key: str,
    on_wrong: str,
    only_correct: bool,
) -> tuple[dict | None, str]:
    """
    Returns (row or None if skipped, status).
    status: teacher_ok | teacher_fallback | skipped_wrong | skipped_only_correct
    """
    base = normalize_sample(row)
    gold = base["answer"]
    question = base["question"]

    try:
        raw = chat_content(question, api_key=api_key, model=model, compact=True)
        pred = extract_answer(raw) if raw else ""
        if answers_equal(pred, gold):
            base["cot_response"] = normalize_cot_response(raw, gold)
            base["cot_source"] = "deepseek_ok"
            base["teacher_model"] = model
            return base, "teacher_ok"
    except Exception as exc:
        base["distill_error"] = str(exc)[:200]

    if only_correct:
        return None, "skipped_only_correct"

    if on_wrong == "skip":
        return None, "skipped_wrong"

    base["cot_response"] = template_cot_response(gold)
    base["cot_source"] = "deepseek_fallback"
    base["teacher_model"] = model
    return base, "teacher_fallback"


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek -> Qwen distillation dataset")
    parser.add_argument("--train", type=Path, default=Path("train.json"))
    parser.add_argument("--output", type=Path, default=Path("train_distill.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("train_distill.checkpoint.jsonl"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api-key-file", type=Path, default=Path("api_key.txt"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, help="Parallel API requests")
    parser.add_argument(
        "--on-wrong",
        choices=("template", "skip"),
        default="template",
        help="When teacher answer != gold: template=use gold CoT; skip=drop sample",
    )
    parser.add_argument(
        "--only-correct",
        action="store_true",
        help="Keep only samples where DeepSeek answer matches gold (recommended)",
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if not args.train.exists():
        print(f"Missing {args.train}", file=sys.stderr)
        sys.exit(1)

    api_key = load_api_key(args.api_key_file)
    rows = load_rows(args.train)
    if args.limit > 0:
        rows = rows[: args.limit]

    done = {} if args.rebuild else load_done(args.checkpoint)
    if args.rebuild and args.checkpoint.exists():
        args.checkpoint.unlink()

    writer = CheckpointWriter(args.checkpoint)
    stats = {
        "teacher_ok": 0,
        "teacher_fallback": 0,
        "skipped_wrong": 0,
        "skipped_only_correct": 0,
        "resumed": 0,
    }
    out_rows: list[dict] = []

    pending: list[dict] = []
    for row in rows:
        sid = str(row.get("id", len(out_rows)))
        if sid in done:
            out_rows.append(done[sid])
            stats["resumed"] += 1
            continue
        pending.append(row)

    print(f"Teacher: DeepSeek {args.model}")
    print(f"Samples: {len(rows)}, pending: {len(pending)}, only_correct={args.only_correct}")

    def process(row: dict) -> tuple[dict | None, str]:
        return distill_one(
            row,
            model=args.model,
            api_key=api_key,
            on_wrong=args.on_wrong,
            only_correct=args.only_correct,
        )

    if args.workers <= 1:
        for row in tqdm(pending, desc="DeepSeek distill"):
            built, status = process(row)
            stats[status] += 1
            if built is not None:
                writer.append(built)
                out_rows.append(built)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, row): row for row in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek distill"):
                built, status = fut.result()
                stats[status] += 1
                if built is not None:
                    writer.append(built)
                    out_rows.append(built)

    # Preserve train.json order for kept samples
    final_map = {str(r["id"]): r for r in out_rows}
    ordered = [final_map[str(row["id"])] for row in rows if str(row.get("id")) in final_map]

    args.output.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(ordered)} rows -> {args.output}")
    print(
        f"Stats: teacher_ok={stats['teacher_ok']}, fallback={stats['teacher_fallback']}, "
        f"skipped_wrong={stats['skipped_wrong']}, skipped_only_correct={stats['skipped_only_correct']}, "
        f"resumed={stats['resumed']}"
    )
    if stats["teacher_ok"] == 0 and stats["teacher_fallback"] == 0 and stats["resumed"] == 0:
        print("WARNING: no new distillation rows produced.", file=sys.stderr)


if __name__ == "__main__":
    main()
