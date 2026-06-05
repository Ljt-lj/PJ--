#!/usr/bin/env python3
"""Score submit.csv against labeled JSON (e.g. train.json) for local accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cot_core import answers_equal, normalize_question


def load_preds(path: Path, limit: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((str(row["id"]), str(row["ret"]).strip()))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_gold(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(r["id"]): r for r in data}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score submit.csv vs labeled JSON")
    parser.add_argument("--submit", type=Path, default=Path("submit.csv"))
    parser.add_argument("--gold", type=Path, default=Path("train.json"))
    parser.add_argument("--limit", type=int, default=4000, help="First N rows in submit.csv (0=all)")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    preds = load_preds(args.submit, args.limit)
    gold_map = load_gold(args.gold)

    correct = 0
    scored = 0
    missing = 0
    errors: list[dict] = []

    for sid, pred in preds:
        row = gold_map.get(sid)
        if row is None:
            missing += 1
            continue
        scored += 1
        gold = str(row["answer"]).strip()
        ok = answers_equal(pred, gold)
        if ok:
            correct += 1
        elif len(errors) < 20:
            errors.append(
                {
                    "id": sid,
                    "gold": gold,
                    "pred": pred,
                    "question": normalize_question(row["question"])[:120],
                }
            )

    acc = correct / scored if scored else 0.0
    report = {
        "submit_rows": len(preds),
        "scored": scored,
        "missing_gold": missing,
        "correct": correct,
        "accuracy": round(acc, 4),
    }

    print(f"Submit rows:  {len(preds)}")
    print(f"Scored:       {scored} (ids found in {args.gold.name})")
    print(f"Missing gold: {missing}")
    print(f"Correct:      {correct}")
    print(f"Accuracy:     {acc:.2%}")

    if errors:
        print(f"\nSample errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  id={e['id']} gold={e['gold']} pred={e['pred']}")

    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport -> {args.report}")


if __name__ == "__main__":
    main()
