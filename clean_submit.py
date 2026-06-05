#!/usr/bin/env python3
"""Clean polluted submit.csv (e.g. '37Human: ...') without re-running inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cot_core import normalize_question
from ft_utils import format_ft_prediction


def load_questions(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {str(r["id"]): normalize_question(r["question"]) for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip Human:/Assistant garbage from submit.csv")
    parser.add_argument("--input", type=Path, default=Path("submit.csv"))
    parser.add_argument("--output", type=Path, default=None, help="Default: overwrite --input")
    parser.add_argument("--test", type=Path, default=Path("test.json"), help="For percent formatting")
    args = parser.parse_args()

    out_path = args.output or args.input
    questions = load_questions(args.test)

    rows: list[tuple[str, str]] = []
    dirty = 0
    with args.input.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or "," not in line:
                continue
            if i == 0 and line.lower().startswith("id,"):
                continue
            sid, ret = line.split(",", 1)
            sid, ret = sid.strip(), ret.strip()
            q = questions.get(sid, "")
            cleaned = format_ft_prediction(ret, q)
            if cleaned != ret:
                dirty += 1
            rows.append((sid, cleaned))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        for sid, ret in rows:
            f.write(f"{sid},{ret}\n")

    print(f"Rows: {len(rows)}, cleaned: {dirty}, output -> {out_path}")


if __name__ == "__main__":
    main()
