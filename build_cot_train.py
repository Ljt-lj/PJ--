#!/usr/bin/env python3
"""Build CoT training JSON from train.json (Ollama generation or template fallback)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from cot_core import answers_equal, extract_answer
from ft_utils import normalize_cot_response, normalize_sample, template_cot_response
from ollama_client import DEFAULT_OLLAMA_URL, OllamaConfig, chat_content


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


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_one(
    row: dict,
    *,
    source: str,
    config: OllamaConfig | None,
) -> tuple[dict, str]:
    """Return (output row, status: ollama_ok|ollama_fallback|template)."""
    base = normalize_sample(row)
    gold = base["answer"]
    question = base["question"]

    if source == "template":
        cot = template_cot_response(gold)
        base["cot_response"] = cot
        base["cot_source"] = "template"
        return base, "template"

    assert config is not None
    try:
        raw = chat_content(question, config=config)
        pred = extract_answer(raw) if raw else ""
        if answers_equal(pred, gold):
            cot = normalize_cot_response(raw, gold)
            base["cot_response"] = cot
            base["cot_source"] = "ollama_ok"
            return base, "ollama_ok"
    except Exception:
        pass

    cot = template_cot_response(gold)
    base["cot_response"] = cot
    base["cot_source"] = "ollama_fallback"
    return base, "ollama_fallback"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train_cot.json for CoT LoRA")
    parser.add_argument("--train", type=Path, default=Path("train.json"))
    parser.add_argument("--output", type=Path, default=Path("train_cot.json"))
    parser.add_argument(
        "--source",
        choices=("auto", "ollama", "template"),
        default="auto",
        help="auto=Ollama with template fallback; template=fast gold-only CoT",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process first N samples (0=all)")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", type=str, default="qwen2.5:0.5b-instruct")
    parser.add_argument("--checkpoint", type=Path, default=Path("train_cot.checkpoint.jsonl"))
    parser.add_argument("--rebuild", action="store_true", help="Ignore checkpoint and rebuild")
    args = parser.parse_args()

    if not args.train.exists():
        print(f"Missing {args.train}", file=sys.stderr)
        sys.exit(1)

    rows = load_rows(args.train)
    if args.limit > 0:
        rows = rows[: args.limit]

    done = {} if args.rebuild else load_done(args.checkpoint)
    if args.rebuild and args.checkpoint.exists():
        args.checkpoint.unlink()

    use_ollama = args.source in ("auto", "ollama")
    config: OllamaConfig | None = None
    if use_ollama:
        config = OllamaConfig(base_url=args.ollama_url, model=args.model, prompt_mode="compact")
        try:
            from ollama_client import list_models

            models = list_models(args.ollama_url)
            print(f"Ollama models: {models[:5]}...")
        except Exception as exc:
            if args.source == "ollama":
                print(f"Ollama unavailable: {exc}", file=sys.stderr)
                sys.exit(1)
            print(f"Ollama unavailable ({exc}), using template fallback.")
            use_ollama = False

    effective_source = "ollama" if use_ollama else "template"
    stats = {"ollama_ok": 0, "ollama_fallback": 0, "template": 0, "skipped": 0}
    out_rows: list[dict] = []

    for row in tqdm(rows, desc="Build CoT train"):
        sid = str(row.get("id", len(out_rows)))
        if sid in done:
            out_rows.append(done[sid])
            stats["skipped"] += 1
            continue

        built, status = build_one(
            row,
            source=effective_source,
            config=config if use_ollama else None,
        )
        stats[status] += 1
        append_row(args.checkpoint, built)
        out_rows.append(built)

    args.output.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(out_rows)} rows -> {args.output}")
    print(
        f"Stats: ollama_ok={stats['ollama_ok']}, "
        f"ollama_fallback={stats['ollama_fallback']}, "
        f"template={stats['template']}, skipped={stats['skipped']}"
    )


if __name__ == "__main__":
    main()
