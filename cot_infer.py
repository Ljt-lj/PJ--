"""
Chain-of-Thought (CoT) inference for elementary math word problems.

Uses local Ollama (Qwen2.5-0.5B-Instruct) with CoT-tuned parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from cot_core import BEST_PROMPT_MODE, format_submit_answer, normalize_question
from ollama_client import (
    COT_NUM_CTX,
    COT_NUM_PREDICT,
    COT_REPEAT_PENALTY,
    COT_TEMPERATURE,
    COT_TOP_P,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaConfig,
    chat,
)


def checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".checkpoint.jsonl")


def load_completed_ids(output_path: Path) -> dict[int, str]:
    done: dict[int, str] = {}
    cp = checkpoint_path(output_path)
    if cp.exists():
        with cp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done[int(row["id"])] = str(row["ret"])
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and header[0].lower() == "id":
            for row in reader:
                if len(row) >= 2:
                    done[int(row[0])] = row[1]
        else:
            if header:
                done[int(header[0])] = header[1]
            for row in reader:
                if len(row) >= 2:
                    done[int(row[0])] = row[1]
    return done


def append_checkpoint(output_path: Path, sample_id: int, answer: str) -> None:
    cp = checkpoint_path(output_path)
    with cp.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": sample_id, "ret": answer}, ensure_ascii=False) + "\n")


def write_submission(output_path: Path, results: dict[int, str]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "ret"])
        for sample_id in sorted(results):
            writer.writerow([sample_id, results[sample_id]])


def infer_one(row: dict, *, config: OllamaConfig) -> tuple[int, str]:
    question = normalize_question(row["question"])
    answer = chat(question, config=config)
    return row["id"], format_submit_answer(question, answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="CoT inference with local Ollama Qwen2.5-0.5B")
    parser.add_argument("--test", type=Path, default=Path("test.json"))
    parser.add_argument("--output", type=Path, default=Path("submit.csv"))
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=COT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=COT_TOP_P)
    parser.add_argument("--num-ctx", type=int, default=COT_NUM_CTX)
    parser.add_argument("--num-predict", type=int, default=COT_NUM_PREDICT)
    parser.add_argument("--repeat-penalty", type=float, default=COT_REPEAT_PENALTY)
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default=BEST_PROMPT_MODE,
        choices=["direct", "compact", "full"],
        help="direct=赛题格式直接输出; compact/full=CoT",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent HTTP requests; use 2-4 only if OLLAMA_NUM_PARALLEL>=workers",
    )
    parser.add_argument("--save-every", type=int, default=500, help="Rewrite submit.csv every N samples")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    config = OllamaConfig(
        base_url=args.ollama_url,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
        repeat_penalty=args.repeat_penalty,
        num_predict=args.num_predict,
        prompt_mode=args.prompt_mode,  # type: ignore[arg-type]
    )

    if not args.test.exists():
        raise FileNotFoundError(
            f"Missing {args.test}. Download from "
            "https://github.com/AI-FDU/Math_Solver/raw/main/test.json"
        )

    with args.test.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.start:
        test_data = test_data[args.start :]
    if args.limit > 0:
        test_data = test_data[: args.limit]

    results = load_completed_ids(args.output)
    pending = [row for row in test_data if row["id"] not in results]

    print(f"Backend: Ollama @ {config.base_url}")
    print(f"Model:   {config.model}, prompt_mode: {config.prompt_mode}")
    print(
        f"Params:  temperature={config.temperature}, top_p={config.top_p}, "
        f"num_ctx={config.num_ctx}, num_predict={config.num_predict}, "
        f"repeat_penalty={config.repeat_penalty}"
    )
    print(f"Workers: {args.workers}")
    print(f"Total: {len(test_data)}, done: {len(results)}, pending: {len(pending)}")

    if not pending:
        write_submission(args.output, results)
        print(f"Already complete: {len(results)} predictions in {args.output}")
        return

    lock = threading.Lock()
    completed_since_save = 0

    def save_if_needed(force: bool = False) -> None:
        nonlocal completed_since_save
        if force or completed_since_save >= args.save_every:
            with lock:
                write_submission(args.output, results)
            completed_since_save = 0

    def record(sample_id: int, answer: str) -> None:
        nonlocal completed_since_save
        with lock:
            results[sample_id] = answer
            append_checkpoint(args.output, sample_id, answer)
            completed_since_save += 1

    if args.workers <= 1:
        for row in tqdm(pending, desc="CoT inference"):
            sample_id, answer = infer_one(row, config=config)
            record(sample_id, answer)
            save_if_needed()
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(infer_one, row, config=config): row for row in pending
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="CoT inference"):
                sample_id, answer = fut.result()
                record(sample_id, answer)
                save_if_needed()

    save_if_needed(force=True)
    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
