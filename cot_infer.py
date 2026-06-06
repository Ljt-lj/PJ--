"""
Chain-of-Thought (CoT) inference for elementary math word problems.

Backends:
  gpu    - PyTorch on cloud/local GPU (recommended on ModelScope DSW)
  ollama - local Ollama HTTP API
  auto   - gpu if CUDA available, else ollama
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
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
            if header and len(header) >= 2:
                done[int(header[0])] = header[1]
            for row in reader:
                if len(row) >= 2:
                    done[int(row[0])] = row[1]
    return done


def append_checkpoint(output_path: Path, sample_id: int, answer: str) -> None:
    cp = checkpoint_path(output_path)
    with cp.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": sample_id, "ret": answer}, ensure_ascii=False) + "\n")


def write_submission(output_path: Path, results: dict[int, str], *, with_header: bool = False) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        if with_header:
            f.write("id,ret\n")
        for sample_id in sorted(results):
            f.write(f"{sample_id},{results[sample_id]}\n")


def infer_one_ollama(row: dict, *, config: OllamaConfig) -> tuple[int, str]:
    question = normalize_question(row["question"])
    answer = chat(question, config=config)
    return row["id"], format_submit_answer(question, answer)


def resolve_backend(name: str) -> str:
    if name != "auto":
        return name
    return "gpu" if torch.cuda.is_available() else "ollama"


def main() -> None:
    parser = argparse.ArgumentParser(description="CoT inference for Qwen2.5-0.5B")
    parser.add_argument("--test", type=Path, default=Path("test.json"))
    parser.add_argument("--output", type=Path, default=Path("submit.csv"))
    parser.add_argument(
        "--backend",
        choices=["auto", "gpu", "ollama"],
        default="auto",
        help="auto=GPU if available else Ollama",
    )
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional LoRA/merged weights")
    parser.add_argument("--temperature", type=float, default=COT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=COT_TOP_P)
    parser.add_argument("--num-ctx", type=int, default=COT_NUM_CTX)
    parser.add_argument("--num-predict", type=int, default=COT_NUM_PREDICT)
    parser.add_argument("--max-new-tokens", type=int, default=128, help="GPU backend only")
    parser.add_argument("--repeat-penalty", type=float, default=COT_REPEAT_PENALTY)
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default=BEST_PROMPT_MODE,
        choices=["direct", "compact", "full"],
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--with-header", action="store_true")
    args = parser.parse_args()

    backend = resolve_backend(args.backend)

    if not args.test.exists():
        raise FileNotFoundError(f"Missing {args.test}")

    with args.test.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.start:
        test_data = test_data[args.start :]
    if args.limit > 0:
        test_data = test_data[: args.limit]

    results = load_completed_ids(args.output)
    pending = [row for row in test_data if row["id"] not in results]

    print(f"Backend: {backend}")
    print(f"prompt_mode: {args.prompt_mode}")
    print(f"Total: {len(test_data)}, done: {len(results)}, pending: {len(pending)}")

    if not pending:
        write_submission(args.output, results, with_header=args.with_header)
        print(f"Already complete: {len(results)} predictions in {args.output}")
        return

    lock = threading.Lock()
    completed_since_save = 0

    def save_if_needed(force: bool = False) -> None:
        nonlocal completed_since_save
        if force or completed_since_save >= args.save_every:
            with lock:
                write_submission(args.output, results, with_header=args.with_header)
            completed_since_save = 0

    def record(sample_id: int, answer: str) -> None:
        nonlocal completed_since_save
        with lock:
            results[sample_id] = answer
            append_checkpoint(args.output, sample_id, answer)
            completed_since_save += 1

    if backend == "gpu":
        from cot_gpu import CotGpuEngine

        engine = CotGpuEngine(
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            checkpoint=args.checkpoint,
            max_new_tokens=args.max_new_tokens,
        )

        def infer_one_gpu(row: dict) -> tuple[int, str]:
            instruction = str(row.get("instruction") or "")
            ans = engine.infer(row["question"], prompt_mode=args.prompt_mode, instruction=instruction)
            return row["id"], ans

        for row in tqdm(pending, desc="CoT GPU inference"):
            sample_id, answer = infer_one_gpu(row)
            record(sample_id, answer)
            save_if_needed()
    else:
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
        print(f"Ollama @ {config.base_url}, model={config.model}")
        print(
            f"Params: temperature={config.temperature}, top_p={config.top_p}, "
            f"num_predict={config.num_predict}"
        )

        if args.workers <= 1:
            for row in tqdm(pending, desc="CoT Ollama inference"):
                sample_id, answer = infer_one_ollama(row, config=config)
                record(sample_id, answer)
                save_if_needed()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(infer_one_ollama, row, config=config): row for row in pending}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="CoT Ollama inference"):
                    sample_id, answer = fut.result()
                    record(sample_id, answer)
                    save_if_needed()

    save_if_needed(force=True)
    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
