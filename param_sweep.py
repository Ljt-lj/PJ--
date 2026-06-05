"""Compare Ollama sampling params on a fixed dev subset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from eval import load_dev, score_predictions
from ollama_client import OllamaConfig, chat


@dataclass(frozen=True)
class ParamSet:
    name: str
    temperature: float
    top_p: float
    num_ctx: int = 4096
    repeat_penalty: float = 1.1
    num_predict: int = 256


PARAM_SETS = [
    ParamSet("baseline", 0.3, 0.9),
    ParamSet("low_temp", 0.1, 0.85),
    ParamSet("greedy", 0.0, 1.0),
    ParamSet("low_temp_rp12", 0.1, 0.9, repeat_penalty=1.2),
    ParamSet("mid_temp", 0.2, 0.9),
    ParamSet("long_out", 0.1, 0.9, num_predict=384),
]


def run_sweep(
    dev_path: Path = Path("dev.json"),
    limit: int = 50,
    prompt_mode: str = "compact",
) -> list[dict]:
    dev_data = load_dev(dev_path)[:limit]
    results: list[dict] = []

    for ps in PARAM_SETS:
        config = OllamaConfig(
            prompt_mode=prompt_mode,  # type: ignore[arg-type]
            temperature=ps.temperature,
            top_p=ps.top_p,
            num_ctx=ps.num_ctx,
            repeat_penalty=ps.repeat_penalty,
            num_predict=ps.num_predict,
        )
        preds: dict[str, str] = {}
        for i, row in enumerate(tqdm(dev_data, desc=ps.name)):
            sid = str(row.get("id", i))
            preds[sid] = chat(row["question"], config=config)

        report, _ = score_predictions(dev_data, preds)
        entry = {
            "name": ps.name,
            "params": {
                "temperature": ps.temperature,
                "top_p": ps.top_p,
                "num_ctx": ps.num_ctx,
                "repeat_penalty": ps.repeat_penalty,
                "num_predict": ps.num_predict,
            },
            "accuracy": report["accuracy"],
            "correct": report["correct"],
            "total": report["total"],
        }
        results.append(entry)
        print(f"{ps.name}: {report['accuracy']:.2%} ({report['correct']}/{report['total']})")

    return results


def main() -> None:
    results = run_sweep()
    out = Path("param_sweep_50.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    best = max(results, key=lambda x: x["accuracy"])
    print(f"Best: {best['name']} = {best['accuracy']:.2%}")


if __name__ == "__main__":
    main()
