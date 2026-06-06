"""GPU CoT inference via transformers (no Ollama required)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_core import DIRECT_INSTRUCTION, normalize_question
from ft_utils import (
    COT_DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_INSTRUCTION,
    DEFAULT_MODEL_ID,
    build_cot_training_prefix,
    build_training_prefix,
    format_cot_prediction,
    format_ft_prediction,
    resolve_model_dir,
)


def require_gpu() -> None:
    if not torch.cuda.is_available():
        print("ERROR: GPU required (--backend gpu). PyTorch sees no CUDA/ROCm device.", file=sys.stderr)
        sys.exit(1)


def load_cot_model(
    *,
    model_id: str,
    cache_dir: Path,
    checkpoint: Path | None = None,
):
    require_gpu()
    base_dir = resolve_model_dir(model_id, cache_dir)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if checkpoint is not None and (checkpoint / "adapter_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True, use_fast=False)
        base = AutoModelForCausalLM.from_pretrained(
            base_dir, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, str(checkpoint))
        model.eval()
        return model, tokenizer

    load_dir = checkpoint if checkpoint is not None and (checkpoint / "config.json").exists() else base_dir
    tokenizer = AutoTokenizer.from_pretrained(load_dir, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        load_dir, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


class CotGpuEngine:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Path = Path("models"),
        checkpoint: Path | None = None,
        max_new_tokens: int = COT_DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        name = torch.cuda.get_device_name(0)
        print(f"GPU backend: {name}")
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = load_cot_model(
            model_id=model_id, cache_dir=cache_dir, checkpoint=checkpoint
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def infer(self, question: str | list, *, prompt_mode: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
        q = normalize_question(question)
        if prompt_mode == "direct":
            prompt = build_training_prefix(self.tokenizer, instruction or DIRECT_INSTRUCTION, q)
            raw = self._generate(prompt)
            return format_ft_prediction(raw, q)
        compact = prompt_mode != "full"
        prompt = build_cot_training_prefix(self.tokenizer, q, compact=compact)
        raw = self._generate(prompt)
        return format_cot_prediction(raw, q)
