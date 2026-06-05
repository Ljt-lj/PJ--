#!/usr/bin/env python3
"""Diagnose GPU / PyTorch CUDA availability before fine-tuning."""

from __future__ import annotations

import shutil
import subprocess
import sys


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"(failed: {exc})"


def main() -> None:
    print("=== GPU environment check ===\n")

    if shutil.which("nvidia-smi"):
        print("nvidia-smi:\n", run(["nvidia-smi", "-L"]), sep="")
    else:
        print("nvidia-smi: NOT FOUND")
        print("  -> Machine likely has no GPU driver, or you picked a CPU-only instance.")
        print("  -> In PAI-DSW / AutoDL / ModelScope: switch to a GPU spec (T4/A10/3090 etc.).")

    print()
    try:
        import torch
    except ImportError:
        print("torch: NOT INSTALLED")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    print(f"torch version:     {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda available:    {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print("\nOK — you can run: python run_ft_pipeline.py --merge-lora")
        return

    print("\nCUDA NOT usable by PyTorch. Common fixes:\n")
    if torch.version.cuda is None:
        print("1. You installed CPU-only PyTorch. Reinstall GPU build, e.g.:")
        print("   pip uninstall -y torch torchvision torchaudio")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("   # if driver is CUDA 11.8, use cu118 instead of cu121")
    else:
        print("1. PyTorch has CUDA build but runtime failed — check driver vs torch CUDA version.")
        print("2. Reinstall matching wheel (cu118 / cu121 / cu124).")

    print("3. Confirm nvidia-smi works in the same shell/venv.")
    print("4. On 魔搭 DSW: create/switch to **GPU** 实例，镜像选 PyTorch + CUDA。")
    sys.exit(1)


if __name__ == "__main__":
    main()
