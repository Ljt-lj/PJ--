#!/usr/bin/env python3
"""Diagnose GPU (NVIDIA CUDA or AMD ROCm) before fine-tuning."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"(failed: {exc})"


def detect_backend() -> str:
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocm-smi") or os.environ.get("ROCM_PATH"):
        return "rocm"
    return "unknown"


def main() -> None:
    backend = detect_backend()
    print("=== GPU environment check ===\n")
    print(f"Detected backend: {backend.upper() if backend != 'unknown' else 'UNKNOWN'}\n")

    if backend == "nvidia":
        print("nvidia-smi:\n", run(["nvidia-smi", "-L"]), sep="")
    elif backend == "rocm":
        print("rocm-smi:\n", run(["rocm-smi", "--showmeminfo"]), sep="")
        print(
            "\nNote: This is an **AMD GPU + ROCm** environment.\n"
            "  nvidia-smi will NOT work here — that tool is NVIDIA-only.\n"
            "  Use rocm-smi to inspect AMD GPUs."
        )
    else:
        print("nvidia-smi: NOT FOUND")
        print("rocm-smi:   NOT FOUND")
        print("  -> CPU-only instance, or GPU drivers not loaded.")

    print()
    try:
        import torch
    except ImportError:
        print("torch: NOT INSTALLED")
        if backend == "rocm":
            print("  On ModelScope AMD images, do NOT pip install cu121 torch.")
            print("  Use the preinstalled ROCm PyTorch from the system image.")
        else:
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    print(f"torch version:      {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"device count:       {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        if backend == "rocm":
            print("\nOK (ROCm) — PyTorch uses the 'cuda' API name but runs on AMD via HIP.")
        else:
            print("\nOK (CUDA) — you can run: python run_ft_pipeline.py --merge-lora")
        return

    print("\nGPU NOT usable by PyTorch.\n")

    if backend == "rocm":
        print("AMD/ROCm fixes:")
        print("1. Do NOT install NVIDIA CUDA wheels (cu121/cu118).")
        print("2. Remove wrong torch and use image default:")
        print("   pip uninstall -y torch torchvision torchaudio")
        print("   pip install torch  # or exit venv and use system PyTorch 2.9.1+rocm")
        print("3. Prefer: train outside .venv-ft, or recreate venv with:")
        print("   python -m venv .venv-ft --system-site-packages")
        print("4. Verify: python -c \"import torch; print(torch.cuda.is_available())\"")
    elif backend == "nvidia":
        print("NVIDIA fixes:")
        print("   pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("Switch to a GPU instance (NVIDIA CUDA or AMD ROCm).")

    sys.exit(1)


if __name__ == "__main__":
    main()
