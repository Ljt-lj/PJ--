#!/usr/bin/env bash
# Cloud GPU one-shot environment setup (Ubuntu 20.04/22.04 + CUDA)
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv-ft}"

echo "==> Creating venv: $VENV_DIR"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel

echo "==> Installing PyTorch (CUDA 12.1 wheel; change if your image uses CUDA 11.8)"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing project fine-tune deps"
pip install -r requirements-ft.txt

echo "==> GPU check"
python check_gpu.py || exit 1

echo
echo "Ready. Run:"
echo "  source $VENV_DIR/bin/activate"
echo "  python run_ft_pipeline.py --merge-lora"
