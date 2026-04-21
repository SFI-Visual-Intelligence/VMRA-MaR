#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHONPATH="${ROOT_DIR}/src" "${ROOT_DIR}/.venv/bin/torchrun" --nproc_per_node=8 -m vmra_mar.train \
  --output-dir "${ROOT_DIR}/artifacts/train_8xv100" \
  --epochs 20 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --num-workers 8 \
  --persistent-workers \
  --precision amp_fp16 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-epochs 1 \
  --clip-grad-norm 1.0
