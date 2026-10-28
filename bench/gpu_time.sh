#!/usr/bin/env bash
# Short GPU timing run (ADR-107): the training build (events off, micro_steps 2), battles/s only. No PS comparison.
# Batch scaling at micro_steps 2 with select access (the GPU default) and the compact state (ADR-107).
# One compile per line (~6 min each on an L40S).
#   git pull && bash bench/gpu_time.sh | tee gpu_time.txt
set -euo pipefail
uv sync --extra gpu
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python -c "import jax; print('devices:', jax.devices())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
git log --oneline -1
for cfg in "4096 select" "8192 select" "16384 select" "32768 select"; do
  set -- $cfg
  echo "== B=$1 micro_steps=2 access=$2 training build (events off)"
  VGCJAX_ACCESS=$2 uv run python bench/throughput.py --stage impl --batch "$1" --steps 32 --micro-steps 2 --no-events --reps 2
done
