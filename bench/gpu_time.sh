#!/usr/bin/env bash
# Short GPU timing run (ADR-107): the training build (events off), battles/s only. No PS comparison.
# Sweeps micro_steps per call at B=8192 (2, 3, 4) and checks B=16384 at 4. One compile per line (~10 min each on an
# L40S at commit e9951ed; less since).
#   git pull && bash bench/gpu_time.sh | tee gpu_time.txt
set -euo pipefail
uv sync --extra gpu
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python -c "import jax; print('devices:', jax.devices())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
git log --oneline -1
for cfg in "8192 2" "8192 3" "8192 4" "16384 4"; do
  set -- $cfg
  echo "== B=$1 micro_steps=$2 training build (events off)"
  uv run python bench/throughput.py --stage impl --batch "$1" --steps 32 --micro-steps "$2" --no-events --reps 2
done
