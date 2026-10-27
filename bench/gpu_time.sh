#!/usr/bin/env bash
# Short GPU timing run (ADR-107): the training build (events off) at the batch sizes that matter, battles/s only.
# No PS comparison. About 3 compiles (~10 min each on an L40S at commit e9951ed) plus a minute.
#   git pull && bash bench/gpu_time.sh | tee gpu_time.txt
set -euo pipefail
uv sync --extra gpu
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python -c "import jax; print('devices:', jax.devices())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
git log --oneline -1
for B in 8192 16384; do
  echo "== B=$B micro_steps=4 training build (events off)"
  uv run python bench/throughput.py --stage impl --batch "$B" --steps 32 --micro-steps 4 --no-events
done
echo "== GPU kernel count, training build (B=8192, micro_steps 4)"
uv run python bench/hlo_kernels.py --stage impl --batch 8192 --no-events --micro-steps 4
