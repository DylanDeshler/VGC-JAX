#!/usr/bin/env bash
# One-shot GPU measurement for ADR-104 (09 §1). ~25-35 min on one GPU, mostly compilation.
# Read decision-steps/s (step calls that end at a decision point); env-steps/s also counts CONT continuation calls.
#   git clone <repo> vgc-jax && cd vgc-jax && bash bench/gpu_check.sh | tee gpu_check.txt
set -euo pipefail
uv sync --extra gpu
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python -c "import jax; print('devices:', jax.devices())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
for B in 4096 16384 65536; do
  for P in 4 8; do
    for EV in "" "--no-events"; do
      echo "== B=$B micro_steps=$P $EV"
      uv run python bench/throughput.py --batch "$B" --steps 32 --micro-steps "$P" $EV
    done
  done
done
echo "== HLO kernel count on GPU (B=16384, micro_steps=8)"
uv run python bench/hlo_kernels.py --batch 16384 --no-events --micro-steps 8
