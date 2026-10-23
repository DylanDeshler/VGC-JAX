#!/usr/bin/env bash
# One-shot GPU measurement for ADR-104 (09 §1). ~15-20 min on one GPU, mostly compilation.
#   git clone <repo> vgc-jax && cd vgc-jax && bash bench/gpu_check.sh | tee gpu_check.txt
set -euo pipefail
uv sync --extra gpu
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python -c "import jax; print('devices:', jax.devices())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
for B in 4096 16384 65536; do
  for EV in "" "--no-events"; do
    echo "== B=$B $EV"
    uv run python bench/throughput.py --batch "$B" --steps 32 $EV
  done
done
echo "== HLO kernel count on GPU (B=16384)"
uv run python bench/hlo_kernels.py --batch 16384 --no-events
