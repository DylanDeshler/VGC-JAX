#!/bin/sh
# jax-metal (Apple GPU) attempt: separate venv pinned to jax 0.4.34 + jax-metal 0.1.1
cd "$(dirname "$0")"
export ENABLE_PJRT_COMPATIBILITY=1
PY=.venv-metal/bin/python
echo "== b1 1A switch on METAL (N=8,32,128) =="
$PY b1_dispatch.py --part A --ns 8,32,128 --B 65536 2>&1 | grep -E "^1A|Error|error" | head -20
echo "== b5 rollout on METAL B=8192 T=32 =="
$PY b5_rollout.py --B 8192 --T 32 --reps 2 2>&1 | grep -E "RESULT|Error|error" | head -5
