#!/bin/sh
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY b6_ergonomics.py > results/b6.log 2>&1
$PY b1_barrier_chain.py > results/b1_barrier_chain.log 2>&1
$PY b5_driver.py extras > results/b5_extras.log 2>&1
$PY hlo_kernels.py results/rollout_B8192.hlo.txt 128 > results/hlo_kernels.log 2>&1
echo DONE > results/run_all_3.done
