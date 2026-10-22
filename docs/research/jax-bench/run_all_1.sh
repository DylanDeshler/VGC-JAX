#!/bin/sh
# sequential benchmark runs (never run two benchmarks concurrently: they share the 14 cores)
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY b1_dispatch.py --part C --timeout 90 > results/b1_C.log 2>&1
$PY b2_control.py --part AB > results/b2.log 2>&1
$PY b3_state.py > results/b3.log 2>&1
$PY b4_rng.py > results/b4.log 2>&1
echo DONE > results/run_all_1.done
