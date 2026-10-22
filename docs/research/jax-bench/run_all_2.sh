#!/bin/sh
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY b4_rng.py > results/b4.log 2>&1
$PY b3_state.py > results/b3.log 2>&1
$PY b5_driver.py > results/b5.log 2>&1
$PY b6_ergonomics.py > results/b6.log 2>&1
echo DONE > results/run_all_2.done
