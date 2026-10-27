#!/usr/bin/env bash
# A/B CPU timing on a noisy host (ADR-107): baseline commit vs the working tree, alternating; battles per CPU-second
# (process CPU time, best of N repetitions), which other load on the host affects much less than wall time.
#   bash scripts/ab_time.sh <baseline-commit> [rounds]
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
BASE=${1:?baseline commit}; ROUNDS=${2:-2}
WT=/tmp/ab_base_$BASE
[ -d "$WT" ] || git -C "$R" worktree add -q "$WT" "$BASE"
ARGS="--stage core --batch 1024 --steps 16 --micro-steps 4 --no-events --reps 4"
for i in $(seq "$ROUNDS"); do
  b=$(cd "$WT" && PYTHONPATH="$WT" uv run --project "$R" python "$R/bench/throughput.py" $ARGS 2>&1 | grep "CPU-second" | awk "{print \$4}")
  n=$(cd "$R" && PYTHONPATH="$R" uv run python bench/throughput.py $ARGS 2>&1 | grep "CPU-second" | awk "{print \$4}")
  echo "round $i: base $BASE $b battles/CPU-s | tree $n battles/CPU-s"
done
