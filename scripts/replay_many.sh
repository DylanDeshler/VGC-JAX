#!/usr/bin/env bash
# Replay several trace directories against the engine and print one summary line each.
#   W=4 bash scripts/replay_many.sh <dir> [<dir> ...]     (a bare name means /tmp/wx/<name>)
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
for d in "$@"; do
  case "$d" in /*) src=$d ;; *) src=/tmp/wx/$d ;; esac
  out=${src%/}_rep
  r=$(cd "$R" && uv run python -m harness.cli replay "$src" --workers "${W:-3}" --out "$out" 2>&1 |
      grep -E '"(CBR|TMR|events_mismatch_battles|battles)"' | tr -d '\n ')
  b=$(python3 -c "import json; d=json.load(open('$out/summary.json')); print(d.get('buckets'), d.get('events_buckets'))" 2>/dev/null)
  echo "$(basename "$src") $r $b"
done
