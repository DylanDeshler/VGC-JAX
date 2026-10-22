#!/usr/bin/env bash
# Local CI tiers T2 (nightly) / T3 (weekly) (07 §12.2, 02 §6). Writes reports/LATEST.md and
# reports/catalogue_status.json. Usage: scripts/nightly.sh t2|t3
set -uo pipefail
cd "$(dirname "$0")/.."
tier=${1:-t2}
date_tag=$(date +%Y-%m-%d)
run="${tier}-${date_tag}"
seed="${run}-$(git rev-parse --short HEAD)"
stage=${STAGE:-core}
case "$tier" in
  t2) n=${N:-20000}; legal=0.1 ;;
  t3) n=${N:-100000}; legal=0.1 ;;
  *) echo "unknown tier $tier"; exit 2 ;;
esac
log="reports/${run}.log"
mkdir -p reports traces
{
  echo "# $run  (seed $seed, stage $stage, n $n)"
  node oracle/bin/oracle.js check-callsites || echo "CALLSITE DRIFT"
  rm -rf "traces/$run"
  # composition (07 §12.2): 50% random, 30% random-weighted, 20% heuristic
  node oracle/bin/oracle.js gen --n $((n / 2)) --seed "$seed-a" --workers 12 --teams "stage:$stage" --policy random --legal $legal --out "traces/$run/a"
  node oracle/bin/oracle.js gen --n $((n * 3 / 10)) --seed "$seed-b" --workers 12 --teams "stage:$stage" --policy random-weighted --legal $legal --out "traces/$run/b"
  node oracle/bin/oracle.js gen --n $((n / 5)) --seed "$seed-c" --workers 12 --teams "stage:$stage" --policy heuristic --legal $legal --out "traces/$run/c"
  for part in a b c; do
    uv run python -m harness.cli replay "traces/$run/$part" --stage "$stage" --workers 12 --out "reports/conformance/$run-$part"
  done
  if [ "$tier" = t3 ]; then
    node oracle/tools/revalidate_dontcare.js --n 5000 --teams chaos --seed "$seed-reval" > "reports/conformance/$run-reval.json"
    PYTHONPATH=. uv run python bench/throughput.py --stage "$stage" > "reports/conformance/$run-throughput.txt" 2>&1
    uv run pytest -m slow -q > "reports/conformance/$run-slow-tests.txt" 2>&1
  fi
} > "$log" 2>&1
PYTHONPATH=. uv run python scripts/write_latest.py "$run" >> "$log" 2>&1
# passing shards are not kept (02 §6 storage)
rm -rf "traces/$run"
