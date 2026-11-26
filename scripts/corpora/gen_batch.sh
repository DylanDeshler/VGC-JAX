#!/usr/bin/env bash
# Generate the batch verification corpora (one per spec line) into traces/wx/b/, deterministic seeds 9101...
#   bash scripts/corpora/gen_batch.sh            then   W=8 bash scripts/replay_many.sh $PWD/traces/wx/b/*/
set -uo pipefail
cd "$(dirname "$0")/../.."
SPECS=${SPECS:-scripts/corpora/batch_specs.txt}; OUT=${OUT:-traces/wx/b}; N=${N:-150}; SEED0=${SEED0:-9100}
mkdir -p "$OUT"
i=0
while read -r spec; do
  i=$((i+1)); name=$(echo "${spec#stagereq:impl:}" | tr ':+' '__')
  pol=(); [[ $spec == *ite* || $spec == *mimikyu* || $spec == *kangaskhan* ]] && pol=(--policy random-weighted)
  [[ -d $OUT/$name ]] && continue
  node oracle/bin/oracle.js gen --n "$N" --seed $((SEED0+i)) --workers "${GEN_WORKERS:-8}" --stage impl --events \
    --out "$OUT/$name" --teams "$spec" "${pol[@]}" > "$OUT/$name.log" 2>&1 || echo "partial: $name (see log)"
done < "$SPECS"
echo GEN_DONE
