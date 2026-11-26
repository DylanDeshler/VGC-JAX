#!/usr/bin/env bash
# Generate the batch verification corpora (one per spec line) into traces/wx/b/, deterministic seeds 9101...
#   bash scripts/corpora/gen_batch.sh            then   W=8 bash scripts/replay_many.sh $PWD/traces/wx/b/*/
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p traces/wx/b
i=0
while read -r spec; do
  i=$((i+1)); name=$(echo "${spec#stagereq:impl:}" | tr ':+' '__')
  pol=(); [[ $spec == *ite* || $spec == *mimikyu* || $spec == *kangaskhan* ]] && pol=(--policy random-weighted)
  [[ -d traces/wx/b/$name ]] && continue
  node oracle/bin/oracle.js gen --n 150 --seed $((9100+i)) --workers "${GEN_WORKERS:-8}" --stage impl --events \
    --out "traces/wx/b/$name" --teams "$spec" "${pol[@]}" > "traces/wx/b/$name.log" 2>&1 || echo "partial: $name (see log)"
done < scripts/corpora/batch_specs.txt
echo GEN_DONE
