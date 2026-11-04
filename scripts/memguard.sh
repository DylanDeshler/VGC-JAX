#!/usr/bin/env bash
# Run a command under a memory cap: sums the RSS of the whole process tree every second and kills it (SIGTERM, then
# SIGKILL) when the total exceeds MEM_CAP_GB (default 36, i.e. margin under a 40 GB budget). Prints the peak on exit.
#   MEM_CAP_GB=36 bash scripts/memguard.sh <command> [args ...]
set -uo pipefail
CAP_KB=$(( ${MEM_CAP_GB:-36} * 1024 * 1024 ))
"$@" &
ROOT=$!
peak=0
tree_rss() {      # total RSS (KB) of $1 and all its descendants
  ps -axo pid=,ppid=,rss= | awk -v root="$1" '
    { pp[$1] = $2; rss[$1] = $3 }
    END { for (p in pp) { q = p; while (q != "" && q != 0 && q != root) q = pp[q]; if (q == root) s += rss[p] } print s + 0 }'
}
while kill -0 "$ROOT" 2>/dev/null; do
  cur=$(tree_rss "$ROOT")
  (( cur > peak )) && peak=$cur
  if (( cur > CAP_KB )); then
    echo "memguard: ${cur} KB > cap ${CAP_KB} KB; killing" >&2
    pkill -TERM -P "$ROOT" 2>/dev/null; kill -TERM "$ROOT" 2>/dev/null; sleep 3
    for p in $(ps -axo pid=,ppid= | awk -v r="$ROOT" '$2==r{print $1}'); do pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; done
    kill -KILL "$ROOT" 2>/dev/null
    break
  fi
  sleep 1
done
wait "$ROOT" 2>/dev/null; rc=$?
echo "memguard: peak $(( peak / 1024 )) MB (cap $(( CAP_KB / 1024 )) MB), exit $rc" >&2
exit $rc
