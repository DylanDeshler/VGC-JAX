#!/usr/bin/env bash
# Run a command under a memory guard (macOS). Every 0.5 s it checks five independent triggers and, if ANY fires, kills
# the command's whole process group (SIGTERM, then SIGKILL 2 s later):
#   1. RSS of the command's process group + descendants  > MEM_CAP_GB        (default 22)
#   2. system available memory (kern.memorystatus_level) < MIN_AVAIL_PCT     (default 25 %)
#   3. swap used grew by more than                         SWAP_GROW_GB      (default 2) since start
#   4. compressed memory ("stored in compressor") grew by  COMP_GROW_GB      (default 4) since start
#   5. system-wide anonymous memory (resident anonymous + compressed + swap) grew by more than SYS_GROW_GB (default MEM_CAP_GB) since start:
#      counts every byte the job allocates however macOS stores it (other apps' growth counts too: conservative)
# Why the system-wide triggers: macOS RSS excludes compressed and swapped pages, so under memory pressure a process's
# RSS stays low while its real footprint grows without bound (this is how a 250 GB blow-up got past an RSS-only guard).
# Triggers 2-5 do not depend on attributing memory to processes at all.
#   MEM_CAP_GB=22 bash scripts/memguard.sh <command> [args ...]
set -uo pipefail
CAP_KB=$(( ${MEM_CAP_GB:-22} * 1024 * 1024 ))
MIN_AVAIL=${MIN_AVAIL_PCT:-25}
SWAP_GROW_MB=$(( ${SWAP_GROW_GB:-2} * 1024 ))
COMP_GROW_PAGES=$(( ${COMP_GROW_GB:-4} * 1024 * 1024 * 1024 / $(pagesize) ))

swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
comp_pages() { vm_stat | awk '/stored in compressor/ { gsub("\\.", "", $NF); print $NF }'; }
anon_mb() {   # resident anonymous + logical compressed pages, in MB, plus swap used
  local ps; ps=$(pagesize)
  echo $(( $(vm_stat | awk '/^Anonymous pages|stored in compressor/ { gsub("\\.", "", $NF); s += $NF } END { print s }') * ps >> 20 ))
}
avail_pct() { sysctl -n kern.memorystatus_level; }

SWAP0=$(swap_mb); COMP0=$(comp_pages); ANON0=$(( $(anon_mb) + SWAP0 )); CAP_MB=$(( ${SYS_GROW_GB:-${MEM_CAP_GB:-22}} * 1024 ))
AV0=$(avail_pct)
if (( AV0 < MIN_AVAIL )); then
  echo "memguard: system available memory already ${AV0}% < ${MIN_AVAIL}%; refusing to start" >&2; exit 99
fi

set -m                       # job control: the command gets its own process group (PGID = its PID)
"$@" &
ROOT=$!
set +m

members() {   # PIDs in the process group, plus any descendants (in case something called setpgid)
  ps -axo pid=,ppid=,pgid= | awk -v root="$ROOT" '
    { pp[$1] = $2; pg[$1] = $3 }
    END { for (p in pp) { q = p; while (q != "" && q != 0 && q != root) q = pp[q]; if (q == root || pg[p] == root) print p } }'
}
tree_rss() { local m; m=$(members); [[ -z $m ]] && { echo 0; return; }
  ps -o rss= -p $(echo $m | tr ' ' ',') 2>/dev/null | awk '{ s += $1 } END { print s + 0 }'; }

kill_all() {
  echo "memguard: KILL ($1)" >&2
  local m; m=$(members)
  kill -TERM -- -"$ROOT" $m 2>/dev/null; sleep 2
  m=$(members); kill -KILL -- -"$ROOT" $m 2>/dev/null
}

peak=0; reason=""
while kill -0 "$ROOT" 2>/dev/null; do
  cur=$(tree_rss); (( cur > peak )) && peak=$cur
  av=$(avail_pct); sw=$(swap_mb); cp=$(comp_pages); an=$(( $(anon_mb) + sw - ANON0 ))
  if   (( cur > CAP_KB ));                       then reason="tree rss ${cur} KB > ${CAP_KB} KB"
  elif (( an > CAP_MB ));                        then reason="system anonymous memory grew ${an} MB > ${CAP_MB} MB"
  elif (( av < MIN_AVAIL ));                     then reason="system available ${av}% < ${MIN_AVAIL}%"
  elif (( sw - SWAP0 > SWAP_GROW_MB ));          then reason="swap grew $(( sw - SWAP0 )) MB > ${SWAP_GROW_MB} MB"
  elif (( cp - COMP0 > COMP_GROW_PAGES ));       then reason="compressor grew $(( (cp - COMP0) * $(pagesize) >> 20 )) MB"
  fi
  if [[ -n $reason ]]; then kill_all "$reason"; break; fi
  sleep 0.5
done
wait "$ROOT" 2>/dev/null; rc=$?
[[ -n $reason ]] && rc=137
echo "memguard: peak rss $(( peak / 1024 )) MB (cap $(( CAP_KB / 1024 )) MB), exit $rc${reason:+ (killed: $reason)}" >&2
exit $rc
