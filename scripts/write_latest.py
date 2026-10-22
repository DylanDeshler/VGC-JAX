"""Write reports/LATEST.md (read at every session start, 11 §9) from a T2/T3 run's summaries."""
import json
import pathlib
import sys

run = sys.argv[1]
root = pathlib.Path(__file__).resolve().parent.parent
L = [f"# LATEST: {run}", ""]
L.append("| part | battles | CBR | TMR | MPR | DPR fails | errors | ext mismatches | top buckets |")
L.append("|---|---|---|---|---|---|---|---|---|")
for part in "abc":
    p = root / "reports" / "conformance" / f"{run}-{part}" / "summary.json"
    if not p.exists():
        L.append(f"| {part} | missing | | | | | | | |")
        continue
    s = json.loads(p.read_text())
    L.append(f"| {part} | {s['battles']} | {s['CBR']:.4%} | {s['TMR']:.4%} | {s['MPR']} | {s['draw_failures']} | "
             f"{s['error_rate']} | {s.get('ext_mismatch_battles')} | {json.dumps(s['buckets'])[:120]} |")
reval = root / "reports" / "conformance" / f"{run}-reval.json"
if reval.exists():
    L += ["", "## Don't-care re-validation", "```", reval.read_text()[:1500], "```"]
log = root / "reports" / f"{run}.log"
if log.exists() and "CALLSITE DRIFT" in log.read_text():
    L += ["", "**RNG call-site drift detected** (oracle check-callsites)."]
(root / "reports" / "LATEST.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
