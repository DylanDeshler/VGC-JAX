"""Progress of a running (or stopped) replay from its incremental results: battles done, failures, failing lines.
    uv run python scripts/corpora/partial_status.py traces/regress_merged_rep [--fails 30]
"""
import argparse
import json
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--fails", type=int, default=30)
a = ap.parse_args()
res = [json.loads(l) for f in sorted((pathlib.Path(a.out) / "partial").glob("partial_w*.jsonl")) for l in open(f)
       if l.strip()]
bad = [r for r in res if not r.get("clean") or r.get("events")]
print(f"done {len(res)}  failing {len(bad)}  clean {100 * (len(res) - len(bad)) / max(len(res), 1):.2f}%")
for r in sorted(bad, key=lambda r: r["index"])[:a.fails]:
    print(r["index"], r.get("category"), r.get("k"), json.dumps(r.get("detail"))[:300], "|",
          json.dumps(r.get("events"))[:300])
