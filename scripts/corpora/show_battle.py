"""Show PS's side of battle #INDEX of a replay_merged.py run (global index over the merged corpora; no engine):
teams, then each decision k's choices and PS events up to --k.
    uv run python scripts/corpora/show_battle.py traces/wx/b 5712 --k 3
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from harness.corpus import iter_battles, shard_files  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("index", type=int)
ap.add_argument("--k", type=int, default=None); ap.add_argument("--from-k", type=int, default=0)
a = ap.parse_args()
files = []
for d in sorted(p for p in pathlib.Path(a.root).iterdir() if p.is_dir() and not p.name.endswith("_rep")):
    files += [(d.name, f) for f in shard_files(d)]
files.sort(key=lambda x: f"{x[0]}__{x[1].name}")
i = 0
for name, f in files:
    for b in iter_battles(f):
        if i == a.index:
            h = b[0]
            print(name, h["battle_id"])
            for t in h["teams"]:
                print("  ", t.replace("]", "\n   "))
            for r in b[1:]:
                k = int(r.get("k", -1))
                if k < a.from_k or (a.k is not None and k > a.k):
                    continue
                print(f"-- k={k} turn={r.get('turn')} choices={r.get('choices')}")
                for e in (r.get("events") or []):
                    print("     ", e["kind"], e["actor"], e["target"], e["move"], e["v1"], e["v2"], e["flags"])
            sys.exit(0)
        i += 1
