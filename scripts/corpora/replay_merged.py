"""Replay many corpora as ONE replay run (each worker compiles the engine once, not once per corpus), then print a
per-corpus summary and the failing battles. A worker's compile peaks at ~44 GB (measured on r7i, 2026-11-25): pick
--workers so workers * 45 GB fits, and run under a hard memory cap (Linux: systemd-run --scope -p MemoryMax=...).

    uv run python scripts/corpora/replay_merged.py traces/wx/b --workers 4
"""
import argparse
import collections
import json
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from harness.corpus import iter_battles, shard_files  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory whose subdirectories are corpora")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="corpus names to include (default: all)")
    ap.add_argument("--dry", action="store_true", help="merge only; no engine")
    a = ap.parse_args()
    root = pathlib.Path(a.root).resolve()
    merged, out = root.parent / (root.name + "_merged"), root.parent / (root.name + "_merged_rep")
    shutil.rmtree(merged, ignore_errors=True)
    merged.mkdir()
    owner = {}                                   # battle_id -> corpus
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.endswith("_rep")):
        if a.only and d.name not in a.only:
            continue
        for f in shard_files(d):
            (merged / f"{d.name}__{f.name}").symlink_to(f)
            for b in iter_battles(f):
                owner[b[0]["battle_id"]] = d.name
    print(f"merged {len(owner)} battles from {len(set(owner.values()))} corpora", flush=True)
    if a.dry:
        return 0
    rc = subprocess.call([sys.executable, "-m", "harness.cli", "replay", str(merged), "--workers", str(a.workers),
                          "--out", str(out)])
    res = [json.loads(l) for l in open(out / "results.jsonl")]
    by = collections.defaultdict(list)
    for r in res:
        by[owner.get(r["battle_id"], "?")].append(r)
    print("\n== per corpus (battles, state-clean %, event-clean %)")
    for name in sorted(by):
        rs = by[name]
        sc = sum(bool(r.get("clean")) for r in rs)
        ec = sum(not r.get("events") for r in rs)
        flag = "" if sc == ec == len(rs) else "   <--"
        print(f"{name:45s} {len(rs):4d} {100 * sc / len(rs):6.1f} {100 * ec / len(rs):6.1f}{flag}")
    print("\n== failing battles (corpus index category k detail | events)")
    for name in sorted(by):
        for r in by[name]:
            if not r.get("clean") or r.get("events"):
                print(name, r["index"], r.get("category"), r.get("k"), json.dumps(r.get("detail"))[:400], "|",
                      json.dumps(r.get("events"))[:400])
    return rc


if __name__ == "__main__":
    sys.exit(main())
