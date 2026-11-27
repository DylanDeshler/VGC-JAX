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
    ap.add_argument("--per-file", type=int, default=None, help="first N battles of each shard (a sample)")
    ap.add_argument("--report", default=None, help="summarize an existing replay output dir (no replay)")
    a = ap.parse_args()
    root = pathlib.Path(a.root).resolve()
    merged, out = root.parent / (root.name + "_merged"), root.parent / (root.name + "_merged_rep")
    if not a.report:                             # report mode must not touch a merged dir a live replay is reading
        shutil.rmtree(merged, ignore_errors=True)
        merged.mkdir()
    # result index -> corpus: harness.cli replay walks the merged dir's shards in sorted name order and numbers the
    # battles it keeps (the first --per-file of each shard) 0, 1, 2, ... (battle_ids repeat across same-seed corpora)
    entries = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.endswith("_rep")):
        if a.only and d.name not in a.only:
            continue
        for f in shard_files(d):
            if not a.report:
                (merged / f"{d.name}__{f.name}").symlink_to(f)
            n = sum(1 for _ in iter_battles(f))
            entries.append((f"{d.name}__{f.name}", d.name, min(n, a.per_file) if a.per_file else n))
    owner_by_index = []
    for _, corpus, n in sorted(entries):
        owner_by_index += [corpus] * n
    print(f"merged {len(owner_by_index)} battles from {len({c for _, c, _ in entries})} corpora", flush=True)
    if a.dry:
        return 0
    if a.report:
        out, rc = pathlib.Path(a.report), 0
    else:
        rc = subprocess.call([sys.executable, "-m", "harness.cli", "replay", str(merged), "--workers", str(a.workers),
                              "--out", str(out)] + (["--per-file", str(a.per_file)] if a.per_file else []))
    if (out / "results.jsonl").exists():
        res = [json.loads(l) for l in open(out / "results.jsonl")]
    else:                                      # a stopped run: the workers' incremental files
        res = [json.loads(l) for f in sorted((out / "partial").glob("partial_w*.jsonl")) for l in open(f) if l.strip()]
        print(f"(partial results: {len(res)} battles)")
    by = collections.defaultdict(list)
    for r in res:
        i = r["index"]
        by[owner_by_index[i] if i < len(owner_by_index) else "?"].append(r)
    n_all = len(res)
    n_ok = sum(bool(r.get("clean")) and not r.get("events") for r in res)
    print(f"\n== overall: {n_ok}/{n_all} battles fully clean ({100 * n_ok / max(n_all, 1):.2f}%)")
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
