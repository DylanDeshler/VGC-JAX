"""Harness CLI.

    uv run python -m harness.cli replay <trace-dir> [--stage core] [--workers 8] [--limit N] [--out reports/conformance/<run>]

`replay` runs the single-battle forced replay (harness/replay_one.py) over all shards with a process pool and writes
results.jsonl + summary.json (CBR, TMR, MPR, DPR, error rate per 07 §1, and failure buckets).
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import pathlib
import subprocess
import time

import orjson


def _load(path):
    raw = subprocess.run(["zstd", "-dc", str(path)], capture_output=True, check=True).stdout
    return [orjson.loads(l) for l in raw.splitlines() if l]


def _worker(args):
    files, stage, w, nw, limit, check_masks = args
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from harness.replay_one import replay_battle, split_battles
    from vgcjax.config import EngineConfig
    cfg = EngineConfig(stage=stage, forced=True, draw_log=True)
    out = []
    gi = 0
    for f in files:
        for b in split_battles(_load(f)):
            if limit is not None and gi >= limit:
                return out
            if gi % nw == w:
                h = b[0]
                n_dec = sum(1 for r in b if r["type"] == "decision")
                n_legal = sum(1 for r in b if r["type"] == "decision" and r.get("legal"))
                try:
                    r = replay_battle(b, cfg, check_masks=check_masks)
                except Exception as e:  # harness/engine crash is a HARNESS bucket
                    r = dict(clean=False, k=0, category="HARNESS", detail=repr(e)[:500])
                out.append(dict(index=gi, battle_id=h["battle_id"], decisions=n_dec, legal=n_legal, **{
                    k: (v if k != "detail" else json.loads(json.dumps(v, default=str))) for k, v in r.items()}))
            gi += 1
    return out


def summarize(results):
    n = len(results)
    clean = sum(r["clean"] for r in results)
    # TMR: decisions before first failure / decisions up to and including it
    num = den = 0
    mask_ok = mask_n = 0
    for r in results:
        d = r["decisions"]
        k = r["k"] if not r["clean"] else d
        num += min(k, d)
        den += min(k + (0 if r["clean"] else 1), d + (0 if r["clean"] else 1))
        if r["legal"]:
            seen = r["legal"] if r["clean"] else min(r["legal"], k + 1)
            mask_n += seen
            mask_ok += seen - (1 if r["category"] == "MASK" else 0)
    buckets = collections.Counter()
    for r in results:
        if not r["clean"]:
            det = r["detail"]
            sig = r["category"]
            if r["category"] == "STATE" and isinstance(det, list) and det:
                sig += ":" + str(det[0][0]).split("[")[0].split(".")[-1]
            buckets[sig] += 1
    return dict(battles=n, clean=clean, CBR=clean / max(n, 1), TMR=num / max(den, 1),
                MPR=mask_ok / max(mask_n, 1) if mask_n else None,
                error_rate=sum(1 for r in results if r["category"] == "ERROR") / max(n, 1),
                draw_failures=sum(1 for r in results if r["category"] in ("DRAW_ARGS", "DRAW_MISMATCH")),
                buckets=dict(buckets.most_common(20)))


def cmd_replay(a):
    files = sorted(pathlib.Path(a.dir).glob("*.ndjson.zst"))
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        parts = pool.map(_worker, [(files, a.stage, w, a.workers, a.limit, not a.no_masks) for w in range(a.workers)])
    results = sorted((r for p in parts for r in p), key=lambda r: r["index"])
    summ = summarize(results)
    summ["seconds"] = round(time.time() - t0, 1)
    summ["trace_dir"] = str(a.dir)
    out = pathlib.Path(a.out or f"reports/conformance/{pathlib.Path(a.dir).name}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("dir")
    r.add_argument("--stage", default="core")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--no-masks", action="store_true")
    a = ap.parse_args()
    if a.cmd == "replay":
        cmd_replay(a)


if __name__ == "__main__":
    main()
