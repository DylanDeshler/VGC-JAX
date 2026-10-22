"""Harness CLI.

    uv run python -m harness.cli replay <trace-dir> [--stage core] [--workers 8] [--limit N] [--out reports/conformance/<run>]
                                    [--fixtures N [--fixture-dir DIR]] [--canary]

`replay` runs the single-battle forced replay (harness/replay_one.py) over all shards with a process pool and writes
results.jsonl + summary.json (CBR, TMR, MPR, DPR, error rate per 07 §1, and failure buckets).
`--fixtures N` turns the first failing battle of up to N distinct signatures into shrunk, turn-isolated fixtures
(harness/fixtures.py + harness/shrink.py, 07 §10.2) under `<out>/fixtures` (or `--fixture-dir`, e.g.
tests/fixtures/conformance to make them merge-blocking). `--canary` replays with the ADR-029 canary bug injected
(harness self-test only).
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


def _cfg(stage, canary=False):
    from vgcjax.config import EngineConfig, Flags
    return EngineConfig(stage=stage, forced=True, draw_log=True, flags=Flags(canary_crit_2048=canary))


def _battle_at(files, index):
    import zstandard
    from harness.corpus import split_frames
    dz = zstandard.ZstdDecompressor()
    gi = 0
    for f in files:
        for fr in split_frames(pathlib.Path(f).read_bytes()):
            if gi == index:
                return [orjson.loads(l) for l in dz.decompress(fr, max_output_size=1 << 30).splitlines() if l]
            gi += 1
    raise IndexError(index)


FIXTURE_CATEGORIES = ("STATE", "DRAW_ARGS", "DRAW_MISMATCH", "ERROR")


def emit_fixtures(files, results, n, out_dir, stage, canary=False):
    """First failing battle per signature -> shrunk fixture (07 §10.2). Returns [(signature, path, entities)]."""
    from harness import fixtures as FX
    from harness.replay_one import replay_battle
    FX.FIXTURE_DIR = pathlib.Path(out_dir)
    cfg = _cfg(stage, canary)
    seen, made = set(), []
    for r in results:
        if len(made) >= n:
            break
        if r["clean"] or r["category"] not in FIXTURE_CATEGORIES:
            continue
        b = _battle_at(files, r["index"])
        rr = replay_battle(b, cfg)          # re-run in-process: detail keeps its structured form
        sig = FX.signature(rr)
        if sig in seen:
            continue
        seen.add(sig)
        path = FX.emit(b, rr, shrink=True, cfg=cfg)
        if path is not None:
            fx = json.loads(path.read_text())
            made.append(dict(signature=sig, path=str(path), entities=fx["entities"], kept_edits=len(fx["shrink_log"])))
    return made


def _worker(args):
    files, stage, w, nw, limit, check_masks, canary = args
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from harness.replay_one import replay_battle, split_battles
    cfg = _cfg(stage, canary)
    out = []
    gi = 0
    import pathlib as _pl
    import zstandard
    from harness.corpus import split_frames
    dz = zstandard.ZstdDecompressor()
    for f in files:
        for fr in split_frames(_pl.Path(f).read_bytes()):
            if limit is not None and gi >= limit:
                return out
            if gi % nw == w:
                b = [orjson.loads(l) for l in dz.decompress(fr, max_output_size=1 << 30).splitlines() if l]
                h = b[0]
                n_dec = sum(1 for r in b if r["type"] == "decision")
                n_legal = sum(1 for r in b if r["type"] == "decision" and r.get("legal"))
                from harness import forced as FO
                kinds = collections.Counter()
                for r_ in b:
                    if r_["type"] == "decision":
                        try:
                            kinds.update(x[2] for x in FO.compared_draws(r_))
                        except FO.HarnessError:
                            pass
                try:
                    r = replay_battle(b, cfg, check_masks=check_masks)
                except Exception as e:  # harness/engine crash is a HARNESS bucket
                    r = dict(clean=False, k=0, category="HARNESS", detail=repr(e)[:500])
                out.append(dict(index=gi, battle_id=h["battle_id"], decisions=n_dec, legal=n_legal, kinds=dict(kinds), **{
                    k: (v if k != "detail" else json.loads(json.dumps(v, default=str))) for k, v in r.items()}))
            gi += 1
    return out


def summarize(results):
    from harness import fixtures as FX
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
            buckets[FX.signature(r)] += 1
    cov = collections.Counter()
    for r in results:
        cov.update(r.get("kinds", {}))
    ext = collections.Counter()
    for r in results:
        e = r.get("ext")
        if e:
            ext[str(e["detail"][0][0]).split("[")[-1]] += 1
    return dict(battles=n, clean=clean, coverage=dict(sorted(cov.items())),
                ext_mismatch_battles=sum(1 for r in results if r.get("ext")), ext_buckets=dict(ext.most_common(15)), CBR=clean / max(n, 1), TMR=num / max(den, 1),
                MPR=mask_ok / max(mask_n, 1) if mask_n else None,
                error_rate=sum(1 for r in results if r["category"] == "ERROR") / max(n, 1),
                draw_failures=sum(1 for r in results if r["category"] in ("DRAW_ARGS", "DRAW_MISMATCH")),
                buckets=dict(buckets.most_common(20)))


def cmd_replay(a):
    files = sorted(pathlib.Path(a.dir).glob("*.ndjson.zst"))
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        parts = pool.map(_worker, [(files, a.stage, w, a.workers, a.limit, not a.no_masks, a.canary)
                                    for w in range(a.workers)])
    results = sorted((r for p in parts for r in p), key=lambda r: r["index"])
    summ = summarize(results)
    summ["seconds"] = round(time.time() - t0, 1)
    summ["trace_dir"] = str(a.dir)
    out = pathlib.Path(a.out or f"reports/conformance/{pathlib.Path(a.dir).name}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    if a.fixtures:
        summ["fixtures"] = emit_fixtures(files, results, a.fixtures, a.fixture_dir or out / "fixtures", a.stage, a.canary)
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
    r.add_argument("--fixtures", type=int, default=0)
    r.add_argument("--fixture-dir", default=None)
    r.add_argument("--canary", action="store_true")
    a = ap.parse_args()
    if a.cmd == "replay":
        cmd_replay(a)


if __name__ == "__main__":
    main()
