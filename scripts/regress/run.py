"""Conformance regression suite: generate the corpora of scripts/regress/suite.txt with the oracle and replay them
against the engine, writing one compact report to paste back (SUMMARY.txt).

    uv run python scripts/regress/run.py                      # all corpora, workers = cpu count - 2
    uv run python scripts/regress/run.py --workers 24 --only 'a_*,impl_*'
    uv run python scripts/regress/run.py --skip-gen           # reuse traces/regress/* (e.g. copied from elsewhere)

Generation needs node, the built oracle (`make oracle`) and, for the `stagereq:` corpora, the Smogon chaos file at
data/raw/smogon-chaos/2026-09/gen9championsvgc2026regmcbo3-1760.json (or CHAOS_STATS=<path>). Replay runs on the CPU
(JAX_PLATFORMS=cpu is set for it): it is one battle per process, so CPU cores, not a GPU, make it fast.
Output: reports/regress/<UTC timestamp>/SUMMARY.txt (+ per-corpus results), also copied to reports/regress/LATEST.txt.
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_suite(path):
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, args = (x.strip() for x in line.split("|", 1))
        out.append((name, args.split()))
    return out


def gen(name, args, workers):
    out = ROOT / "traces" / "regress" / name
    meta = out / "meta.json"
    if meta.exists():
        try:
            if json.loads(meta.read_text()).get("argv", [])[1:-2] == args + ["--workers", str(workers), "--events"]:
                return out, "cached"
        except Exception:
            pass
    subprocess.run(["rm", "-rf", str(out)], check=True)
    cmd = ["node", str(ROOT / "oracle/bin/oracle.js"), "gen", *args, "--workers", str(workers), "--events",
           "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return out, (r.stderr.strip().splitlines() or ["?"])[-1]


def replay(trace_dir, out_dir, workers):
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    cmd = [sys.executable, "-m", "harness.cli", "replay", str(trace_dir), "--workers", str(workers), "--out",
           str(out_dir)]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    summ = out_dir / "summary.json"
    if not summ.exists():
        return None, (r.stderr or r.stdout)[-2000:]
    return json.loads(summ.read_text()), None


def failures(out_dir, limit=3):
    rows = []
    res = out_dir / "results.jsonl"
    if not res.exists():
        return rows
    for line in res.read_text().splitlines():
        d = json.loads(line)
        if not d.get("clean") or d.get("events"):
            rows.append(dict(index=d["index"], k=d.get("k"), category=d.get("category"),
                             detail=json.dumps(d.get("detail"))[:400],
                             events=json.dumps(d.get("events"))[:400] if d.get("events") else None))
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(ROOT / "scripts/regress/suite.txt"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--only", default="*", help="comma-separated glob patterns over corpus names")
    ap.add_argument("--skip-gen", action="store_true")
    a = ap.parse_args()
    pats = a.only.split(",")
    suite = [(n, g) for n, g in load_suite(pathlib.Path(a.suite)) if any(fnmatch.fnmatch(n, p) for p in pats)]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = ROOT / "reports" / "regress" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    lines = [f"regress {stamp}  commit {git.strip()}  corpora {len(suite)}  workers {a.workers}", ""]
    bad = 0
    t0 = time.time()
    for name, args in suite:
        tdir = ROOT / "traces" / "regress" / name
        gmsg = "skipped"
        if not a.skip_gen:
            tdir, gmsg = gen(name, args, a.workers)
        s, err = replay(tdir, out_root / name, a.workers)
        if s is None:
            line = f"{name:28s} REPLAY ERROR  gen: {gmsg}\n    {err}"
            bad += 1
        else:
            ok = s.get("CBR") == 1.0 and s.get("TMR") == 1.0 and not s.get("events_mismatch_battles")
            bad += not ok
            line = (f"{name:28s} {'ok  ' if ok else 'FAIL'} battles {s.get('battles', 0):5d}  CBR {s.get('CBR', 0):.4f}"
                    f"  events_mismatch {s.get('events_mismatch_battles', 0)}  buckets {s.get('buckets', {})}"
                    f"  ev {s.get('events_buckets', {})}")
            if not ok:
                for f in failures(out_root / name):
                    line += f"\n    #{f['index']} k={f['k']} {f['category']} {f['detail']}" + (
                        f" events={f['events']}" if f["events"] else "")
        lines.append(line)
        print(line, flush=True)
    lines += ["", f"{len(suite) - bad}/{len(suite)} corpora clean, {time.time() - t0:.0f}s"]
    text = "\n".join(lines) + "\n"
    (out_root / "SUMMARY.txt").write_text(text)
    (ROOT / "reports" / "regress" / "LATEST.txt").write_text(text)
    print(f"\nwrote {out_root / 'SUMMARY.txt'}")


if __name__ == "__main__":
    main()
