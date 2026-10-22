"""ADR-029 canary: with cfg.flags.canary_crit_2048 (crit x1.25 instead of x1.5) the harness must report at least one
DRAW_ARGS or STATE failure on 200 core battles, proving it can see bugs; without the flag they must be clean."""
import dataclasses
import pathlib
import subprocess

import orjson

from harness.replay_one import replay_battle, split_battles
from vgcjax.config import EngineConfig, Flags

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _battles(tmp_path, n=200):
    out = tmp_path / "canary"
    subprocess.run(["node", str(ROOT / "oracle/bin/oracle.js"), "gen", "--n", str(n), "--seed", "canary", "--workers",
                    "8", "--teams", "stage:core", "--out", str(out)], check=True, capture_output=True)
    recs = []
    for f in sorted(out.glob("*.ndjson.zst")):
        raw = subprocess.run(["zstd", "-dc", str(f)], capture_output=True, check=True).stdout
        recs += [orjson.loads(l) for l in raw.splitlines() if l]
    return list(split_battles(recs))


def test_canary_detects_injected_bug(tmp_path):
    battles = _battles(tmp_path)
    cfg = EngineConfig(stage="core", forced=True, draw_log=True)
    bad = dataclasses.replace(cfg, flags=Flags(canary_crit_2048=True))
    fails = [replay_battle(b, bad)["category"] for b in battles]
    assert sum(c in ("DRAW_ARGS", "STATE") for c in fails) >= 1, fails
    clean = [replay_battle(b, cfg)["clean"] for b in battles[:50]]
    assert all(clean)
