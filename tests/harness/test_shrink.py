"""Shrinker (07 §10.2): a canary-detected fixture is reduced in PS + engine, keeps its signature, still replays clean
with the correct engine and still fails with the canary flag."""
import dataclasses
import pathlib
import subprocess

import orjson

from harness import fixtures as FX
from harness.replay_one import replay_battle, split_battles
from harness.shrink import shrink
from vgcjax.config import EngineConfig, Flags

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _size(fx):
    """(non-neutral moves, items, living bench mons, damaged mons) over brought mons."""
    mons = [p for side in fx["snapshot"]["state"]["sides"] for p in side["pokemon"]]
    return (sum(m["id"] != fx.get("neutral_move") for p in mons for m in p["moveSlots"]),
            sum(bool(p["item"]) for p in mons), sum(not p["isActive"] and not p["fainted"] for p in mons),
            sum(0 < p["hp"] < p["maxhp"] for p in mons))


def test_shrink_canary_fixture(tmp_path):
    out = tmp_path / "t"
    subprocess.run(["node", str(ROOT / "oracle/bin/oracle.js"), "gen", "--n", "200", "--seed", "fx-rt", "--workers", "8",
                    "--teams", "stage:core", "--out", str(out)], check=True, capture_output=True)
    recs = []
    for f in sorted(out.glob("*.ndjson.zst")):
        recs += [orjson.loads(l) for l in subprocess.run(["zstd", "-dc", str(f)], capture_output=True,
                                                          check=True).stdout.splitlines() if l]
    cfg = EngineConfig(stage="core", forced=True, draw_log=True)
    bad = dataclasses.replace(cfg, flags=Flags(canary_crit_2048=True))
    for b in split_battles(recs):
        r = replay_battle(b, bad)
        if not r["clean"] and r["category"] in ("STATE", "DRAW_ARGS"):
            fx = FX.build(b, r)
            break
    else:
        raise AssertionError("canary produced no failure")
    small = shrink(fx, bad)
    assert small["shrink_log"], "no edit kept"
    assert small["signature"] == FX.signature(FX.replay_fixture(fx, bad))
    assert sum(_size(small)) < sum(_size(fx)), (_size(fx), _size(small))
    assert FX.replay_fixture(small, cfg)["clean"]
    r_bad = FX.replay_fixture(small, bad)
    assert not r_bad["clean"] and FX.signature(r_bad) == small["signature"]
    assert FX.entities(small)
