"""Core scenario: Struggle (randomNormal) with both foes alive draws a non-degenerate RANDOM_TARGET at execution
getTarget (contract §5). Crafted PS states (all PP 0) are continued in PS under random play and replayed through the
engine with forced outcomes; every battle must be clean and the corpus must contain RANDOM_TARGET draws."""
import random
import subprocess
import pathlib

import pytest

from harness import forced as FO
from harness.oracle_client import default_client
from harness.replay_one import replay_battle
from vgcjax.battle.snapshot import from_ps_snapshot
from vgcjax.config import EngineConfig

ROOT = pathlib.Path(__file__).resolve().parents[3]


def core_battle_headers(n):
    import orjson
    out = ROOT / "scratch" / "struggle_src"
    if not out.exists():
        subprocess.run(["node", str(ROOT / "oracle/bin/oracle.js"), "gen", "--n", str(n), "--seed", "struggle-src",
                        "--workers", "8", "--teams", "stage:core", "--out", str(out)], check=True, capture_output=True)
    from harness.corpus import iter_battles, shard_files
    return [b[0] for f in shard_files(out) for b in iter_battles(f)][:n]


def run(n_scen):
    oc = default_client()
    cfg = EngineConfig(stage="core", forced=True, draw_log=True)
    rng = random.Random(1)
    hits = 0
    for i, h in enumerate(core_battle_headers(n_scen)):
        mons = [{"side": s, "b": b, "pp": [0, 0, 0, 0]} for s in range(2) for b in range(4) if rng.random() < 0.7]
        snap = oc.call("craft", {"teams": h["teams"], "seed": h["seed"], "preview": ["team 1234", "team 1234"],
                                 "mons": mons})["snapshot"]
        cont = oc.call("from_snapshot", {"snapshot": snap, "policy": "random", "policy_seed": i, "stop_after": 6})
        decisions = cont["decisions"]
        for d in decisions:
            hits += sum(1 for x in FO.compared_draws(d) if x[2] == "RANDOM_TARGET")
        records = [dict(h, type="header")] + decisions + ([cont["final"]] if cont["final"] else [])
        st = from_ps_snapshot(snap, h["team_sets"])
        r = replay_battle(records, cfg, start=(st, 0))
        assert r["clean"], (i, r)
    return hits


def test_struggle_random_target():
    assert run(60) >= 20


@pytest.mark.slow
def test_struggle_random_target_many():
    assert run(400) >= 50
