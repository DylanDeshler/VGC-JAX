"""M2.7: from_ps_snapshot(PS snapshot at decision k) projects equal to PS, and the forced replay continued from it
stays clean (exercises queue remainders at mid-turn SWITCH pauses)."""
import random
import subprocess
import pathlib

import orjson
import pytest

from harness.oracle_client import default_client
from harness import project as P
from harness.replay_one import replay_battle, split_battles
from vgcjax.battle.snapshot import from_ps_snapshot
from vgcjax.config import EngineConfig

ROOT = pathlib.Path(__file__).resolve().parents[2]


def corpus(tmp_path, n, seed, policy):
    out = tmp_path / f"snap{seed}"
    subprocess.run(["node", str(ROOT / "oracle/bin/oracle.js"), "gen", "--n", str(n), "--seed", seed, "--workers", "8",
                    "--teams", "stage:core", "--policy", policy, "--out", str(out)], check=True, capture_output=True)
    recs = []
    for f in sorted(out.glob("*.ndjson.zst")):
        recs += [orjson.loads(l) for l in subprocess.run(["zstd", "-dc", str(f)], capture_output=True,
                                                          check=True).stdout.splitlines() if l]
    return list(split_battles(recs))


def run(battles, points, check_continue=True):
    oc = default_client()
    cfg = EngineConfig(stage="core", forced=True, draw_log=True)
    rng = random.Random(0)
    n_pause = 0
    for b in battles:
        h = b[0]
        dec = [r for r in b if r["type"] == "decision"]
        ks = [k for k in range(1, len(dec)) if dec[k]["phase"] == "switch" and dec[k]["turn"] == dec[k - 1]["turn"]]
        others = list(range(1, len(dec)))
        rng.shuffle(others)
        for k in sorted(set(ks[:1] + others[:points])):
            n_pause += k in ks
            snap = oc.call("snapshot", {"teams": h["teams"], "seed": h["seed"], "choices": [d["choices"] for d in dec],
                                        "at": k})["snapshot"]
            st = from_ps_snapshot(snap, h["team_sets"])
            d = P.diff(P.engine_to_json(st), dec[k]["projection"])
            assert not d, (h["battle_id"], k, d[:5])
            if check_continue:
                r = replay_battle(b, cfg, start=(st, k))
                assert r["clean"], (h["battle_id"], k, r)
    return n_pause


def test_snapshot_roundtrip(tmp_path):
    n = run(corpus(tmp_path, 40, "snap-a", "random-weighted"), points=2)
    assert n > 0


@pytest.mark.slow
def test_snapshot_10k(tmp_path):
    # 10 M2.7: 10k snapshots (3 random decision points per battle incl. pivot pauses), projection equality only
    run(corpus(tmp_path, 3400, "snap-b", "random-weighted"), points=3, check_continue=False)
