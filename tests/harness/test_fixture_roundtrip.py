"""Fixture emission round trip: a canary-detected failure is turned into a turn-isolated fixture that replays clean
with the correct engine and fails with the canary flag (07 §10.2)."""
import dataclasses
import json
import pathlib
import subprocess

import orjson

from harness import fixtures as FX
from harness.replay_one import replay_battle, split_battles
from vgcjax.battle.snapshot import from_ps_snapshot
from vgcjax.config import EngineConfig, Flags

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_fixture_roundtrip(tmp_path, monkeypatch):
    out = tmp_path / "t"
    subprocess.run(["node", str(ROOT / "oracle/bin/oracle.js"), "gen", "--n", "200", "--seed", "fx-rt", "--workers", "8",
                    "--teams", "stage:core", "--out", str(out)], check=True, capture_output=True)
    recs = []
    for f in sorted(out.glob("*.ndjson.zst")):
        recs += [orjson.loads(l) for l in subprocess.run(["zstd", "-dc", str(f)], capture_output=True,
                                                          check=True).stdout.splitlines() if l]
    cfg = EngineConfig(stage="core", forced=True, draw_log=True)
    bad = dataclasses.replace(cfg, flags=Flags(canary_crit_2048=True))
    monkeypatch.setattr(FX, "FIXTURE_DIR", tmp_path / "fx")
    for b in split_battles(recs):
        r = replay_battle(b, bad)
        if not r["clean"] and r["category"] in ("STATE", "DRAW_ARGS"):
            path = FX.emit(b, r)
            break
    else:
        raise AssertionError("canary produced no failure")
    fx = json.loads(path.read_text())
    header = {"type": "header", "team_sets": fx["team_sets"]}
    dec = dict(fx["decision"], type="decision")
    records = [header, dec, {"type": "final", "projection": fx["expected"]}]
    st = from_ps_snapshot(fx["snapshot"], fx["team_sets"])
    r_ok = replay_battle(records, cfg, start=(st, 0)); assert r_ok["clean"], r_ok
    assert not replay_battle(records, bad, start=(st, 0))["clean"]
