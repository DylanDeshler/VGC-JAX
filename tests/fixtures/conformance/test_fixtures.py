"""Replay every auto-generated conformance fixture (harness/fixtures.py). A failing fixture blocks merges (07 §10.2)."""
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = sorted(p for p in HERE.rglob("*.json"))


@pytest.mark.parametrize("path", FIXTURES, ids=[str(p.relative_to(HERE)) for p in FIXTURES])
def test_fixture(path):
    from harness.replay_one import replay_battle
    from vgcjax.battle.snapshot import from_ps_snapshot
    from vgcjax.config import EngineConfig
    fx = json.loads(path.read_text())
    header = {"type": "header", "team_sets": fx["team_sets"]}
    dec = dict(fx["decision"], type="decision", k=fx["k"])
    after = {"type": "decision", "projection": fx["expected"], "phase": "x", "requests": [None, None],
             "choices": [None, None], "draws": [], "ties": []}
    records = [header] + [dict(dec, k=i) if i == fx["k"] else {"type": "decision"} for i in range(fx["k"] + 1)]
    records = records[:1] + [dec] + [dict(after, type="final")]
    st = from_ps_snapshot(fx["snapshot"], fx["team_sets"])
    cfg = EngineConfig(stage=fx.get("stage", "core"), forced=True, draw_log=True)
    r = replay_battle(records, cfg, start=(st, 0))
    assert r["clean"], r
