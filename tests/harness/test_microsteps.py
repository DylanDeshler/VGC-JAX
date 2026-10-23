"""ADR-104 option 4: the micro-step budget is not observable. Replaying fresh core battles with micro_steps=1 (every
turn spills over many step calls in phase CONT) must match PS exactly, events included."""
import pathlib
import subprocess

import orjson

from harness.replay_one import replay_battle, split_battles
from vgcjax.config import EngineConfig

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_spillover_matches_ps(tmp_path):
    out = tmp_path / "mp"
    subprocess.run(["node", str(ROOT / "oracle/bin/oracle.js"), "gen", "--n", "100", "--seed", "microsteps-t1", "--workers",
                    "8", "--teams", "stage:core", "--events", "--out", str(out)], check=True, capture_output=True)
    recs = []
    for f in sorted(out.glob("*.ndjson.zst")):
        recs += [orjson.loads(l) for l in subprocess.run(["zstd", "-dc", str(f)], capture_output=True,
                                                          check=True).stdout.splitlines() if l]
    cfg = EngineConfig(stage="core", forced=True, draw_log=True, micro_steps=1)
    for b in split_battles(recs):
        r = replay_battle(b, cfg, check_masks=True)
        assert r["clean"], (b[0]["battle_id"], r)
        assert not r["events"], (b[0]["battle_id"], r["events"])
