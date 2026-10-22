"""PublicEvents conformance (03 §4.7): engine event records equal PS's protocol (oracle/lib/protocol.js reduction)
step by step on fresh core battles; the only tolerated difference is ADR-103's unrecorded switch-order tie."""
import pathlib
import subprocess

import orjson

from harness.replay_one import replay_battle, split_battles
from vgcjax.config import EngineConfig

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_events_match_ps(tmp_path):
    out = tmp_path / "ev"
    subprocess.run(["node", str(ROOT / "oracle/bin/oracle.js"), "gen", "--n", "200", "--seed", "events-t1", "--workers",
                    "8", "--teams", "stage:core", "--events", "--out", str(out)], check=True, capture_output=True)
    recs = []
    for f in sorted(out.glob("*.ndjson.zst")):
        recs += [orjson.loads(l) for l in subprocess.run(["zstd", "-dc", str(f)], capture_output=True,
                                                          check=True).stdout.splitlines() if l]
    cfg = EngineConfig(stage="core", forced=True, draw_log=True)
    bad = []
    n_events = 0
    for b in split_battles(recs):
        n_events += sum(len(d.get("events") or []) for d in b if d["type"] == "decision")
        r = replay_battle(b, cfg)
        assert r["clean"], r
        assert r["events_checked"]
        if r["events"] and r["events"]["detail"]["bucket"] != "SWITCH_ORDER(ADR-103)":
            bad.append((b[0]["battle_id"], r["events"]))
    assert n_events > 10_000
    assert not bad, bad[:3]
