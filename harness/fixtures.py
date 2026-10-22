"""Conformance fixtures (07 §10.2-10.3): turn-isolated failing decisions, auto-generated, never hand-edited.

A fixture holds the PS snapshot at decision k (before choices), the choices and forced draws/ties of decision k,
PS's projection after the step, the failure signature and provenance. tests/fixtures/conformance/test_fixtures.py
replays every fixture through the engine. `emit(..., shrink=True)` reduces the fixture first (harness/shrink.py).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

from harness.oracle_client import default_client

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "conformance"


def signature(result: dict) -> str:
    """Triage signature: category plus the first mismatching field with mon / slot indices erased (so the same bug on a
    different mon, or after shrinking, keeps its signature), or the mismatching draw kinds."""
    cat = result["category"]
    det = result.get("detail")
    if cat == "STATE" and isinstance(det, list) and det:
        return "STATE:" + re.sub(r"\[\d+\]", "[]", str(det[0][0]))
    if cat in ("DRAW_ARGS", "DRAW_MISMATCH") and isinstance(det, dict):
        rows = (det.get("ps_only") or []) + (det.get("engine_only") or [])
        kinds = sorted({str(r[0].get("kind")) for r in rows if r and isinstance(r[0], dict)})
        return f"{cat}:{','.join(kinds)}"
    return str(cat)


def replay_fixture(fx: dict, cfg=None) -> dict:
    """Engine replay of one fixture: build the engine from the snapshot, step decision k, compare draws and the
    projection after the step. Returns harness.replay_one.replay_battle's result dict."""
    from harness.replay_one import replay_battle
    from vgcjax.battle.snapshot import from_ps_snapshot
    from vgcjax.config import EngineConfig
    cfg = cfg or EngineConfig(stage=fx.get("stage", "core"), forced=True, draw_log=True)
    header = {"type": "header", "team_sets": fx["team_sets"]}
    dec = dict(fx["decision"], type="decision", k=fx["k"])
    final = {"type": "final", "projection": fx["expected"]}
    st = from_ps_snapshot(fx["snapshot"], fx["team_sets"])
    return replay_battle([header, dec, final], cfg, start=(st, 0))


def entities(fx: dict) -> list[str]:
    """Catalogue entities still present in a fixture (brought mons only): the triage hint for a shrunk fixture."""
    out = set()
    for s, side in enumerate(fx["snapshot"]["ident"]):
        for e in side:
            ts = fx["team_sets"][s][e["sheet"]]
            out.add(f"species:{ts['species']}")
            out.update(f"move:{m}" for m in ts["moves"] if m and m != fx.get("neutral_move"))
            if ts["item"]:
                out.add(f"item:{ts['item']}")
            if ts["ability"]:
                out.add(f"ability:{ts['ability']}")
    return sorted(out)


def build(battle_records: list, result: dict, run_seed=None) -> dict | None:
    """The turn-isolated fixture dict for a failing battle (None if not isolable)."""
    header = battle_records[0]
    decisions = [r for r in battle_records if r["type"] == "decision"]
    k = result["k"]
    if k is None:
        return None
    if result["category"] == "STATE":
        k = k - 1          # a projection mismatch seen at decision k was produced by step k-1
    if k < 0 or k >= len(decisions):
        return None
    final = battle_records[-1] if battle_records[-1]["type"] == "final" else None
    expected = decisions[k + 1]["projection"] if k + 1 < len(decisions) else (final or {}).get("projection")
    snap = default_client().call("snapshot", {"teams": header["teams"], "seed": header["seed"],
                                              "choices": [d["choices"] for d in decisions], "at": k})["snapshot"]
    dec = decisions[k]
    return {
        "signature": signature(result), "category": result["category"], "detail": result.get("detail"),
        "ps_commit": header["ps_commit"], "oracle_git": header["oracle_git"], "contract_version": header["contract_version"],
        "source_battle_id": header["battle_id"], "source_run_seed": run_seed or header.get("run_seed"), "k": k,
        "stage": header.get("stage") or "core", "team_sets": header["team_sets"], "snapshot": snap,
        "decision": {key: dec[key] for key in ("phase", "turn", "requests", "choices", "draws", "ties", "projection")},
        "expected": expected,
    }


def emit(battle_records: list, result: dict, run_seed=None, shrink=False, cfg=None) -> pathlib.Path | None:
    """Write the (optionally shrunk) turn-isolated fixture for a failing battle; returns its path."""
    fx = build(battle_records, result, run_seed)
    if fx is None:
        return None
    if shrink:
        from harness.shrink import shrink as do_shrink
        fx = do_shrink(fx, cfg)
    fx["entities"] = entities(fx)
    h = hashlib.sha256(json.dumps([fx["signature"], fx["source_battle_id"], fx["k"]]).encode()).hexdigest()[:16]
    out = FIXTURE_DIR / fx["category"] / f"{h}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fx, default=str))
    return out
