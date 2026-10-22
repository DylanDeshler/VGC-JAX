"""Datagen self-test 2 (06 §3): every stage entity is in the legal pool / census, and every condition
reachable from a stage entity through the census closure is in the stage's condition list."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pytest

from .common import REPO, SHEETS, STAGE_NAMES, load_json, load_tool

KIND_KEY = {"move": "moves", "ability": "abilities", "item": "items"}


@pytest.fixture(scope="module")
def dex_stages() -> Any:
    return load_tool("dex_stages", "tools/datagen/dex_stages.py")


@pytest.mark.parametrize("stage", STAGE_NAMES)
def test_entities_in_legal_pool_and_census(stage: str, stages: dict, legal: dict, census_raw: dict) -> None:
    doc = stages[stage]
    rec = census_raw["records"]
    species_pool = set(legal["species"]) | set(legal["battleFormes"])
    assert set(doc["species"]) <= species_pool
    for kind, key in KIND_KEY.items():
        legal_key = {"move": "moves", "ability": "abilities", "item": "items"}[kind]
        assert set(doc[key]) <= set(legal[legal_key]), (stage, kind)
        assert set(doc[key]) <= set(rec[kind]), (stage, kind)
    assert set(doc["conditions"]) <= set(rec["condition"])
    assert doc["meta"]["format"] == legal["formatId"] == census_raw["format"]


@pytest.mark.parametrize("stage", STAGE_NAMES)
def test_condition_closure(stage: str, stages: dict, census_raw: dict, dex_stages: Any) -> None:
    doc = stages[stage]
    census = dex_stages.Census(census_raw)
    dex = {k: set(doc[key]) for k, key in KIND_KEY.items()}
    conds = set(doc["conditions"])
    assert census.closure(dex) == conds
    # direct references of every stage entity and every stage condition are covered
    rec = census_raw["records"]
    for kind in KIND_KEY:
        for eid in dex[kind]:
            for c in census.cond_refs(kind, eid):
                if c in rec["condition"] and c not in dex_stages.UNREACHABLE_CONDITIONS:
                    assert c in conds, (stage, kind, eid, c)
    for c in conds:
        for k in dex_stages.COND_KEYS:
            for x in rec["condition"][c].get("code_refs", {}).get(k, []):
                if x in rec["condition"] and x not in dex_stages.UNREACHABLE_CONDITIONS:
                    assert x in conds, (stage, c, x)
    assert set(census_raw["engine_conditions"]) <= conds


def test_stages_nested(stages: dict) -> None:
    for a, b in (("v0", "v1"), ("v1", "v2")):
        for key in ("species", "moves", "abilities", "items", "conditions"):
            assert set(stages[a][key]) <= set(stages[b][key]), (a, b, key)


def test_v2_is_whole_legal_pool(stages: dict, legal: dict) -> None:
    v2 = stages["v2"]
    assert set(v2["moves"]) == set(legal["moves"])
    assert set(v2["abilities"]) == set(legal["abilities"])
    assert set(v2["items"]) == set(legal["items"])


def test_core_definition(stages: dict, census_raw: dict, legal: dict) -> None:
    core = stages["core"]
    inert = load_json(os.path.join(REPO, "tools/datagen/core_inert.json"))["abilities"]
    assert core["abilities"] == sorted(inert)
    assert core["items"] == []
    for m in core["moves"]:
        r = census_raw["records"]["move"][m]
        assert r["class"] == "DATA_ONLY" and r["data_refs"] == {} and r["core_refs"] == {}
    for s in core["species"]:
        assert s in legal["species"]
        assert any(a["id"] in inert for a in legal["species"][s]["abilities"])


def test_self_check_counts(stages: dict) -> None:
    """05 §1 self-check numbers at the pinned PS commit."""
    if stages["v0"]["meta"]["ps_commit"] != "d601443":
        pytest.skip("self-check numbers are for PS d601443")
    expect = {"v0": (377, 110, 142, 220), "v1": (454, 172, 160, 376), "v2": (510, 215, 166, 483)}
    for st, (m, a, i, custom) in expect.items():
        c = stages[st]["summary"]["counts"]
        assert (c["moves"], c["abilities"], c["items"]) == (m, a, i), st
        assert stages[st]["summary"]["cost"]["n_custom_entities"] == custom, st
    core = stages["core"]["summary"]["counts"]
    assert (core["moves"], core["abilities"], core["items"], core["species"]) == (197, 6, 0, 20)


def test_real_sheet_corpus_frozen() -> None:
    with open(SHEETS, "rb") as f:
        data = f.read()
    with open(SHEETS[: -len(".jsonl")] + ".sha256") as f:
        recorded = f.read().split()[0]
    assert hashlib.sha256(data).hexdigest() == recorded
    rows = [json.loads(line) for line in data.decode().splitlines()]
    assert [r["sheet_sha256"] for r in rows] == sorted(r["sheet_sha256"] for r in rows)
    assert len({r["sheet_sha256"] for r in rows}) == len(rows)
    for r in rows:
        assert set(r) == {"fills", "n_appearances", "packed", "rating_bucket", "sheet_sha256"}
        assert r["rating_bucket"] is None or r["rating_bucket"] % 100 == 0
        mons = r["packed"].split("]")
        assert len(mons) == 6
        for m in mons:
            assert m.split("|")[7] in ("M", "F", "N"), "gender must always be set"
        assert len(r["fills"]["sp"]) == 6, "SP must be filled for every slot"
