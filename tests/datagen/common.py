"""Shared paths and helpers for datagen self-tests (06 §3)."""
from __future__ import annotations

import importlib.util
import json
import os
from types import ModuleType
from typing import Any

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPORT = os.path.join(REPO, "data", "export")
STAGES_DIR = os.path.join(EXPORT, "stages")
FORMAT = "gen9championsvgc2026regmc"
CHAOS = {
    "bo3_1760": os.path.join(REPO, "data/raw/smogon-chaos/2026-09/gen9championsvgc2026regmcbo3-1760.json"),
    "bo1_0": os.path.join(REPO, "data/raw/smogon-chaos/2026-09/gen9championsvgc2026regmc-0.json"),
}
REPLAYS = os.path.join(REPO, "data/raw/ps-replays/2026-10/gen9championsvgc2026regmcbo3_teams.jsonl")
SHEETS = os.path.join(REPO, "data/teams/real_sheets_mc_v1.jsonl")
STAGE_NAMES = ("core", "v0", "v1", "v2")


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def load_tool(name: str, rel_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


needs_raw = pytest.mark.skipif(
    not all(os.path.exists(p) for p in [*CHAOS.values(), REPLAYS]),
    reason="data/raw inputs not present (git-ignored)")
