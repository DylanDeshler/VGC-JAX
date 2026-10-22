"""Datagen self-test 4 (06 §3): re-running datagen produces byte-identical outputs."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from .common import CHAOS, EXPORT, FORMAT, REPLAYS, REPO, SHEETS, STAGE_NAMES, STAGES_DIR, needs_raw

LEGAL = os.path.join(EXPORT, "legal_pool.json")
CENSUS = os.path.join(EXPORT, "handler_census.json")


def run(*cmd: str) -> None:
    subprocess.run(list(cmd), cwd=REPO, check=True, capture_output=True)


def same_bytes(a: str, b: str) -> bool:
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def test_legal_pool_stable(tmp_path) -> None:
    out = str(tmp_path / "legal_pool.json")
    run("node", "tools/datagen/legal_pool.js", "--format", FORMAT, "--mod", "champions", "--out", out)
    assert same_bytes(out, LEGAL)


def test_census_stable(tmp_path) -> None:
    out = str(tmp_path / "handler_census.json")
    run("node", "tools/datagen/census.js", "--mod", "champions", "--legal", LEGAL, "--out", out)
    assert same_bytes(out, CENSUS)


@pytest.mark.slow
@needs_raw
def test_real_sheets_stable(tmp_path) -> None:
    out = str(tmp_path / os.path.basename(SHEETS))
    run(sys.executable, "-I", "tools/datagen/teams/real_sheets.py", "--format", FORMAT, "--mod", "champions",
        "--replays", REPLAYS, "--chaos", CHAOS["bo3_1760"], "--chaos", CHAOS["bo1_0"], "--legal", LEGAL, "--out", out)
    stem, ref = out[: -len(".jsonl")], SHEETS[: -len(".jsonl")]
    for ext in (".jsonl", ".sha256", ".meta.json"):
        assert same_bytes(stem + ext, ref + ext), ext


@pytest.mark.slow
@needs_raw
def test_dex_stages_stable(tmp_path) -> None:
    run(sys.executable, "-I", "tools/datagen/dex_stages.py", "--format", FORMAT, "--legal", LEGAL, "--census", CENSUS,
        "--sheets", SHEETS, "--core-inert", "tools/datagen/core_inert.json",
        "--chaos", f"bo3_1760={CHAOS['bo3_1760']}", "--chaos", f"bo1_0={CHAOS['bo1_0']}", "--out-dir", str(tmp_path))
    for s in STAGE_NAMES:
        assert same_bytes(str(tmp_path / f"{s}.json"), os.path.join(STAGES_DIR, f"{s}.json")), s
