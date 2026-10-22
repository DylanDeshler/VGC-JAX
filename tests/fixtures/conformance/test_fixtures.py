"""Replay every auto-generated conformance fixture (harness/fixtures.py). A failing fixture blocks merges (07 §10.2)."""
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = sorted(p for p in HERE.rglob("*.json"))


@pytest.mark.parametrize("path", FIXTURES, ids=[str(p.relative_to(HERE)) for p in FIXTURES])
def test_fixture(path):
    from harness.fixtures import replay_fixture
    r = replay_fixture(json.loads(path.read_text()))
    assert r["clean"], r
