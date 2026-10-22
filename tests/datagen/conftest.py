"""Fixtures for datagen self-tests (06 §3)."""
from __future__ import annotations

import os
from typing import Any

import pytest

from .common import EXPORT, STAGE_NAMES, STAGES_DIR, load_json


@pytest.fixture(scope="session")
def legal() -> dict[str, Any]:
    return load_json(os.path.join(EXPORT, "legal_pool.json"))


@pytest.fixture(scope="session")
def census_raw() -> dict[str, Any]:
    return load_json(os.path.join(EXPORT, "handler_census.json"))


@pytest.fixture(scope="session")
def stages() -> dict[str, dict[str, Any]]:
    return {s: load_json(os.path.join(STAGES_DIR, f"{s}.json")) for s in STAGE_NAMES}
