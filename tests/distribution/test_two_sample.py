"""07 §11 two-sample checks: post-step HP distributions of the engine (natural RNG) and PS agree on fresh core
states (Bonferroni-corrected p >= 0.001), and the check has power: the ADR-029 canary (crit x1.25) is rejected."""
import dataclasses

import pytest

from harness import distribution as D
from vgcjax.config import EngineConfig, Flags

N_SCEN, N = 12, 4000


@pytest.fixture(scope="module")
def data():
    scs = D.scenarios(N_SCEN, "dist-core-1")
    return [(sc, D.ps_samples(sc, N)) for sc in scs]


@pytest.mark.slow
def test_engine_matches_ps(data):
    ps = [(i, s, b, p) for i, (sc, x) in enumerate(data) for (s, b, p) in D.compare(x, D.engine_samples(sc, N, seed=i))]
    assert len(ps) >= N_SCEN
    worst = min(ps, key=lambda t: t[3])
    assert worst[3] * len(ps) >= 1e-3, worst


@pytest.mark.slow
def test_canary_is_rejected(data):
    bad = dataclasses.replace(EngineConfig(stage="core"), flags=Flags(canary_crit_2048=True))
    ps = [p for i, (sc, x) in enumerate(data) for (_, _, p) in D.compare(x, D.engine_samples(sc, 4 * N, bad, seed=i))]
    assert min(ps) * len(ps) < 1e-3
