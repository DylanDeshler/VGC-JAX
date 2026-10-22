"""M1.3: Champions stats + stages + paralysis + cap + action speed equal the PS `stats` RPC on random sets."""
import numpy as np
import pytest

from harness.oracle_client import chunks, default_client
from vgcjax.core import stats as S
from vgcjax.core.dex import load_tables_np
from vgcjax.data import ids

STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]
N = 1000


def random_sp(rng):
    """SP <= 32 per stat, <= 66 total (CH §1.3)."""
    sp = np.zeros(6, int)
    budget = int(rng.integers(0, 67))
    for _ in range(budget):
        k = int(rng.integers(0, 6))
        if sp[k] < 32:
            sp[k] += 1
    return sp


@pytest.fixture(scope="module")
def cases():
    t = load_tables_np()
    rng = np.random.default_rng(7)
    species = [i for i in range(1, ids.NS) if t["supported_v2_species"][i]]
    sets, mods, meta = [], [], []
    for _ in range(N):
        si = int(rng.choice(species))
        ni = int(rng.integers(0, ids.NNAT))
        sp = random_sp(rng)
        boosts = {k: int(rng.integers(-6, 7)) for k in ["atk", "def", "spa", "spd", "spe"]}
        par = bool(rng.random() < 0.3)
        tr = bool(rng.random() < 0.3)
        sets.append({"species": ids.SPECIES[si], "ability": "Levitate", "item": "", "moves": ["tackle"],
                     "nature": ids.NATURES[ni], "evs": dict(zip(STAT_KEYS, map(int, sp))), "gender": "M",
                     "level": 50})
        mods.append({"boosts": boosts, "status": "par" if par else None, "trickroom": tr})
        meta.append((si, ni, sp, boosts, par, tr))
    oc = default_client()
    res = []
    for a, b in zip(chunks(sets, 200), chunks(mods, 200)):
        res += oc.call("stats", {"sets": a, "mods": b})
    return t, meta, res


def test_stored_stats_and_hp(cases):
    t, meta, res = cases
    si = np.array([m[0] for m in meta]); ni = np.array([m[1] for m in meta]); sp = np.array([m[2] for m in meta])
    maxhp, st = S.calc_stats(t["species_base_stats"][si], sp, t["nature_plus"][ni], t["nature_minus"][ni])
    np.testing.assert_array_equal(np.asarray(maxhp), [r["maxhp"] for r in res])
    np.testing.assert_array_equal(np.asarray(st), [[r["storedStats"][k] for k in STAT_KEYS[1:]] for r in res])


def test_get_stat_and_action_speed(cases):
    t, meta, res = cases
    for (si, ni, sp, boosts, par, tr), r in zip(meta, res):
        stored = np.array([r["storedStats"][k] for k in STAT_KEYS[1:]])
        b7 = np.array([boosts[k] for k in STAT_KEYS[1:]] + [0, 0])
        got = [int(S.get_stat_hookless(stored, b7, k, par)) for k in range(5)]
        assert got == [r["getStat"][k] for k in STAT_KEYS[1:]], (r, boosts, par)
        assert int(S.action_speed(got[4], tr)) == r["actionSpeed"]
