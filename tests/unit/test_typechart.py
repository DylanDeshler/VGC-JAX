"""M1.4: typechart equals PS Dex.getEffectiveness / getImmunity for every attacking type x defender typing in the pool."""
import numpy as np

from harness.oracle_client import default_client
from vgcjax.core import typechart as tc
from vgcjax.core.dex import load_dex, load_tables_np
from vgcjax.data import ids

TYPE_NAMES = ["", "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying",
              "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy", "???"]


def typings():
    t = load_tables_np()
    rows = {tuple(int(x) for x in r) for r in t["species_types"][1:]}
    rows |= {(a, -1) for a in range(1, 20)}
    return sorted(rows)


def test_effectiveness_and_immunity():
    dex = load_dex("v2")
    tys = typings()
    cases, idx = [], []
    for atk in range(1, 20):
        for d in tys:
            cases.append([TYPE_NAMES[atk], [TYPE_NAMES[x] for x in d if x > 0]])
            idx.append((atk, d))
    res = default_client().call("typechart", {"cases": cases})
    atk = np.array([a for a, _ in idx]); dt = np.array([d for _, d in idx])
    eff = np.asarray(tc.effectiveness(dex, atk, dt)); imm = np.asarray(tc.immune(dex, atk, dt))
    np.testing.assert_array_equal(eff, [r["eff"] if r["eff"] is not None else 0 for r in res])
    np.testing.assert_array_equal(imm, [r["immune"] for r in res])


def test_status_immunity():
    dex = load_dex("v2")
    tys = typings()
    cases, idx = [], []
    for k, key in enumerate(ids.IMMUNITY_KEYS):
        for d in tys:
            cases.append([key, [TYPE_NAMES[x] for x in d if x > 0]])
            idx.append((k, d))
    res = default_client().call("typechart", {"cases": cases})
    k = np.array([a for a, _ in idx]); dt = np.array([d for _, d in idx])
    np.testing.assert_array_equal(np.asarray(tc.status_immune(dex, k, dt)), [r["immune"] for r in res])
