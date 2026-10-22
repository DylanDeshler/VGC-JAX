"""M1.5: hook-free damage equals the PS `damage` RPC on random core-subset cases (no items, core abilities,
no side conditions / weather / terrain). Expectations come only from PS."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from harness.oracle_client import chunks, default_client
from vgcjax.battle import damage as D
from vgcjax.core import stats as S
from vgcjax.core.dex import load_dex, load_tables_np
from vgcjax.data import ids

STATS6 = ["hp", "atk", "def", "spa", "spd", "spe"]
B5 = ["atk", "def", "spa", "spd", "spe"]
CORE_ABILITIES = ["battlearmor", "shellarmor", "corrosion", "earlybird", "levitate", "stall"]


def make_cases(n, seed):
    t = load_tables_np()
    rng = np.random.default_rng(seed)
    species = np.flatnonzero(t["supported_core_species"])
    moves = np.flatnonzero(t["supported_core_move"])
    moves = moves[moves != ids.MOVE_STRUGGLE]
    cases, meta = [], []
    for _ in range(n):
        a_sp, d_sp = (int(x) for x in rng.choice(species, 2))
        mv = int(rng.choice(moves))
        side = []
        for sp in (a_sp, d_sp):
            sp6 = np.zeros(6, int)
            for _ in range(int(rng.integers(0, 67))):
                k = int(rng.integers(0, 6))
                sp6[k] = min(32, sp6[k] + 1)
            side.append(dict(sp=sp6, nature=int(rng.integers(0, ids.NNAT)), ability=str(rng.choice(CORE_ABILITIES)),
                             boosts={k: int(rng.integers(-6, 7)) for k in B5}))
        a_types = t["species_types"][a_sp]
        burned = bool(rng.random() < 0.25) and ids.TYPE_FIRE not in a_types
        crit = bool(rng.random() < 0.4)
        roll = int(rng.integers(0, 16))
        spread = bool(rng.random() < 0.3)

        def pset(sp_id, s):
            return {"species": ids.SPECIES[sp_id], "ability": s["ability"], "item": "", "moves": [ids.MOVES[mv]],
                    "nature": ids.NATURES[s["nature"]], "evs": dict(zip(STATS6, map(int, s["sp"]))), "gender": "M",
                    "level": 50}
        cases.append({"attacker": pset(a_sp, side[0]), "defender": pset(d_sp, side[1]), "move": ids.MOVES[mv],
                      "crit": crit, "roll": roll, "spread": spread, "atkBoosts": side[0]["boosts"],
                      "defBoosts": side[1]["boosts"], "atkStatus": "brn" if burned else None})
        meta.append((a_sp, d_sp, mv, side, burned, crit, roll, spread))
    return cases, meta


def engine_damage(meta):
    t = load_tables_np()
    dex = load_dex("core")
    cols = {k: [] for k in ["mv", "as", "ab", "at", "brn", "ds", "db", "dt", "dhp", "dg", "dci", "crit", "roll", "spr"]}
    for a_sp, d_sp, mv, side, burned, crit, roll, spread in meta:
        st = []
        for sp_id, s in ((a_sp, side[0]), (d_sp, side[1])):
            hp, s5 = S.calc_stats(t["species_base_stats"][sp_id], s["sp"], t["nature_plus"][s["nature"]],
                                  t["nature_minus"][s["nature"]])
            st.append((int(hp), np.asarray(s5)))
        d_types = t["species_types"][d_sp]
        cols["mv"].append(mv); cols["as"].append(st[0][1]); cols["ab"].append([side[0]["boosts"][k] for k in B5] + [0, 0])
        cols["at"].append(t["species_types"][a_sp]); cols["brn"].append(burned); cols["ds"].append(st[1][1])
        cols["db"].append([side[1]["boosts"][k] for k in B5] + [0, 0]); cols["dt"].append(d_types)
        cols["dhp"].append(st[1][0])
        cols["dg"].append(side[1]["ability"] != "levitate" and ids.TYPE_FLYING not in d_types)
        cols["dci"].append(side[1]["ability"] in ("battlearmor", "shellarmor"))
        cols["crit"].append(crit); cols["roll"].append(roll); cols["spr"].append(spread)
    c = {k: jnp.asarray(np.array(v)) for k, v in cols.items()}

    @jax.jit
    def f(c):
        return jax.vmap(lambda mv, a_s, a_b, a_t, brn, d_s, d_b, d_t, dhp, dg, dci, crit, roll, spr: D.get_damage_core(
            dex, mv, atk_stats5=a_s, atk_boosts7=a_b, atk_types=a_t, atk_burned=brn, def_stats5=d_s, def_boosts7=d_b,
            def_types=d_t, def_maxhp=dhp, def_grounded=dg, def_crit_immune=dci, crit_draw=crit, roll=roll,
            spread=spr))(c["mv"], c["as"], c["ab"], c["at"], c["brn"], c["ds"], c["db"], c["dt"], c["dhp"], c["dg"],
                         c["dci"], c["crit"], c["roll"], c["spr"])
    dmg, kind, _ = f(c)
    return np.asarray(dmg), np.asarray(kind)


def run(n, seed):
    cases, meta = make_cases(n, seed)
    oc = default_client()
    ps = []
    for part in oc.map("damage", [{"cases": c} for c in chunks(cases, 1000)]):
        ps += part
    dmg, kind = engine_damage(meta)
    exp_num = np.array([x is not None for x in ps])
    got_num = kind == D.DMG_NUMBER
    bad = np.flatnonzero((exp_num != got_num) | (exp_num & (dmg != np.array([x or 0 for x in ps]))))
    assert not len(bad), [(cases[i]["move"], cases[i]["attacker"]["species"], cases[i]["defender"]["species"],
                           ps[i], int(dmg[i]), int(kind[i])) for i in bad[:10]]


def test_damage_core_20k():
    run(20_000, 1)


@pytest.mark.slow
def test_damage_core_100k():
    run(100_000, 2)
