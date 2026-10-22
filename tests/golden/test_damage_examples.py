"""M1.5: the verified research damage examples (research/ps-move/damage-examples.js) x 16 rolls, through the engine's
formula primitives with the hook contributions supplied explicitly (Life Orb 5324, Reflect skipped on crit, ...).
Expectations: PS `damage` RPC."""
import numpy as np

from harness.oracle_client import default_client
from vgcjax.battle import damage as D
from vgcjax.core import fixedpoint as fp
from vgcjax.core import stats as S
from vgcjax.core.dex import load_tables_np
from vgcjax.data import ids

STATS6 = ["hp", "atk", "def", "spa", "spd", "spe"]


def pset(species, item, ability, moves, nature, evs):
    e = {k: 0 for k in STATS6}; e.update(evs)
    return {"species": species, "item": item, "ability": ability, "moves": moves, "nature": nature, "evs": e,
            "gender": "M", "level": 50}


def stat(species, key, sp, nature):
    t = load_tables_np()
    s = ids.SPECIES.index(species.lower().replace("-", ""))
    n = ids.NATURES.index(nature.lower())
    sp6 = np.zeros(6, int); sp6[STATS6.index(key)] = sp
    _, s5 = S.calc_stats(t["species_base_stats"][s], sp6, t["nature_plus"][n], t["nature_minus"][n])
    return int(np.asarray(s5)[STATS6.index(key) - 1])


def test_ex1_spread_earthquake_life_orb():
    a = pset("Garchomp", "Life Orb", "Rough Skin", ["Earthquake"], "Jolly", {"hp": 2, "atk": 32, "spe": 32})
    d = pset("Incineroar", "Leftovers", "Blaze", ["Bulk Up"], "Careful", {"hp": 32, "def": 20, "spd": 14})
    ps = default_client().call("damage", {"cases": [{"attacker": a, "defender": d, "move": "earthquake", "crit": False,
                                                     "roll": r, "spread": True} for r in range(16)]})
    base = D.base_damage(50, 100, stat("Garchomp", "atk", 32, "Jolly"), stat("Incineroar", "def", 20, "Careful"))
    got = [int(D.modify_damage(base, spread=True, pbond_hit2=False, crit=False, roll=r, stab_m4096=6144, type_mod=1,
                               burn=False, final_m4096=5324)) for r in range(16)]
    assert got == ps


def test_ex2_crit_burned_boosted_reflect():
    a = pset("Sneasler", "Life Orb", "Unburden", ["Mach Punch"], "Adamant", {"atk": 32, "spe": 32})
    d = pset("Kingambit", "Leftovers", "Supreme Overlord", ["Swords Dance"], "Impish", {"hp": 32, "def": 32})
    ps = default_client().call("damage", {"cases": [{"attacker": a, "defender": d, "move": "machpunch", "crit": True,
                                                     "roll": r, "atkBoosts": {"atk": -1}, "defBoosts": {"def": 2},
                                                     "atkStatus": "brn", "sideConditions": {"p2": ["reflect"]}}
                                                    for r in range(16)]})
    # crit ignores -1 Atk and +2 Def; Reflect skipped on crit; Life Orb in the ModifyDamage chain
    base = D.base_damage(50, 40, stat("Sneasler", "atk", 32, "Adamant"), stat("Kingambit", "def", 32, "Impish"))
    got = [int(D.modify_damage(base, spread=False, pbond_hit2=False, crit=True, roll=r, stab_m4096=6144, type_mod=2,
                               burn=True, final_m4096=5324)) for r in range(16)]
    assert got == ps


def test_ex3_rain_mystic_water_light_screen():
    a = pset("Pelipper", "Mystic Water", "Drizzle", ["Water Pulse"], "Modest", {"hp": 32, "spa": 32})
    d = pset("Incineroar", "Leftovers", "Blaze", ["Bulk Up"], "Careful", {"hp": 32, "spd": 32})
    ps = default_client().call("damage", {"cases": [{"attacker": a, "defender": d, "move": "waterpulse", "crit": False,
                                                     "roll": r, "weather": "raindance",
                                                     "sideConditions": {"p2": ["lightscreen"]}} for r in range(16)]})
    bp = int(fp.modify_raw(60, fp.mod4096(4915, 4096)))  # Mystic Water BasePower (no Helping Hand in this RPC case)
    base = D.base_damage(50, bp, stat("Pelipper", "spa", 32, "Modest"), stat("Incineroar", "spd", 32, "Careful"))
    got = [int(D.modify_damage(base, spread=False, pbond_hit2=False, crit=False, roll=r, stab_m4096=6144, type_mod=1,
                               burn=False, final_m4096=2732, weather_m4096=6144)) for r in range(16)]
    assert got == ps
