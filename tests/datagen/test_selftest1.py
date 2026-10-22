"""Datagen self-test 1 (06 §3): random legal sets vs constructed PS Pokemon — storedStats/maxhp (via core.stats),
maxpp, types and weight from the generated tables."""
import numpy as np

from harness.oracle_client import chunks, default_client
from vgcjax.core import stats as S
from vgcjax.core.dex import load_tables_np
from vgcjax.data import ids

STATS6 = ["hp", "atk", "def", "spa", "spd", "spe"]
TYPE_NAMES = ["", "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying",
              "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy", "???"]


def test_random_sets_match_ps():
    t = load_tables_np()
    rng = np.random.default_rng(11)
    species = np.flatnonzero(t["supported_v2_species"])
    moves = np.flatnonzero(t["supported_v2_move"])
    sets, meta = [], []
    for _ in range(1000):
        sp = int(rng.choice(species))
        mv = [int(x) for x in rng.choice(moves, 4, replace=False)]
        nat = int(rng.integers(0, ids.NNAT))
        sp6 = np.zeros(6, int)
        for _ in range(int(rng.integers(0, 67))):
            k = int(rng.integers(0, 6)); sp6[k] = min(32, sp6[k] + 1)
        sets.append({"species": ids.SPECIES[sp], "ability": "Levitate", "item": "", "moves": [ids.MOVES[m] for m in mv],
                     "nature": ids.NATURES[nat], "evs": dict(zip(STATS6, map(int, sp6))), "gender": "M", "level": 50})
        meta.append((sp, mv, nat, sp6))
    res = []
    for part in default_client().map("stats", [{"sets": c} for c in chunks(sets, 120)]):
        res += part
    for (sp, mv, nat, sp6), r in zip(meta, res):
        hp, s5 = S.calc_stats(t["species_base_stats"][sp], sp6, t["nature_plus"][nat], t["nature_minus"][nat])
        assert int(hp) == r["maxhp"]
        assert [int(x) for x in s5] == [r["storedStats"][k] for k in STATS6[1:]]
        assert [int(t["move_pp"][m]) for m in mv] == r["maxpp"]
        assert [TYPE_NAMES[x] for x in t["species_types"][sp] if x > 0] == r["types"]
        assert int(t["species_weight_hg"][sp]) == r["weighthg"]
