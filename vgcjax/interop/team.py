"""Team sets (trace header `team_sets`, ids toID'd) -> engine `Team` arrays (03 §3)."""
from __future__ import annotations

import numpy as np

from vgcjax.battle.state import Team
from vgcjax.data import ids

GENDER = {"M": 0, "F": 1, "N": 2, "": 2}
_IDX = {name: {x: i for i, x in enumerate(getattr(ids, name))} for name in ("SPECIES", "MOVES", "ABILITIES", "ITEMS",
                                                                             "NATURES")}


def team_from_sets(team_sets) -> Team:
    """team_sets: [2][6] dicts {species, item, ability, moves, nature, evs[6], gender}."""
    sp = np.zeros((2, 6), np.int16); it = np.zeros((2, 6), np.int16); ab = np.zeros((2, 6), np.int16)
    na = np.zeros((2, 6), np.int8); ge = np.zeros((2, 6), np.int8); mv = np.zeros((2, 6, 4), np.int16)
    spt = np.zeros((2, 6, 6), np.int8)
    for s in range(2):
        for i, x in enumerate(team_sets[s]):
            sp[s, i] = _IDX["SPECIES"][x["species"]]
            it[s, i] = _IDX["ITEMS"][x["item"]] if x["item"] else 0
            ab[s, i] = _IDX["ABILITIES"][x["ability"]]
            na[s, i] = _IDX["NATURES"][x["nature"]]
            ge[s, i] = GENDER[x.get("gender", "")]
            for j, m in enumerate(x["moves"][:4]):
                mv[s, i, j] = _IDX["MOVES"][m]
            spt[s, i] = x["evs"]
    return Team(species=sp, item=it, ability=ab, nature=na, gender=ge, moves=mv, sp=spt)
