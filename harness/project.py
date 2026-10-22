"""Canonical projection of an engine state (07 §5): `engine_to_json` mirrors oracle/lib/project.js.

Host-side, one battle at a time (numpy). The on-device diff (07 §5.2 `ps_to_expected`) comes with the batched replay.
"""
from __future__ import annotations

import numpy as np

from vgcjax.battle import state as S
from vgcjax.data import ids

TYPE_NAMES = ["", "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying",
              "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy", "???"]
STAT5 = ["atk", "def", "spa", "spd", "spe"]
BOOST7 = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]


def _np(tree):
    import jax
    return jax.tree_util.tree_map(np.asarray, tree)


def _pokemon(st, side: int, b: int) -> dict:
    m = st.mons
    g = lambda arr: arr[side, b]  # noqa: E731
    pos = -1
    for slot in range(2):
        if int(st.sides.active[side, slot]) == b:
            pos = slot
    species = ids.SPECIES[int(g(m.species))]
    types = [TYPE_NAMES[int(t)] for t in g(m.types) if int(t) > 0]
    if int(g(m.added_type)) > 0:
        types.append(TYPE_NAMES[int(g(m.added_type))])
    status = int(g(m.status))
    item = int(g(m.item))
    moves = []
    for j in range(4):
        mid = int(g(m.move_ids)[j])
        if mid == 0:
            continue
        moves.append({"id": ids.MOVES[mid], "pp": int(g(m.pp)[j]), "maxpp": int(g(m.maxpp)[j]),
                      "disabled": bool(int(g(m.move_disabled)[j]))})
    last_move = int(g(m.last_move))
    return {
        "id": f"p{side + 1}:{int(g(m.sheet_idx))}",
        "b": b,
        "slot": f"p{side + 1}{'ab'[pos]}" if pos >= 0 and bool(g(m.is_active)) else None,
        "species": species,
        "types": types,
        "hp": int(g(m.hp)), "maxhp": int(g(m.maxhp)), "fainted": bool(g(m.fainted)),
        "status": ids.CONDITIONS[status] if status else "",
        "statusState": None if not status else {},
        "boosts": {k: int(v) for k, v in zip(BOOST7, g(m.boosts))},
        "stats": {k: int(v) for k, v in zip(STAT5, g(m.stats))},
        "volatiles": {},
        "item": ids.ITEMS[item] if item else "",
        "itemState": {} if item else None,
        "lastItem": ids.ITEMS[int(g(m.last_item))] if int(g(m.last_item)) else "",
        "ability": ids.ABILITIES[int(g(m.ability))],
        "abilityState": {},
        "moves": moves,
        "canMegaEvo": bool(g(m.can_mega)),
        "lastMove": ids.MOVES[last_move] if last_move else None,
        "trapped": bool(int(g(m.trapped))),
        "weightkg": int(g(m.weight_hg)) / 10,
    }


def engine_to_json(st: S.State) -> dict:
    st = _np(st)
    meta = st.meta
    winner = int(meta.winner)
    out = {
        "turn": int(meta.turn),
        "ended": bool(int(meta.phase) == S.DONE),
        "winner": None if winner < 0 else ("" if winner == 2 else f"p{winner + 1}"),
        "field": {"weather": "", "weatherState": None, "terrain": "", "terrainState": None, "pseudoWeather": {}},
        "sides": [],
    }
    for side in range(2):
        active = []
        for slot in range(2):
            b = int(st.sides.active[side, slot])
            if b < 0:
                active.append(None)
            else:
                # oracle/lib/context.js monLabel: slot label if active, else side:species (fainted occupant)
                label = (f"p{side + 1}{'ab'[slot]}" if bool(st.mons.is_active[side, b])
                         else f"p{side + 1}:{ids.SPECIES[int(st.mons.species[side, b])]}")
                active.append(f"{label}={ids.SPECIES[int(st.mons.base_species[side, b])]}")
        order = [int(x) for x in st.sides.ps_order[side]]
        out["sides"].append({
            "id": f"p{side + 1}",
            "pokemonLeft": int(st.sides.pokemon_left[side]),
            "sideConditions": {},
            "slotConditions": [{}, {}],
            "active": active,
            "pokemon": [_pokemon(st, side, b) for b in order],
        })
    return out


import json as _json
import pathlib as _pl

FIELDS = _json.loads((_pl.Path(__file__).resolve().parent / "contract" / "projection_fields.json").read_text())


def restrict(proj: dict) -> dict:
    """Keep only the protected compared fields (harness/contract/projection_fields.json)."""
    out = {k: proj.get(k) for k in FIELDS["battle"]}
    out["field"] = {k: (proj.get("field") or {}).get(k) for k in FIELDS["field"]}
    out["sides"] = []
    for s in proj.get("sides", []):
        side = {k: s.get(k) for k in FIELDS["side"] if k != "pokemon"}
        side["pokemon"] = [{k: p.get(k) for k in FIELDS["pokemon"]} for p in s.get("pokemon", [])]
        out["sides"].append(side)
    return out


def compare(engine: dict, ps: dict):
    return diff(restrict(engine), restrict(ps))


def normalise_ps(proj: dict) -> dict:
    """PS-side normalisations that the comparison applies (07 §5.4): fainted-in-slot labels, monLabel quirks."""
    return proj


def diff(a, b, path="", out=None, limit=50):
    """Field-level diff of two projections: list of (path, a, b)."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}" if path else k, out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append((path, a, b))
    return out
