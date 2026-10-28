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


def _eff_state(cond: str, state, presence_slot: int) -> dict:
    """oracle/lib/project.js effState of an engine condition state vector: `duration` (slot 0, when > 0) plus the
    declared fields (vgcjax/effects/conditions.py FIELDS) whose presence bit is set."""
    from vgcjax.effects import conditions as CO
    from vgcjax.effects import registry  # noqa: F401  (field declarations)
    s = [int(x) for x in state]
    out = {}
    if s[0] > 0:
        out["duration"] = s[0]
    for key, slot, kind in CO.FIELDS.get(cond, []):
        if (s[presence_slot] & 0xFF) >> slot & 1:
            v = s[slot]
            out[key] = bool(v) if kind == "bool" else (ids.MOVES[v] if kind == "move" else
                                                        (1.5 ** v if kind == "x1.5" else v))
    return dict(sorted(out.items()))


def _ability_state(ability: int, state) -> dict:
    """abilityState keys declared in vgcjax/effects/abilities.py FIELDS (presence bits in slot 3)."""
    from vgcjax.effects import abilities as AB
    s = [int(x) for x in state]
    out = {}
    for key, slot, kind in AB.FIELDS.get(ids.ABILITIES[ability], []):
        if s[3] >> slot & 1:
            out[key] = bool(s[slot]) if kind == "bool" else s[slot]
    return dict(sorted(out.items()))


def _named_field(st, key: str) -> dict:
    """field.weather / terrain (ConditionId) and its state (presence slot 2)."""
    c = int(getattr(st.field, key))
    if c <= 0:
        return {key: "", f"{key}State": None}
    name = ids.CONDITIONS[c]
    return {key: name, f"{key}State": _eff_state(name, getattr(st.field, f"{key}_state"), 2)}


def _pseudo_weather(st) -> dict:
    bits = int(st.field.pw_bits)
    out = {}
    for k in range(S.NPW):
        if bits >> k & 1:
            name = ids.CONDITIONS[ids.PW_BASE + k]
            out[name] = _eff_state(name, st.field.pw_state[k], 2)
    return dict(sorted(out.items()))


def _side_conditions(st, side: int) -> dict:
    bits = int(st.sides.side_cond_bits[side])
    out = {}
    for k in range(S.NSC):
        if bits >> k & 1:
            name = ids.CONDITIONS[ids.SIDE_BASE + k]
            out[name] = _eff_state(name, st.sides.side_cond_state[side, k], 3)
    return dict(sorted(out.items()))


def _volatiles(st, side: int, b: int) -> dict:
    """PS pokemon.volatiles (effMap): every set volatile bit with its effState."""
    words = [int(w) for w in st.mons.vol_bits[side, b]]
    out = {}
    for k in range(S.NV):
        if words[k // 32] >> (k % 32) & 1:
            name = ids.CONDITIONS[ids.VOL_BASE + k]
            v6 = _vol6(st, side, b, k)
            out[name] = _eff_state(name, v6, 5)
            if name == "twoturnmove":
                out.update(_twoturn_move_volatile(v6))
            if name == "roost":
                # roost.onType sets effectState.typeWas = pokemon.types on every getTypes() call, and
                # oracle/lib/project.js calls getTypes() before projecting volatiles: the species' types
                from vgcjax.core.dex import load_tables_np
                sp = int(st.mons.species[side, b])
                out[name]["typeWas"] = [TYPE_NAMES[int(t)] for t in load_tables_np()["species_types"][sp] if int(t) > 0]
                out[name] = dict(sorted(out[name].items()))
    return dict(sorted(out.items()))


def _vol6(st, side: int, b: int, k: int) -> list:
    """The 6-vector view of a volatile's state from the compact storage (effects/conditions.py vol_state)."""
    fi = int(S.VOL_FIDX[k])
    fx = [int(x) for x in st.mons.vol_fx[side, b, fi]] if fi >= 0 else [0, 0, 0, 0]
    return [int(st.mons.vol_dur[side, b, k]), fx[0], fx[1], fx[2], 0, fx[3]]


def _twoturn_move_volatile(s) -> dict:
    """The move volatile twoturnmove.onStart adds (attacker.addVolatile(move.id), effects/locked.py): {targetLoc} plus
    the move's condition duration (exists-false conditions have none). Kept in twoturnmove's raw slots 2 / 3."""
    from vgcjax.effects import locked as LK
    s = [int(x) for x in s]
    if not s[LK.TT_PRESENT]:
        return {}
    from vgcjax.core.dex import load_tables_np
    mv = ids.MOVES[s[1]]
    out = {"targetLoc": s[LK.TT_LOC]}
    dur = int(load_tables_np()["move_cond_duration"][s[1]])
    if dur > 0:
        out["duration"] = s[0]
    return {mv: dict(sorted(out.items()))}


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
        "statusState": None if not status else _eff_state(
            ids.CONDITIONS[(int(g(m.status_state)[3]) >> 8) or status], g(m.status_state), 3),
        "boosts": {k: int(v) for k, v in zip(BOOST7, g(m.boosts))},
        "stats": {k: int(v) for k, v in zip(STAT5, g(m.stats))},
        "volatiles": _volatiles(st, side, b),
        "item": ids.ITEMS[item] if item else "",
        "itemState": {} if item else None,
        "lastItem": ids.ITEMS[int(g(m.last_item))] if int(g(m.last_item)) else "",
        "ability": ids.ABILITIES[int(g(m.ability))],
        "abilityState": _ability_state(int(g(m.ability)), g(m.ability_state)),
        "moves": moves,
        "canMegaEvo": bool(g(m.can_mega)),
        "lastMove": ids.MOVES[last_move] if last_move else None,
        "trapped": bool(int(g(m.trapped))),
        "weightkg": int(g(m.weight_hg)) / 10,
        "ext": _mon_ext(m, side, b),
    }


MR_JSON = {S.MR_UNDEF: "undefined", S.MR_NULL: None, S.MR_FALSE: False, S.MR_TRUE: True}


def _mid(x):
    x = int(x)
    return ids.MOVES[x] if x else None


def _mon_ext(m, side, b):
    """Extended non-gating fields (ADR-103 proposal), mirroring oracle/lib/project.js `ext`."""
    g = lambda arr: arr[side, b]  # noqa: E731
    sf = int(g(m.switch_flag))
    h = int(g(m.hurt_this_turn))
    return {
        "activeTurns": int(g(m.active_turns)), "activeMoveActions": int(g(m.active_move_actions)),
        "speed": int(g(m.speed)), "moveThisTurn": _mid(g(m.move_this_turn)) or "",
        "lastMoveUsed": _mid(g(m.last_move_used)), "lastMoveTargetLoc": int(g(m.last_move_target_loc)),
        "moveThisTurnResult": MR_JSON[int(g(m.move_this_turn_result))],
        "moveLastTurnResult": MR_JSON[int(g(m.move_last_turn_result))],
        "hurtThisTurn": None if h < 0 else h, "timesAttacked": int(g(m.times_attacked)),
        "switchFlag": False if sf == 0 else True if sf == 1 else ids.MOVES[sf - 2],
        "forceSwitchFlag": bool(g(m.force_switch_flag)), "isStarted": bool(g(m.is_started)),
        "newlySwitched": bool(g(m.newly_switched)), "previouslySwitchedIn": int(g(m.previously_switched_in)),
        "statsRaisedThisTurn": bool(g(m.stats_raised_this_turn)), "statsLoweredThisTurn": bool(g(m.stats_lowered_this_turn)),
        "usedItemThisTurn": bool(g(m.used_item_this_turn)), "abilityOrder": int(g(m.ability_seq)),
        "itemOrder": int(g(m.item_seq)),
    }


def engine_to_json(st: S.State) -> dict:
    st = _np(st)
    meta = st.meta
    winner = int(meta.winner)
    out = {
        "turn": int(meta.turn),
        "ext": {"effectOrder": int(meta.effect_seq), "midTurn": bool(meta.mid_turn),
                "lastSuccessfulMove": _mid(meta.last_successful_move)},
        "ended": bool(int(meta.phase) == S.DONE),
        "winner": None if winner < 0 else ("" if winner == 2 else f"p{winner + 1}"),
        "field": {**_named_field(st, "weather"), **_named_field(st, "terrain"), "pseudoWeather": _pseudo_weather(st)},
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
            "ext": {"totalFainted": int(st.sides.total_fainted[side]),
                    "faintedThisTurn": bool(st.sides.fainted_this_turn[side]),
                    "faintedLastTurn": bool(st.sides.fainted_last_turn[side])},
            "sideConditions": _side_conditions(st, side),
            "slotConditions": [{}, {}],
            "active": active,
            "pokemon": [_pokemon(st, side, b) for b in order],
        })
    return out


import json as _json
import pathlib as _pl

FIELDS = _json.loads((_pl.Path(__file__).resolve().parent / "contract" / "projection_fields.json").read_text())


def restrict(proj: dict) -> dict:
    """Copy of the protected compared fields (harness/contract/projection_fields.json)."""
    out = {k: proj.get(k) for k in FIELDS["battle"]}
    out["field"] = {k: (proj.get("field") or {}).get(k) for k in FIELDS["field"]}
    out["sides"] = []
    for s in proj.get("sides", []):
        side = {k: s.get(k) for k in FIELDS["side"] if k != "pokemon"}
        side["pokemon"] = [{k: p.get(k) for k in FIELDS["pokemon"]} for p in s.get("pokemon", [])]
        out["sides"].append(side)
    return out


def compare(engine: dict, ps: dict):
    """Protected comparison (projection_fields.json incl. `ext`, contract v2) with the normalisations.json rules."""
    e, p = restrict(engine), restrict(ps)
    _normalise_target_loc(e, p, ps)
    return diff(e, p)


def _normalise_target_loc(e: dict, p: dict, ps_full: dict) -> None:
    """normalisations.json `ext.lastMoveTargetLoc` (in place on the restricted copies)."""
    from vgcjax.core.dex import load_tables_np
    t = load_tables_np()["move_target"]
    for se, sp, side in zip(e.get("sides", []), p.get("sides", []), ps_full.get("sides", [])):
        for me, mp, mon in zip(se.get("pokemon", []), sp.get("pokemon", []), side.get("pokemon", [])):
            xe, xp = (me or {}).get("ext"), (mp or {}).get("ext")
            if xe is None or xp is None:
                continue
            lm = xp.get("lastMoveUsed") or mon.get("lastMove")
            if not lm or (lm in ids.MOVES and ids.TARGETS[int(t[ids.MOVES.index(lm)])] in RANDOM_LOC_TARGETS):
                me["ext"] = dict(xe, lastMoveTargetLoc="random")
                mp["ext"] = dict(xp, lastMoveTargetLoc="random")


def extract_ext(proj: dict) -> dict:
    return {"battle": proj.get("ext"), "sides": [{"ext": s.get("ext"), "mons": [p.get("ext") for p in s.get("pokemon", [])]}
                                                 for s in proj.get("sides", [])]}


RANDOM_LOC_TARGETS = {"randomNormal", "allAdjacentFoes", "allAdjacent", "foeSide", "allies", "scripted"}


def compare_ext(engine: dict, ps: dict):
    """Non-gating comparison of the extended fields (ADR-103 proposal). lastMoveTargetLoc is skipped when the last
    move's target was resolved by a don't-care random foe pick (PS resolveAction for spread / randomNormal moves)."""
    from vgcjax.core.dex import load_tables_np
    import copy
    t = load_tables_np()["move_target"]
    e, p = copy.deepcopy(extract_ext(engine)), copy.deepcopy(extract_ext(ps))   # normalise copies, not the inputs
    for se, sp, side in zip(e["sides"], p["sides"], ps.get("sides", [])):
        for me, mp, mon in zip(se["mons"], sp["mons"], side.get("pokemon", [])):
            lm = (mp or {}).get("lastMoveUsed") or mon.get("lastMove")
            # lastMoveTargetLoc survives clearVolatile while lastMove is cleared: its origin is then unknown
            if not lm or (lm in ids.MOVES and ids.TARGETS[int(t[ids.MOVES.index(lm)])] in RANDOM_LOC_TARGETS):
                if me is not None and mp is not None:
                    me["lastMoveTargetLoc"] = mp["lastMoveTargetLoc"] = "random"
    return diff(e, p)


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
