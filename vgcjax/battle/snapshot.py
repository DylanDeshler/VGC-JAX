"""`from_ps_snapshot`: PS serialized battle (State.serializeBattle) -> engine State (07 §7, M2.7).

The only place outside harness/ that parses PS JSON (02 §4.3). Input is the oracle's snapshot payload
{state, ident, turn_state} (oracle/lib/trace.js snapshotOf) plus the battle's team sets. PS Pokemon refs in a
snapshot are position-based: "[Pokemon:p2b]" = side p2, side.pokemon[1] in that snapshot.
"""
from __future__ import annotations

import numpy as np

from vgcjax.battle import state as S
from vgcjax.data import ids
from vgcjax.interop.team import team_from_sets

TYPE_IDX = {t: i + 1 for i, t in enumerate(["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
                                            "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark",
                                            "Steel", "Fairy"])} | {"???": 19}
QKIND = {"move": S.Q_MOVE, "switch": S.Q_SWITCH, "instaswitch": S.Q_INSTASWITCH, "runSwitch": S.Q_RUN_SWITCH,
         "megaEvo": S.Q_MEGA, "beforeTurn": S.Q_BEFORE_TURN, "beforeTurnMove": S.Q_BEFORE_TURN_MOVE,
         "priorityChargeMove": S.Q_PRIORITY_CHARGE, "revivalblessing": S.Q_REVIVAL, "residual": S.Q_RESIDUAL,
         "start": S.Q_START}
_I = {name: {x: i for i, x in enumerate(getattr(ids, name))} for name in ("SPECIES", "MOVES", "ABILITIES", "ITEMS",
                                                                           "CONDITIONS")}
BOOST7 = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
STAT5 = ["atk", "def", "spa", "spd", "spe"]
MR = {True: S.MR_TRUE, False: S.MR_FALSE, None: S.MR_NULL}


def _strip(ref: str | None) -> str | None:
    """'[Species:falinks]' -> 'falinks'."""
    if ref is None:
        return None
    if isinstance(ref, str) and ref.startswith("[") and ":" in ref:
        return ref[ref.index(":") + 1:-1]
    return ref


def _move_id(x) -> int:
    if not x:
        return 0
    if isinstance(x, dict):
        x = x.get("move") or x.get("id")
    return _I["MOVES"][_strip(x)]


class _Refs:
    """Resolve position-based Pokemon refs of one snapshot to engine refs (side*4 + b)."""

    def __init__(self, state: dict, ident):
        self.by_species = [{e["species"]: b for b, e in enumerate(ident[s])} for s in range(2)]
        self.pos_ref = {}
        for s, side in enumerate(state["sides"]):
            for pos, p in enumerate(side["pokemon"]):
                self.pos_ref[f"p{s + 1}{'abcdef'[pos]}"] = s * 4 + self.by_species[s][p["set"]["species"]]

    def __call__(self, ref) -> int:
        if not ref:
            return -1
        return self.pos_ref.get(_strip(ref), -1)


def from_ps_snapshot(snap: dict, team_sets) -> S.State:
    ps = snap["state"]
    ident = snap["ident"]
    ts = snap.get("turn_state") or {}
    refs = _Refs(ps, ident)
    team = team_from_sets(team_sets)
    st = S.empty_state(team)
    st = _np_tree(st)
    m = st.mons
    sides = st.sides
    for s, side in enumerate(ps["sides"]):
        sides.brought_sheet_idx[s] = [e["sheet"] for e in ident[s]]
        sides.ps_order[s] = [refs.by_species[s][p["set"]["species"]] for p in side["pokemon"]]
        for slot, a in enumerate(side["active"]):
            r = refs(a)
            sides.active[s, slot] = r % 4 if r >= 0 else -1
        sides.pokemon_left[s] = side["pokemonLeft"]
        sides.total_fainted[s] = side["totalFainted"]
        sides.fainted_this_turn[s] = bool(side.get("faintedThisTurn"))
        sides.fainted_last_turn[s] = bool(side.get("faintedLastTurn"))
        for p in side["pokemon"]:
            b = refs.by_species[s][p["set"]["species"]]
            _mon(m, s, b, p, ident[s][b]["sheet"])
    meta = st.meta
    rs = ps.get("requestState", "")
    meta_vals = dict(
        phase=S.DONE if ps.get("ended") else {"move": S.MOVE, "switch": S.SWITCH, "teampreview": S.PREVIEW}.get(rs, S.MOVE),
        turn=ps["turn"], mid_turn=bool(ps.get("midTurn")), effect_seq=ps.get("effectOrder", 0),
        last_successful_move=_move_id(_strip(ps.get("lastSuccessfulMoveThisTurn"))),
    )
    for k, v in meta_vals.items():
        getattr(meta, k)[...] = v
    if ps.get("ended"):
        w = ps.get("winner") or ""
        meta.winner[...] = 0 if w == "p1" else 1 if w == "p2" else 2
    fq = [refs(f["target"]) for f in ps.get("faintQueue", [])]
    meta.faint_queue[:len(fq)] = fq
    meta.faint_queue_len[...] = len(fq)
    _queue(st, ps.get("queue", []), refs, ts.get("prepend", []))
    sc = st.scratch
    ords = ts.get("ords") or {}
    if ts.get("ordTurn") == ps["turn"]:
        sc.select_ord[...] = ords.get("select", 0)
        sc.runswitch_ord[...] = ords.get("runswitch", 0)
        sc.fieldevent_ord[...] = ords.get("fieldevent", 0)
        sc.eachevent_ord[...] = ords.get("eachevent", 0)
        sc.mulhook_ord[...] = ords.get("mulhook", 0)
        sc.insert_ord[...] = ords.get("insert", 0)
        for r, n in ts.get("moveActs", []):
            if r >= 0:
                sc.move_actions[r // 4, r % 4] = n
    _field(st, ps["field"])
    import jax
    return jax.tree_util.tree_map(lambda x: np.asarray(x), st)


def _np_tree(st):
    import jax
    return jax.tree_util.tree_map(lambda x: np.array(x), st)


def _mon(m, s, b, p, sheet):
    w = lambda name, v: getattr(m, name).__setitem__((s, b), v)  # noqa: E731
    w("sheet_idx", sheet)
    w("species", _I["SPECIES"][_strip(p["species"])])
    w("base_species", _I["SPECIES"][_strip(p["baseSpecies"])])
    types = [TYPE_IDX[t] for t in p["types"]] + [-1, -1]
    m.types[s, b] = types[:2]
    w("added_type", TYPE_IDX.get(p.get("addedType") or "", 0))
    m.stats[s, b] = [p["storedStats"][k] for k in STAT5]
    w("hp", p["hp"])
    w("maxhp", p["maxhp"])
    w("ability", _I["ABILITIES"][p["ability"]])
    w("base_ability", _I["ABILITIES"][p["baseAbility"]])
    w("ability_seq", (p.get("abilityState") or {}).get("effectOrder", 0))
    w("item", _I["ITEMS"][p["item"]] if p["item"] else 0)
    w("item_seq", (p.get("itemState") or {}).get("effectOrder", 0))
    w("last_item", _I["ITEMS"][p["lastItem"]] if p.get("lastItem") else 0)
    nat = p["set"].get("nature") or "Serious"
    w("nature", ids.NATURES.index(nat.lower()))
    evs = p["set"].get("evs") or {}
    m.sp[s, b] = [evs.get(k, 0) for k in ["hp", "atk", "def", "spa", "spd", "spe"]]
    for j, ms in enumerate(p["moveSlots"][:4]):
        m.move_ids[s, b, j] = _I["MOVES"][ms["id"]]
        m.pp[s, b, j] = ms["pp"]
        m.maxpp[s, b, j] = ms["maxpp"]
        d = ms.get("disabled")
        m.move_disabled[s, b, j] = 2 if d == "hidden" else 1 if d else 0
        m.move_used[s, b, j] = bool(ms.get("used"))
    st_ = p.get("status") or ""
    w("status", _I["CONDITIONS"][st_] if st_ else 0)
    m.boosts[s, b] = [p["boosts"][k] for k in BOOST7]
    for k in ("fainted", "faintQueued", "isActive", "newlySwitched", "beingCalledBack", "isStarted", "forceSwitchFlag",
              "statsRaisedThisTurn", "statsLoweredThisTurn", "usedItemThisTurn"):
        name = {"faintQueued": "faint_queued", "isActive": "is_active", "newlySwitched": "newly_switched",
                "beingCalledBack": "being_called_back", "isStarted": "is_started", "forceSwitchFlag": "force_switch_flag",
                "statsRaisedThisTurn": "stats_raised_this_turn", "statsLoweredThisTurn": "stats_lowered_this_turn",
                "usedItemThisTurn": "used_item_this_turn"}.get(k, k)
        w(name, bool(p.get(k)))
    w("active_turns", p.get("activeTurns", 0))
    w("active_move_actions", p.get("activeMoveActions", 0))
    sf = p.get("switchFlag")
    w("switch_flag", 0 if not sf else 1 if sf is True else _I["MOVES"][sf] + 2)
    w("speed", p.get("speed", 0))
    w("last_move", _move_id(p.get("lastMove")))
    w("last_move_used", _move_id(p.get("lastMoveUsed")))
    w("last_move_target_loc", p.get("lastMoveTargetLoc") or 0)
    w("move_this_turn", _move_id(p.get("moveThisTurn")))
    w("move_this_turn_result", MR.get(p["moveThisTurnResult"], S.MR_UNDEF) if "moveThisTurnResult" in p else S.MR_UNDEF)
    w("move_last_turn_result", MR.get(p["moveLastTurnResult"], S.MR_UNDEF) if "moveLastTurnResult" in p else S.MR_UNDEF)
    h = p.get("hurtThisTurn")
    w("hurt_this_turn", -1 if h is None else h)
    w("last_damage", p.get("lastDamage", 0))
    w("times_attacked", p.get("timesAttacked", 0))
    tr = p.get("trapped")
    w("trapped", 2 if tr == "hidden" else 1 if tr else 0)
    w("can_mega", bool(p.get("canMegaEvo")))
    w("weight_hg", p["weighthg"])
    w("previously_switched_in", p.get("previouslySwitchedIn", 0))


def _queue(st, queue, refs, prepend):
    q = st.queue
    for i, a in enumerate(queue[:S.QMAX]):
        kind = QKIND[a["choice"]]
        r = refs(a.get("pokemon"))
        side = r // 4 if r >= 0 else (0 if a.get("side") in (None, "[Side:p1]") else 1)
        frac = a.get("fractionalPriority", 0) or 0
        prio = a.get("priority", 0) or 0
        q.valid[i] = True
        q.prepend[i] = i in prepend
        q.kind[i] = kind
        q.order[i] = a.get("order", S.ORDER.get(kind, 200))
        q.frac10[i] = round(frac * 10)
        q.priority10[i] = round(prio * 10)
        q.speed[i] = a.get("speed", 0) or 0
        q.side[i] = side
        q.actor_b[i] = r % 4 if r >= 0 else -1
        q.move_slot[i] = a.get("moveSlot", -1) if a.get("moveSlot") is not None else -1
        mid = a.get("moveid") or (_strip(a["move"]["move"]) if isinstance(a.get("move"), dict) else None)
        q.move_id[i] = _I["MOVES"][mid] if mid else 0
        q.target_loc[i] = a.get("targetLoc", 0) or 0
        q.original_target[i] = refs(a.get("originalTarget"))
        q.mega[i] = bool(a.get("mega"))
        t = refs(a.get("target")) if a["choice"] in ("switch", "instaswitch") else -1
        q.switch_target_b[i] = t % 4 if t >= 0 else -1
        se = a.get("sourceEffect")
        q.source_effect[i] = _move_id(_strip(se)) if isinstance(se, str) and se.startswith("[Move:") else 0
        q.seq[i] = i
    st.scratch.queue_seq[...] = len(queue)


def _field(st, field):
    f = st.field
    for key, arr, state in (("weather", f.weather, f.weather_state), ("terrain", f.terrain, f.terrain_state)):
        v = field.get(key)
        if v:
            arr[...] = _I["CONDITIONS"][v]
            es = field.get(key + "State") or {}
            state[0] = es.get("duration", 0) or 0
            state[1] = es.get("effectOrder", 0)
    bits = 0
    for pw, es in (field.get("pseudoWeather") or {}).items():
        k = _I["CONDITIONS"][pw] - ids.PW_BASE
        bits |= 1 << k
        f.pw_state[k, 0] = es.get("duration", 0) or 0
        f.pw_state[k, 1] = es.get("effectOrder", 0)
    f.pw_bits[...] = bits
