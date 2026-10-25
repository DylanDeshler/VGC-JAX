"""Abilities that veto or edit an effect before it applies: TryBoost, SetStatus, TryAddVolatile, Immunity, Hit,
DragOut (data/abilities.ts; no Champions overrides). Run undispatched as masked passes at the call sites (perf rule,
PROGRESS.md): battle.boost (move_pipeline.boost_v), setStatus (statuses.set_status / try_set_status_v),
addVolatile (conditions.add_volatile), runStatusImmunity (statuses), runEvent('Hit') (spread_move_hit), dragIn.

Handler order inside one runEvent: comparePriority (order, priority, holder speed desc, subOrder); a mon holder sorts
before the field (speed 0). Ties between two mon holders of equal speed are a don't-care kind (contract §6) and are
resolved here by holder rank.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle.util import mget
from vgcjax.data import ids
from vgcjax.effects.abilities import aeff, aid

I32 = jnp.int32
STAT_IDX = {"atk": 0, "def": 1, "spa": 2, "spd": 3, "spe": 4, "accuracy": 5, "evasion": 6}


def _has(ab, *names):
    out = jnp.zeros(jnp.shape(ab), bool)
    for n in names:
        if aid(n) > 0:
            out = out | (ab == aid(n))
    return out


def _allies(st, refs):
    """The two active slots of each ref's side: (refs [T, 2], present [T, 2])."""
    s = jnp.clip(refs, 0, 7) // 4
    act = st.sides.active.astype(I32)[s]                              # [T, 2]
    r = s[:, None] * 4 + jnp.maximum(act, 0)
    ok = (act >= 0) & ~mget(st.mons.fainted, r) & (mget(st.mons.hp, r) > 0)
    return r, ok


# ------------------------------------------------------------------ TryBoost (battle.boost, PS sim/battle.ts:2049)
def try_boost(st, targets, capped, source, effect, has_secondaries, go):
    """runEvent('TryBoost', target, source, effect, boost) on the capped boost dicts (keys = nonzero deltas):
    Clear Body / White Smoke (all drops from others), Hyper Cutter (atk), Keen Eye (accuracy), Big Pecks (def), Flower
    Veil (ally Grass mon, onAllyTryBoost, '-block'), Inner Focus / Own Tempo / Oblivious / Scrappy (Intimidate's atk),
    Guard Dog (priority 2: Intimidate's atk -> boost({atk: 1}, target, target, null, false, isSelf)). Messages:
    '-fail' target 'unboost' [stat] '[from] ability: X' '[of] target' unless effect.secondaries (or Octolock).
    Returns (capped [T, 7], pre) where pre is the message / Guard Dog data emit_boost writes first."""
    m = st.mons
    T = targets.shape[0]
    t = jnp.clip(targets, 0, 7)
    ab = mget(m.ability, t)
    src = jnp.asarray(source, I32)
    other = ~((src >= 0) & (src == targets))
    neg = capped < 0
    intim = jnp.asarray(effect, I32) == aeff("intimidate") if aid("intimidate") > 0 else jnp.zeros((), bool)
    show = ~jnp.asarray(has_secondaries)
    del_all = go & other & _has(ab, "clearbody", "whitesmoke")
    keys = jnp.zeros((T, 7), bool)
    keys = keys | (del_all[:, None] & neg)
    single = {"hypercutter": 0, "keeneye": 5, "bigpecks": 1}
    fail_stat = jnp.full(T, -1, I32)
    fail_on = del_all & jnp.any(neg, axis=-1) & show
    for name, k in single.items():
        hit = go & other & _has(ab, name) & neg[:, k]
        keys = keys.at[:, k].set(keys[:, k] | hit)
        fail_on = fail_on | (hit & show)
        fail_stat = jnp.where(hit, k, fail_stat)
    blockers = go & intim & (capped[:, 0] != 0) & _has(ab, "innerfocus", "owntempo", "oblivious", "scrappy")
    keys = keys.at[:, 0].set(keys[:, 0] | blockers)
    fail_on = fail_on | blockers
    fail_stat = jnp.where(blockers, 0, fail_stat)
    guard = go & intim & (capped[:, 0] != 0) & _has(ab, "guarddog")
    keys = keys.at[:, 0].set(keys[:, 0] | guard)
    # Flower Veil (an ally or self holder): a Grass target, drops from others, not already removed
    fv = jnp.zeros(T, bool)
    fv_of = jnp.full(T, -1, I32)
    if aid("flowerveil") > 0:
        ar, aok = _allies(st, targets)
        grass = jnp.any(mget(m.types, t) == ids.TYPE_GRASS, axis=-1)
        for j in range(2):
            h = aok[:, j] & (mget(m.ability, ar[:, j]) == aid("flowerveil"))
            hit = go & other & grass & h & ~fv & jnp.any(neg & ~keys, axis=-1)
            fv_of = jnp.where(hit, ar[:, j], fv_of)
            fv = fv | hit
        keys = keys | (fv[:, None] & neg)
    capped = jnp.where(keys, 0, capped)
    pre = dict(fail=fail_on, fail_stat=fail_stat, fail_ab=ids.ABILITY_BASE + ab - 1, block=fv & show, block_of=fv_of,
               guard=guard, deleted=keys)
    return capped, pre


def emit_try_boost(st, pre, actor, idx):
    """The TryBoost messages of target idx, before its boost messages."""
    st = EV.emit(st, pre["fail"][idx], EV.EV_FAIL, actor=actor, v1=pre["fail_ab"][idx], v2=pre["fail_stat"][idx],
                 flags=EV.F_FAIL_UNBOOST)
    st = EV.emit(st, pre["block"][idx], EV.EV_BLOCK, actor=actor, target=pre["block_of"][idx], move=aeff("flowerveil"))
    return st


def guard_dog(st, targets, pre):
    """Guard Dog's nested boost({atk: 1}, target, target, null, false, isSelf): '-ability' X 'Guard Dog' 'boost'
    then '-boost' X atk n (capped; nothing at +6 since isSelf). Applied per target in order by the caller."""
    return pre["guard"]


# ------------------------------------------------------------------ SetStatus (setStatus, PS sim/pokemon.ts:1743-1749)
STATUS_IMMUNE = {"limber": ("par",), "insomnia": ("slp",), "vitalspirit": ("slp",), "immunity": ("psn", "tox"),
                 "waterveil": ("brn",), "waterbubble": ("brn",), "purifyingsalt": ("brn", "par", "psn", "tox", "slp",
                                                                                  "frz")}


def set_status_block(st, refs, status, source, source_effect, move_status, go):
    """runEvent('SetStatus', target, source, effect, status) for the abilities (the terrains follow, as field holders
    sort last): the target's own Limber / Insomnia / Vital Spirit / Immunity / Water Veil / Water Bubble / Purifying
    Salt / Leaf Guard (Sun) ('-immune' target '[from] ability: X' when the effect is a move with a primary status),
    and an ally-or-self Sweet Veil (slp, '-block' target 'ability: Sweet Veil' '[of] holder', null) / Flower Veil
    (Grass target, from another source, not Yawn; '-block' for a move without secondaries). Returns (blocked,
    immune_msg, block_msg, block_eff, block_of, winner speed) per ref. The field's handlers (terrains: speed 0,
    subOrder 5) run before a mon holder whose cached speed is <= 0 (Trick Room stores negative speeds)."""
    from vgcjax.effects import weather as WE
    m = st.mons
    refs = jnp.asarray(refs, I32)
    T = refs.shape[0]
    r = jnp.clip(refs, 0, 7)
    ab = mget(m.ability, r)
    status = jnp.broadcast_to(jnp.asarray(status, I32), (T,))
    own = jnp.zeros(T, bool)
    for name, sts in STATUS_IMMUNE.items():
        if aid(name) > 0:
            hit = ab == aid(name)
            any_s = jnp.zeros(T, bool)
            for s_ in sts:
                any_s = any_s | (status == getattr(ids, f"COND_{s_.upper()}"))
            own = own | (hit & any_s)
    if aid("leafguard") > 0:
        own = own | ((ab == aid("leafguard")) & jax_sun(st, r) & (status > 0))
    own = go & own & (status > 0)
    # ally / self veils (onAllySetStatus); all mon holders sort together by speed (desc), the first veto wins
    own_spd = mget(m.speed, r).astype(I32)
    best = jnp.where(own, own_spd, -(1 << 20))              # speed of the winning row so far
    win_own = own
    blk = jnp.zeros(T, bool)
    blk_eff = jnp.full(T, -1, I32)
    blk_of = jnp.full(T, -1, I32)
    blk_msg = jnp.zeros(T, bool)
    if aid("sweetveil") > 0 or aid("flowerveil") > 0:
        ar, aok = _allies(st, refs)
        grass = jnp.any(mget(m.types, r) == ids.TYPE_GRASS, axis=-1)
        src = jnp.asarray(source, I32)
        eff = jnp.asarray(source_effect, I32)
        is_move = (eff >= ids.MOVE_BASE) & (eff < ids.MOVE_BASE + len(ids.MOVES) - 1)
        mv = jnp.clip(eff - ids.MOVE_BASE + 1, 0, len(ids.MOVES) - 1)
        yawn = eff == (ids.COND_BASE + ids.CONDITIONS.index("yawn") - 1) if "yawn" in ids.CONDITIONS else False
        for j in range(2):
            h, ok = ar[:, j], aok[:, j]
            hab = mget(m.ability, h)
            sv = ok & (hab == aid("sweetveil")) & (status == ids.COND_SLP)
            fv = ok & (hab == aid("flowerveil")) & grass & (src >= 0) & (src != refs) & ~yawn & (status > 0)
            spd = mget(m.speed, h).astype(I32)
            hit = go & (status > 0) & (sv | fv) & (spd > best)    # strictly faster (ties: earlier row, by rank)
            win_own = win_own & ~hit
            blk_msg = jnp.where(hit, sv | (fv & is_move & (st_dex_nsec(mv) == 0)), blk_msg)
            blk_eff = jnp.where(hit, ids.ABILITY_BASE + hab - 1, blk_eff)
            blk_of = jnp.where(hit, h, blk_of)
            blk = blk | hit
            best = jnp.where(hit, spd, best)
    blk = blk & ~win_own
    imm_msg = win_own & (jnp.asarray(move_status, I32) > 0)
    blk_msg = blk_msg & blk
    win_spd = jnp.where(win_own, own_spd, best)
    return own | blk, imm_msg, blk_msg, blk_eff, blk_of, win_spd


_NSEC = [None]


def st_dex_nsec(mv):
    """move.secondaries is set (static table, host side to avoid threading dex)."""
    if _NSEC[0] is None:
        from vgcjax.core.dex import load_tables_np
        _NSEC[0] = jnp.asarray(load_tables_np()["move_nsec"])
    return _NSEC[0][mv]


def jax_sun(st, r):
    from vgcjax.effects import weather as WE
    return WE.mon_weather(st, r) == WE.wid("sunnyday")


def emit_set_status_block(st, ref, imm_msg, blk_msg, blk_eff, blk_of, ab_eff):
    st = EV.emit(st, imm_msg, EV.EV_IMMUNE, actor=ref, v1=ab_eff)
    return EV.emit(st, blk_msg, EV.EV_BLOCK, actor=ref, target=blk_of, move=blk_eff)


# ------------------------------------------------------------------ TryAddVolatile / Immunity
def add_volatile_block(st, ref, c):
    """runEvent('TryAddVolatile'): Inner Focus (flinch) / Own Tempo (confusion) return null silently (the ability
    holder sorts before the terrains). Returns blocked."""
    ab = mget(st.mons.ability, jnp.clip(ref, 0, 7))
    out = jnp.zeros((), bool)
    for name, cond in (("innerfocus", "flinch"), ("owntempo", "confusion")):
        if aid(name) > 0 and cond in ids.CONDITIONS:
            out = out | ((ab == aid(name)) & (c == ids.CONDITIONS.index(cond)))
    return out


def status_immunity(st, refs, key):
    """runEvent('Immunity', pokemon, null, null, type): Magma Armor (frz) and Sand Force (sandstorm) return false."""
    ab = mget(st.mons.ability, jnp.clip(refs, 0, 7))
    out = jnp.zeros(jnp.shape(ab), bool)
    if aid("magmaarmor") > 0:
        out = out | ((ab == aid("magmaarmor")) & (key == ids.COND_FRZ))
    if aid("sandforce") > 0 and "sandstorm" in ids.CONDITIONS:
        out = out | ((ab == aid("sandforce")) & (key == ids.CONDITIONS.index("sandstorm")))
    return out
