"""Move execution (04 §7): runMove -> useMove -> trySpreadMoveHit -> Champions hit loop -> spreadMoveHit.

Core-stage scope (M2.5): data-driven moves only. Hook points are marked `HOOK(Hn)` and are identities until M3.
Per-target work runs over the 3 target slots of a move (`targets` refs, -1 = none) in PS target order.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import damage as D
from vgcjax.battle import events as EV
from vgcjax.battle import faint as F
from vgcjax.battle import results as RS
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset, mset_many, side_of
from vgcjax.core import rng as R
from vgcjax.core import typechart as TC
from vgcjax.data import ids

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32
T = ids
SINGLE_TARGET = (T.TARGET_NORMAL, T.TARGET_ANY, T.TARGET_ADJACENTFOE, T.TARGET_ADJACENTALLY, T.TARGET_RANDOMNORMAL,
                 T.TARGET_ADJACENTALLYORSELF)
MULTIHIT_TABLE = jnp.array([2] * 7 + [3] * 7 + [4] * 3 + [5] * 3, I32)  # ch/scripts.ts:439
NT = 3


def opt_id(name: str) -> int:
    """Id constant for an entity that may be outside the format's pool (-999 never matches)."""
    return getattr(ids, name, -999)


def _in(x, vals):
    out = jnp.zeros(jnp.shape(x), bool)
    for v in vals:
        out = out | (x == v)
    return out


def mflag(dex, move, flag):
    w, b = flag
    return ((dex.move_flags[move, w] >> b) & 1) == 1


# ------------------------------------------------------------------ positions (TL §9)
def position(st, ref):
    """Slot index of an active-slot occupant (incl. fainted), else -1."""
    s = side_of(ref)
    b = jnp.asarray(ref, I32) % 4
    row = st.sides.active[s].astype(I32)
    return jnp.where(row[0] == b, 0, jnp.where(row[1] == b, 1, -1))


def at_loc(st, u, loc):
    """PS: sim/pokemon.ts:791-799 getAtLoc (may return a fainted mon); -1 if none."""
    us = side_of(u)
    loc = jnp.asarray(loc, I32)
    side = jnp.where(loc < 0, us, 1 - us)
    slot = jnp.abs(loc) - 1
    ok = (loc != 0) & (slot >= 0) & (slot < 2)
    b = st.sides.active[side, jnp.clip(slot, 0, 1)].astype(I32)
    return jnp.where(ok & (b >= 0), side * 4 + b, -1)


def valid_target_loc(st, u, loc, ttype):
    """PS: sim/battle.ts:2415-2447 validTargetLoc (doubles)."""
    loc = jnp.asarray(loc, I32)
    src = -(position(st, u) + 1)
    is_self = src == loc
    is_foe = loc > 0
    across = -(3 - loc)
    adj = jnp.where(loc > 0, jnp.abs(across - src) <= 1, jnp.abs(loc - src) == 1)
    out = jnp.where(_in(ttype, (T.TARGET_RANDOMNORMAL, T.TARGET_SCRIPTED, T.TARGET_NORMAL)), adj,
          jnp.where(ttype == T.TARGET_ADJACENTALLY, adj & ~is_foe,
          jnp.where(ttype == T.TARGET_ADJACENTALLYORSELF, (adj & ~is_foe) | is_self,
          jnp.where(ttype == T.TARGET_ADJACENTFOE, adj & is_foe,
          jnp.where(ttype == T.TARGET_ANY, ~is_self, False)))))
    return (loc == 0) | ((jnp.abs(loc) <= 2) & out)


def living_foes(st, u):
    """foe refs in slot order with validity (alive)."""
    fs = 1 - side_of(u)
    refs, ok = [], []
    for slot in range(2):
        b = st.sides.active[fs, slot].astype(I32)
        r = jnp.where(b >= 0, fs * 4 + b, -1)
        refs.append(r)
        ok.append((b >= 0) & (mget(st.mons.hp, r) > 0))
    return jnp.stack(refs), jnp.stack(ok)


def living_ally(st, u):
    """The adjacent living ally (not u) or -1."""
    us = side_of(u)
    p = position(st, u)
    b = st.sides.active[us, jnp.clip(1 - p, 0, 1)].astype(I32)
    r = jnp.where((b >= 0) & (p >= 0), us * 4 + b, -1)
    return jnp.where((r >= 0) & (mget(st.mons.hp, r) > 0) & ~mget(st.mons.fainted, r), r, -1)


def get_random_target(st, dex, rng, u, move, phase, on):
    """PS: sim/battle.ts:2506-2538 getRandomTarget. Spread / side targets only make a nominal (don't-care) draw,
    which the engine does not make (contract §6). Returns (ref | -1, rng)."""
    tt = dex.move_target[move]
    selfish = _in(tt, (T.TARGET_SELF, T.TARGET_ALL, T.TARGET_ALLYSIDE, T.TARGET_ALLYTEAM, T.TARGET_ADJACENTALLYORSELF))
    ally = living_ally(st, u)
    refs, ok = living_foes(st, u)
    single = _in(tt, SINGLE_TARGET)
    key = C.dkey(st, ids.DK_RANDOM_TARGET, actor=u, subject=C.eff_move(move), ord=phase)
    on = jnp.asarray(on)
    # adjacentAlly: sample(adjacentAllies) (one ally at most in doubles -> degenerate, no draw)
    pick, rng = R.sample_ref(rng, key, refs, ok, active=on & ~selfish & (tt != T.TARGET_ADJACENTALLY) & single)
    foe_pick = jnp.where(jnp.any(ok), jnp.where(single, pick, refs[jnp.argmax(ok)]), refs[0])
    out = jnp.where(selfish, u, jnp.where(tt == T.TARGET_ADJACENTALLY, ally, foe_pick))
    return out, rng


def get_target(st, dex, rng, u, move, loc, orig, on):
    """PS: sim/battle.ts:2453-2504 getTarget. Returns (ref | -1, rng)."""
    tt = dex.move_target[move]
    m = st.mons
    tracks = (dex.move_tracks_target[move] == 1) | _in(mget(m.ability, u), (opt_id('ABILITY_STALWART'), opt_id('ABILITY_PROPELLERTAIL')))
    orig_ok = (jnp.asarray(orig) >= 0) & mget(m.is_active, orig)
    smart = dex.move_smart_target[move] == 1
    cur = at_loc(st, u, loc)
    cur_alive = (cur >= 0) & ~mget(m.fainted, cur)
    self_loc = -(position(st, u) + 1)
    own_slot_fail = _in(tt, (T.TARGET_ADJACENTALLY, T.TARGET_ANY, T.TARGET_NORMAL)) & (jnp.asarray(loc) == self_loc)
    futuremove = mflag(dex, move, T.MF_FUTUREMOVE)
    valid = (tt != T.TARGET_RANDOMNORMAL) & valid_target_loc(st, u, loc, tt)
    tgt_fainted = (cur >= 0) & mget(m.fainted, cur)
    tgt_ally = (cur >= 0) & (side_of(cur) == side_of(u))
    need_random = ~(tracks & orig_ok)
    need_random = need_random & ~(smart & cur_alive)
    need_random = need_random & ~(~smart & own_slot_fail)
    direct = ~smart & ~own_slot_fail & valid & ((tgt_fainted & tgt_ally) | cur_alive)
    need_random = need_random & ~direct
    rnd, rng = get_random_target(st, dex, rng, u, move, 1, jnp.asarray(on) & need_random)
    direct_ref = jnp.where(tgt_fainted & tgt_ally, jnp.where(tt == T.TARGET_ADJACENTALLYORSELF, u, cur), cur)
    out = jnp.where(tracks & orig_ok, orig,
          jnp.where(smart, jnp.where(cur_alive, cur, rnd),
          jnp.where(own_slot_fail, jnp.where(futuremove, u, -1),
          jnp.where(direct, direct_ref, rnd))))
    return out, rng


def get_move_targets(st, dex, rng, u, move, target, on):
    """PS: sim/pokemon.ts:812-877 getMoveTargets. Returns (refs[3], smart_still, rng); refs -1 = no target."""
    tt = dex.move_target[move]
    m = st.mons
    foes, fok = living_foes(st, u)
    ally = living_ally(st, u)
    none = jnp.full(NT, -1, I32)
    # allAdjacent: ally first, then foes; allAdjacentFoes: foes
    def pack(cands, oks):
        out = none
        n = jnp.int32(0)
        for c, ok in zip(cands, oks):
            out = out.at[jnp.clip(n, 0, NT - 1)].set(jnp.where(ok, c, out[jnp.clip(n, 0, NT - 1)]))
            n = n + ok.astype(I32)
        return out
    t_adj = pack([ally, foes[0], foes[1]], [ally >= 0, fok[0], fok[1]])
    t_adjfoes = pack([foes[0], foes[1]], [fok[0], fok[1]])
    us = side_of(u)
    own = []
    for slot in range(2):
        b = st.sides.active[us, slot].astype(I32)
        r = jnp.where(b >= 0, us * 4 + b, -1)
        own.append((r, (r >= 0) & (mget(m.hp, r) > 0)))
    t_allies = pack([own[0][0], own[1][0]], [own[0][1], own[1][1]])
    # single target
    t = jnp.asarray(target, I32)
    retarget = (t < 0) | (mget(m.fainted, t) & (side_of(t) != us))
    rnd, rng = get_random_target(st, dex, rng, u, move, 2, jnp.asarray(on) & retarget & _in(tt, SINGLE_TARGET + (
        T.TARGET_SCRIPTED,)))
    t2 = jnp.where(retarget, rnd, t)
    # HOOK(H4) RedirectTarget (not for tracksTarget)
    smart = dex.move_smart_target[move] == 1
    tally = jnp.where(t2 >= 0, living_ally(st, t2), -1)
    smart_two = smart & (tally >= 0) & (tally != u) & (mget(m.hp, t2) > 0)
    smart_ally_only = smart & ~smart_two & (tally >= 0) & (tally != u) & (mget(m.hp, t2) <= 0)
    single = jnp.where(smart_two, none.at[0].set(t2).at[1].set(tally),
                       jnp.where(smart_ally_only, none.at[0].set(tally), none.at[0].set(t2)))
    first = single[0]
    dead = (first < 0) | (mget(m.fainted, first) & ~mflag(dex, move, T.MF_FUTUREMOVE))
    single = jnp.where(dead, none, single)
    out = jnp.where(tt == T.TARGET_ALLADJACENT, t_adj,
          jnp.where(tt == T.TARGET_ALLADJACENTFOES, t_adjfoes,
          jnp.where(tt == T.TARGET_ALLIES, t_allies, single)))
    return out, smart_two, rng


# ------------------------------------------------------------------ boosts (PS: sim/battle.ts:2037-2106)
def boost_math(cur, d, go, order=None, primary=False):
    """Pure core of battle.boost over leading dims: cur, d i32[..., 7], go bool[...]. Returns (new [..., 7],
    capped [..., 7] (zero where not go), msgs). msgs = (on, stat, amount, flags) [..., 7] in the boosts dict's key order (`order`, default
    canonical): one -boost/-unboost per key with the applied amount; a zero change is shown only for primary effects
    (not isSecondary / isSelf). PS :2053-2095."""
    capped = jnp.where((d != 0) & jnp.asarray(go)[..., None], jnp.clip(cur + d, -6, 6) - cur, 0)   # 0 when not go
    order = jnp.arange(7) if order is None else jnp.asarray(order, I32)
    d_o, c_o, cur_o = (jnp.take(x, order, axis=-1) for x in (d, capped, cur))
    on = jnp.asarray(go)[..., None] & (d_o != 0) & ((c_o != 0) | primary)
    flags = jnp.where((d_o < 0) | (cur_o + c_o == -6), EV.F_UNBOOST, 0)
    return cur + capped, capped, (on, jnp.broadcast_to(order, on.shape), jnp.abs(c_o), flags)


def emit_boost(st, msgs, actor, idx=()):
    """Emit the -boost/-unboost records of one boost application (msgs from boost_math, indexed by `idx`)."""
    on, stat, amt, flags = (x[idx] for x in msgs)
    for k in range(7):
        st = EV.emit(st, on[k], EV.EV_BOOST, actor=actor, v1=stat[k], v2=amt[k], flags=flags[k])
    return st


def boost_v(st, targets, deltas7, on, order=None, primary=False):
    """battle.boost (no hooks yet: ChangeBoost / TryBoost / AfterEachBoost join in M3) on distinct refs targets [T]
    at once. Returns (st, result (tags [T], vals [T]), msgs); messages are not emitted (callers interleave them)."""
    m = st.mons
    targets = jnp.asarray(targets, I32)
    T = targets.shape[0]
    hp = mget(m.hp, targets)
    active = mget(m.is_active, targets)
    foe_left = st.sides.pokemon_left[1 - side_of(jnp.clip(targets, 0, 7))] > 0
    cur = mget(m.boosts, targets).astype(I32)
    d = jnp.broadcast_to(jnp.asarray(deltas7, I32), (T, 7))
    go = jnp.asarray(on) & (targets >= 0) & (hp > 0) & active & foe_left
    new, capped, msgs = boost_math(cur, d, go, order, primary)
    success = jnp.any(capped != 0, axis=-1)
    m = m.replace(boosts=mset_many(m.boosts, targets, new, go),
                  stats_raised_this_turn=mset_many(m.stats_raised_this_turn, targets, True,
                                                   go & success & jnp.any(capped > 0, axis=-1)),
                  stats_lowered_this_turn=mset_many(m.stats_lowered_this_turn, targets, True,
                                                    go & success & jnp.any(capped < 0, axis=-1)))
    r = RS.where(hp <= 0, RS.R(RS.NUM, 0), RS.where(~active | ~foe_left, RS.R(RS.BOOL, 0),
                                                    RS.where(success, RS.R(RS.BOOL, 1), RS.R(RS.NULL))))
    return st.replace(mons=m), r, msgs


def boost(st, target, deltas7, on, order=None, primary=False):
    """battle.boost on one target, messages emitted. Returns (st, result)."""
    t = jnp.asarray(target, I32)[None]
    st, r, msgs = boost_v(st, t, jnp.asarray(deltas7)[None], jnp.asarray(on)[None], order, primary)
    st = emit_boost(st, msgs, t[0], 0)
    return st, (r[0][0], r[1][0])


def battle_heal(st, target, amount, on, drain_of=None):
    """PS: sim/battle.ts heal(damage, target, ...): (st, result) with result tag NUM (healed) or BOOL false.
    Message (PS :2293-2305): '-heal' target hp, with '[from] drain' '[of] source' for drain."""
    m = st.mons
    amount = jnp.asarray(amount, I32)
    amount = jnp.where((amount > 0) & (amount <= 1), 1, amount)
    hp = mget(m.hp, target)
    ok = jnp.asarray(on) & (amount != 0) & (hp > 0) & mget(m.is_active, target) & (hp < mget(m.maxhp, target))
    st, healed = F.heal(st, target, amount, ok)
    st = EV.emit_hp(st, ok, EV.EV_HEAL, target, target=-1 if drain_of is None else drain_of,
                    flags=0 if drain_of is None else EV.F_DRAIN)
    r = RS.where(amount == 0, RS.R(RS.NUM, 0), RS.where(ok, RS.R(RS.NUM, healed), RS.R(RS.BOOL, 0)))
    return st, r


# ------------------------------------------------------------------ accuracy (PS: sim/battle-actions.ts:752-816)
def accuracy_step(st, dex, rng, u, move, targets, tv, on, smart=False, spread=False):
    """hitStepAccuracy for all target slots at once (09 J5). Returns (st, ok[3], rng).
    Messages, per target in order: '-immune' target '[ohko]' (PS :768); a miss adds '-miss' user target and marks the
    move line [miss] when not spreadHit (PS :800-806); smartTarget misses are silent."""
    m = st.mons
    acc_base = dex.move_accuracy[move].astype(I32)        # 0 = true
    ohko = dex.move_ohko[move]
    cat = dex.move_category[move]
    tt = dex.move_target[move]
    u_types = mget(m.types, u)
    targets = jnp.asarray(targets, I32)
    live = jnp.asarray(on) & (targets >= 0) & jnp.asarray(tv)
    t_types = mget(m.types, targets)                                              # [3, 2]
    # OHKO: 30 (Sheer Cold 20 for non-Ice users), Sheer Cold fails on Ice targets
    ohko_acc = jnp.where((ohko == 2) & ~jnp.any(u_types == T.TYPE_ICE), 20, 30)
    ohko_immune = (ohko > 0) & (ohko == 2) & jnp.any(t_types == T.TYPE_ICE, axis=-1)
    # stages (HOOK H11 ModifyAccuracy / ModifyBoost = identity)
    b = jnp.clip(mget(m.boosts, u)[5].astype(I32), -6, 6)
    b = jnp.where(dex.move_ignore_evasion[move] == 1, b,
                  jnp.clip(b - mget(m.boosts, targets)[:, 6].astype(I32), -6, 6))
    staged = jnp.where(b > 0, acc_base * (3 + b) // 3, jnp.where(b < 0, acc_base * 3 // (3 - b), acc_base))
    acc = jnp.where(ohko > 0, ohko_acc, staged)
    always = ((acc_base == 0) & (ohko == 0)) | ((tt == T.TARGET_SELF) & (cat == T.CAT_STATUS))
    key = C.dkey(st, ids.DK_ACCURACY, actor=u, target=targets, subject=C.eff_move(move), hit=0)
    hit, rng = R.chance_v(rng, key, acc, 100, active=live & ~always & ~ohko_immune)
    ok = jnp.where(always, True, hit & ~ohko_immune)
    oimm = live & ~always & ohko_immune
    miss = live & ~ok & ~oimm & ~smart
    for j in range(NT):
        st = EV.emit(st, oimm[j], EV.EV_IMMUNE, actor=targets[j], flags=EV.F_OHKO)
        st = EV.attr_last_move(st, miss[j] & ~spread, EV.F_MISS)
        st = EV.emit(st, miss[j], EV.EV_MISS, actor=u, target=targets[j])
    return st, ok, rng


# ------------------------------------------------------------------ getDamage for all targets
def get_damage(st, dex, rng, cfg, u, move, targets, spread, hit_no, on):
    """PS getDamage for each target slot at once (09 J5): returns (st, (tags[3], vals[3]), rng). Struggle is typeless
    via its onModifyMove."""
    m = st.mons
    targets = jnp.asarray(targets, I32)
    on = jnp.asarray(on)
    ab_t = mget(m.ability, targets)
    crit_immune = _in(ab_t, (T.ABILITY_BATTLEARMOR, T.ABILITY_SHELLARMOR))   # HOOK(H13) CriticalHit
    crit_ratio = jnp.clip(dex.move_crit_ratio[move].astype(I32), 0, 4)
    will_crit = dex.move_will_crit[move] == 1
    bp = dex.move_bp[move].astype(I32)
    fixed = dex.move_fixed_damage[move]
    ohko = dex.move_ohko[move] > 0
    rolls_damage = on & (bp > 0) & ~ohko & (fixed == 0)
    ckey = C.dkey(st, ids.DK_CRIT, actor=u, target=targets, subject=C.eff_move(move), hit=hit_no)
    crit_draw, rng = R.chance_v(rng, ckey, 1, D.crit_denominator(crit_ratio),
                                active=rolls_damage & ~will_crit & (crit_ratio > 0))
    rkey = C.dkey(st, ids.DK_DAMAGE_ROLL, actor=u, target=targets, subject=C.eff_move(move), hit=hit_no)
    roll, rng = R.randint_v(rng, rkey, 0, 16, active=rolls_damage)
    t_types = mget(m.types, targets)
    grounded = ~jnp.any(t_types == T.TYPE_FLYING, axis=-1) & (ab_t != T.ABILITY_LEVITATE)
    is_struggle = move == T.MOVE_STRUGGLE
    # Struggle: onModifyMove type '???' (data/moves.ts struggle) -> treat as typeless via a typeless attacker types
    typeless = jnp.array([-1, -1], I8)
    def_types = jnp.where(is_struggle, typeless[None, :], t_types)
    atk = dict(atk_stats5=mget(m.stats, u), atk_boosts7=mget(m.boosts, u),
               atk_types=jnp.where(is_struggle, typeless, mget(m.types, u)), atk_burned=mget(m.status, u) == T.COND_BRN)

    def one(ds, db, dt, dmx, dg, dci, cd, r):
        return D.get_damage_core(dex, move, **atk, def_stats5=ds, def_boosts7=db, def_types=dt, def_maxhp=dmx,
                                 def_grounded=dg, def_crit_immune=dci, crit_draw=cd, roll=r, spread=spread,
                                 canary_crit=cfg.flags.canary_crit_2048)
    dmg, kind, crit = jax.vmap(one)(mget(m.stats, targets), mget(m.boosts, targets), def_types, mget(m.maxhp, targets),
                                    grounded, crit_immune, crit_draw, roll)
    # [C] modifyDamage messages (data/mods/champions/scripts.ts:266-284): effectiveness then crit, per target
    formula = rolls_damage & (kind == D.DMG_NUMBER)
    eff = jax.vmap(lambda tt: TC.effectiveness(dex, dex.move_type[move], tt))(t_types)
    tm = jnp.clip(jnp.where(is_struggle, 0, eff), -6, 6)
    for j in range(NT):
        st = EV.emit(st, formula[j] & (tm[j] > 0), EV.EV_SUPER, actor=targets[j], v1=jnp.minimum(tm[j], 2))
        st = EV.emit(st, formula[j] & (tm[j] < 0), EV.EV_RESIST, actor=targets[j], v1=jnp.minimum(-tm[j], 2))
        st = EV.emit(st, formula[j] & crit[j], EV.EV_CRIT, actor=targets[j])
    res = RS.where(kind == D.DMG_NUMBER, RS.R(RS.NUM, dmg), RS.where(kind == D.DMG_FALSE, RS.R(RS.BOOL, 0),
                                                                     RS.where(kind == D.DMG_UNDEFINED, RS.R(RS.UNDEF),
                                                                              RS.R(RS.NUM, 0))))
    return st, res, rng


# ------------------------------------------------------------------ spreadMoveHit (ch/scripts.ts:314-423), core
def spread_move_hit(st, dex, rng, cfg, u, move, targets, tvalid, spread, hit_no, on):
    """One hit on the current targets (ch/scripts.ts:314-423, core), vectorised over the target slots (09 J5): reads
    are batched gathers, writes one masked select per field (targets are distinct mons), and PS's per-target message
    order is reproduced by emitting afterwards. Returns (st, rng, (tags[3], vals[3]))."""
    on = jnp.asarray(on)
    targets = jnp.asarray(targets, I32)
    tags = jnp.full(NT, RS.BOOL, I32)
    vals = jnp.ones(NT, I32)
    tv = jnp.asarray(tvalid) & on
    # TryHit (move onTryHit; none in core) / TryPrimaryHit (Substitute; HOOK H12) -> true

    # 1. getSpreadDamage
    st, res, rng = get_damage(st, dex, rng, cfg, u, move, targets, spread, hit_no, tv)
    tags, vals = RS.where(tv, res, (tags, vals))
    tv = tv & ~RS.is_false((tags, vals))

    # 2. spreadDamage (PS: sim/battle.ts:2108-2210) + drain (PS :2181-2186, sequential on the user)
    m = st.mons
    hp_t = mget(m.hp, targets).astype(I32)
    mx_t = mget(m.maxhp, targets).astype(I32)
    numeric = RS.is_num((tags, vals))
    hp0 = hp_t <= 0
    inactive = ~mget(m.is_active, targets)
    go = tv & numeric & ~hp0 & ~inactive
    amt = jnp.where(vals != 0, jnp.maximum(vals, 1), 0)                     # HOOK(H21) Damage event
    hit_t = go & (amt > 0)
    dealt = jnp.where(hit_t, jnp.minimum(amt, hp_t), 0)
    new_hp = hp_t - dealt
    last_j = NT - 1 - jnp.argmax(go[::-1])                                  # source.lastDamage: last target
    m = m.replace(hp=mset_many(m.hp, targets, new_hp, hit_t),
                  hurt_this_turn=mset_many(m.hurt_this_turn, targets, new_hp, go & (dealt != 0)),
                  last_damage=mset(m.last_damage, u, dealt[last_j], jnp.any(go)))
    st = F.faint_many(st.replace(mons=m), targets, hit_t & (hp_t - amt <= 0))
    tags, vals = RS.where(tv & numeric, RS.where(hp0, RS.R(RS.NUM, 0), RS.where(inactive, RS.R(RS.BOOL, 0),
                                                                                 RS.R(RS.NUM, dealt))), (tags, vals))
    drain_n = dex.move_drain[move, 0].astype(I32)
    drain_d = jnp.maximum(dex.move_drain[move, 1].astype(I32), 1)
    m = st.mons
    hp_u = mget(m.hp, u).astype(I32)
    mx_u = mget(m.maxhp, u).astype(I32)
    act_u = mget(m.is_active, u)
    healed = jnp.zeros((), bool)
    for j in range(NT):
        pct, col = EV.hp_public(new_hp[j], mx_t[j])
        st = EV.emit(st, go[j], EV.EV_DAMAGE, actor=targets[j], v1=pct, v2=col)  # PS: sim/battle.ts:2171
        a = (2 * dealt[j] * drain_n + drain_d) // (2 * drain_d)                    # Math.round
        ok = go[j] & (dealt[j] > 0) & (drain_n > 0) & (a != 0) & (hp_u > 0) & act_u & (hp_u < mx_u)
        hp_u = jnp.where(ok, jnp.minimum(mx_u, hp_u + a), hp_u)
        pu, cu = EV.hp_public(hp_u, mx_u)
        st = EV.emit(st, ok, EV.EV_HEAL, actor=u, target=targets[j], v1=pu, v2=cu, flags=EV.F_DRAIN)
        healed = healed | ok
    st = st.replace(mons=st.mons.replace(hp=mset(st.mons.hp, u, hp_u, healed)))
    tv = tv & ~RS.is_false((tags, vals))

    # 3. runMoveEffects (PS: sim/battle-actions.ts:1241-1372), per-target work batched
    tv_in = jnp.asarray(tvalid) & on
    da = (tags[0], vals[0])
    for j in range(1, NT):
        da = RS.where(tv_in[j], RS.combine(da, (tags[j], vals[j])), da)
    has_boosts = jnp.any(dex.move_boosts[move] != 0)
    heal_n = dex.move_heal[move, 0].astype(I32)
    heal_d = jnp.maximum(dex.move_heal[move, 1].astype(I32), 1)
    force_sw = dex.move_force_switch[move] == 1
    self_switch = dex.move_self_switch[move] > 0
    selfdestruct_ifhit = dex.move_selfdestruct[move] == 2
    m = st.mons
    not_fainted = ~mget(m.fainted, targets)
    on_b = tv & has_boosts & not_fainted
    st, br, bmsg = boost_v(st, targets, dex.move_boosts[move], on_b, order=dex.move_boost_order[move], primary=True)
    did = RS.where(on_b, RS.combine(RS.R(RS.UNDEF), br), RS.R(jnp.full(NT, RS.UNDEF, I32)))
    m = st.mons
    do_heal = tv & (heal_n > 0) & not_fainted
    hp2 = mget(m.hp, targets).astype(I32)
    mx2 = mget(m.maxhp, targets).astype(I32)
    full = hp2 >= mx2
    heal_amt = (2 * mx2 * heal_n + heal_d) // (2 * heal_d)
    heal_amt = jnp.where((heal_amt > 0) & (heal_amt <= 1), 1, heal_amt)
    hok = do_heal & ~full & (heal_amt != 0) & (hp2 > 0) & mget(m.is_active, targets) & (hp2 < mx2)
    new_h = jnp.minimum(mx2, hp2 + heal_amt)
    st = st.replace(mons=m.replace(hp=mset_many(m.hp, targets, new_h, hok)))
    hr = RS.where(heal_amt == 0, RS.R(RS.NUM, 0), RS.where(hok, RS.R(RS.NUM, new_h - hp2), RS.R(RS.BOOL, 0)))
    heal_fail = do_heal & (full | (~RS.truthy(hr) & ~((hr[0] == RS.NUM) & (hr[1] == 0))))
    fail_full = do_heal & full                                                    # PS :1257-1259
    tags, vals = RS.where(heal_fail, RS.combine((tags, vals), RS.R(RS.BOOL, 0)), (tags, vals))
    cont = tv & ~heal_fail
    did = RS.where(do_heal & ~heal_fail, RS.R(RS.BOOL, 1), did)
    # status / volatile / side / slot / weather / terrain / pseudoWeather: M3+
    cs_side = jnp.stack([F.can_switch(st, 0) > 0, F.can_switch(st, 1) > 0])
    cs = cs_side[side_of(jnp.clip(targets, 0, 7))]
    did = RS.where(cont & force_sw, RS.combine(did, RS.R(RS.BOOL, cs.astype(I32))), did)
    # HOOK(H22) onHit / Hit
    st = F.faint(st, u, jnp.any(cont & selfdestruct_ifhit & ~RS.is_false((tags, vals))))   # selfdestruct 'ifHit'
    css = cs_side[side_of(u)]                     # faint() only queues: canSwitch is unchanged
    did = RS.where(cont & self_switch, RS.where(css, RS.R(RS.BOOL, 1), RS.combine(did, RS.R(RS.BOOL, 0))), did)
    did = RS.where(did[0] == RS.UNDEF, RS.R(RS.BOOL, 1), did)
    tags, vals = RS.where(cont, RS.combine((tags, vals), RS.null_to_false(did)), (tags, vals))
    for j in range(NT):
        da = RS.where(heal_fail[j], RS.combine(da, RS.R(RS.NULL)), da)
        da = RS.where(cont[j], RS.combine(da, (did[0][j], did[1][j])), da)
        st = emit_boost(st, bmsg, targets[j], j)
        ph, chh = EV.hp_public(new_h[j], mx2[j])
        st = EV.emit(st, hok[j], EV.EV_HEAL, actor=targets[j], v1=ph, v2=chh)
        st = EV.emit(st, fail_full[j], EV.EV_FAIL, actor=targets[j], flags=EV.F_HEAL)
        st = EV.attr_last_move(st, fail_full[j], EV.F_STILL)
    m = st.mons
    has_self = jnp.any(dex.move_self_boosts[move] != 0) | (dex.move_self_volatile[move] > 0)
    # PS :1352-1358: a primary effect that did nothing (didAnything === false) -> '-fail' source [still]
    fail_msg = on & (da[0] == RS.BOOL) & (da[1] == 0) & ~has_self & (dex.move_self_chance[move] < 0) & \
        (dex.move_selfdestruct[move] == 0)
    st = EV.emit(st, fail_msg, EV.EV_FAIL, actor=u)
    st = EV.attr_last_move(st, fail_msg, EV.F_STILL)
    did_nothing = ~RS.truthy(da) & ~((da[0] == RS.NUM) & (da[1] == 0)) & ~has_self & (dex.move_selfdestruct[move] == 0)
    set_flag = on & ~did_nothing & self_switch & (mget(m.hp, u) > 0) & css
    st = st.replace(mons=m.replace(switch_flag=mset(m.switch_flag, u, move + 2, set_flag)))
    tv = tv & RS.hit_or_zero((tags, vals))

    # 4. selfDrops (self.boosts): SELF_EFFECT_ROLL only when self.chance is defined (contract §6)
    self_b = dex.move_self_boosts[move]
    has_self_boosts = jnp.any(self_b != 0)
    any_target = jnp.any(tv)
    self_chance = dex.move_self_chance[move].astype(I32)
    skey = C.dkey(st, ids.DK_SELF_EFFECT_ROLL, actor=u, subject=C.eff_move(move), hit=hit_no)
    sroll, rng = R.randint(rng, skey, 0, 100, active=any_target & has_self_boosts & (self_chance >= 0))
    apply_self = any_target & has_self_boosts & ((self_chance < 0) | (sroll < self_chance))
    st, _ = boost(st, u, self_b, apply_self)

    # 5. secondaries, PS order per target then per secondary; SECONDARY draws only for numeric chance. Target boosts
    #    touch only that target (batched per secondary); the user's self-boosts run as a scalar recurrence.
    nsec = dex.move_nsec[move]
    oks = []
    for k in range(2):
        live = tv & (k < nsec)
        chance = dex.move_sec_chance[move, k].astype(I32)
        key = C.dkey(st, ids.DK_SECONDARY, actor=u, target=targets, subject=C.eff_move(move), hit=hit_no, ord=k)
        roll, rng = R.randint_v(rng, key, 0, 100, active=live & (chance >= 0))
        oks.append(live & ((chance < 0) | (roll < chance)))
    tmsgs = []
    for k in range(2):
        sb = dex.move_sec_boosts[move, k]
        st, _, msg = boost_v(st, targets, sb, oks[k] & jnp.any(sb != 0) & ~mget(st.mons.fainted, targets))
        tmsgs.append(msg)
    m = st.mons
    cur_u = mget(m.boosts, u).astype(I32)
    base_u = (mget(m.hp, u) > 0) & mget(m.is_active, u) & (st.sides.pokemon_left[1 - side_of(u)] > 0)
    up = down = applied = jnp.zeros((), bool)
    smsgs = {}
    for j in range(NT):
        for k in range(2):
            ssb = dex.move_sec_self_boosts[move, k].astype(I32)
            g = oks[k][j] & jnp.any(ssb != 0) & base_u
            cur_u, capped, smsgs[j, k] = boost_math(cur_u, ssb, g)
            succ = g & jnp.any(capped != 0)
            up, down = up | (succ & jnp.any(capped > 0)), down | (succ & jnp.any(capped < 0))
            applied = applied | g
    m = m.replace(boosts=mset(m.boosts, u, cur_u, applied),
                  stats_raised_this_turn=mset(m.stats_raised_this_turn, u, True, up),
                  stats_lowered_this_turn=mset(m.stats_lowered_this_turn, u, True, down))
    st = st.replace(mons=m)
    for j in range(NT):
        for k in range(2):
            st = emit_boost(st, tmsgs[k], targets[j], j)
            st = emit_boost(st, smsgs[j, k], u)
        # secondary status / volatile: M3

    # 6. forceSwitch -> forceSwitchFlag (HOOK DragOut)
    m = st.mons
    go6 = tv & force_sw & (mget(m.hp, targets) > 0) & (mget(m.hp, u) > 0) & cs
    st = st.replace(mons=m.replace(force_switch_flag=mset_many(m.force_switch_flag, targets, True, go6)))
    # DamagingHit / AfterHit / EmergencyExit: M3
    return st, rng, (tags, vals)


# ------------------------------------------------------------------ resumable move execution (ADR-104 option 4)
# runMove -> useMove -> trySpreadMoveHit -> Champions hit loop, split at the hits: move_begin runs every prelude and
# stores what the PS call stack keeps (st.exec); move_hit runs one hit per micro-step; move_end runs every tail in
# PS order (hit loop, trySpreadMoveHit, useMove, runMove).
def _exec_where(on, new: S.Exec, old: S.Exec) -> S.Exec:
    return jax.tree_util.tree_map(lambda a, b: jnp.where(on, a, b), new, old)


def move_begin(st, dex, rng, cfg, side, b, move_slot, move_id, target_loc, orig, on):
    """PS runMove (sim/battle-actions.ts:267-418) up to useMove, useMoveInner (:429-582) up to trySpreadMoveHit,
    trySpreadMoveHit (:612-682) steps 0-4 and the hit-loop setup (ch/scripts.ts:426-461). Returns (st, rng)."""
    on = jnp.asarray(on)
    u = jnp.asarray(side, I32) * 4 + jnp.asarray(b, I32)
    move = jnp.asarray(move_id, I32)
    # ---- runMove
    m = st.mons
    st = st.replace(mons=m.replace(active_move_actions=mset(m.active_move_actions, u, mget(m.active_move_actions, u) + 1, on)))
    target, rng = get_target(st, dex, rng, u, move, target_loc, orig, on)
    # HOOK(H2) OverrideAction / LockMove; HOOK(H1) BeforeMove gates; beforeMoveCallback
    m = st.mons
    m = m.replace(last_damage=mset(m.last_damage, u, 0, on))
    slot = jnp.asarray(move_slot, I32)
    is_struggle = move == T.MOVE_STRUGGLE
    has_slot = (slot >= 0) & ~is_struggle
    pp_now = mget(m.pp, u)[jnp.clip(slot, 0, 3)].astype(I32)
    deduct = on & has_slot
    pp_new = mget(m.pp, u).at[jnp.clip(slot, 0, 3)].set(jnp.maximum(pp_now - 1, 0).astype(I8))
    used_new = mget(m.move_used, u).at[jnp.clip(slot, 0, 3)].set(True)
    m = m.replace(pp=mset(m.pp, u, pp_new, deduct & (pp_now > 0)), move_used=mset(m.move_used, u, used_new, deduct))
    nopp = on & ~is_struggle & ~(has_slot & (pp_now > 0))
    go_rm = on & ~nopp
    m = m.replace(move_this_turn_result=mset(m.move_this_turn_result, u, S.MR_FALSE, nopp),
                  last_move=mset(m.last_move, u, move, go_rm),
                  last_move_target_loc=mset(m.last_move_target_loc, u, target_loc, go_rm),
                  move_this_turn=mset(m.move_this_turn, u, move, go_rm))
    # ---- useMove / useMoveInner
    m = m.replace(move_this_turn_result=mset(m.move_this_turn_result, u, S.MR_UNDEF, go_rm),
                  last_move_used=mset(m.last_move_used, u, move, go_rm))
    st = st.replace(mons=m)
    tt = dex.move_target[move]
    target = jnp.where(_in(tt, (T.TARGET_SELF, T.TARGET_ALLIES)), u, target)
    # HOOK(H3) ModifyType / ModifyMove
    alive = ~mget(m.fainted, u)
    # ModifyMove: Struggle adds '-activate' pokemon 'move: Struggle' (data/moves.ts:18220-18223)
    st = EV.emit(st, go_rm & is_struggle, EV.EV_ACTIVATE, actor=u, move=move)
    # PS :519-528: addMove('move', pokemon, move, target); no target -> [notarget] + '-fail'
    st = EV.emit_move(st, go_rm & alive, u, target, move)
    no_target = go_rm & alive & (target < 0)
    st = EV.attr_last_move(st, no_target, EV.F_NOTARGET)
    st = EV.emit(st, no_target, EV.EV_FAIL, actor=u)
    go_um = go_rm & alive & (target >= 0)
    targets, smart, rng = get_move_targets(st, dex, rng, u, move, target, go_um)
    # HOOK(H5) Pressure / DeductPP; HOOK(H6) TryMove
    side_field = _in(tt, (T.TARGET_ALL, T.TARGET_FOESIDE, T.TARGET_ALLYSIDE, T.TARGET_ALLYTEAM))
    has_t = targets[0] >= 0
    st = F.faint(st, u, go_um & (dex.move_selfdestruct[move] == 1))
    empty = go_um & ~side_field & ~has_t                                # PS :573-577
    st = EV.attr_last_move(st, empty, EV.F_NOTARGET)
    st = EV.emit(st, empty, EV.EV_FAIL, actor=u)
    on_ts = go_um & has_t & ~side_field
    # ---- trySpreadMoveHit steps 0-4 (side/field moves: tryMoveHit, M3)
    m = st.mons
    tv = (targets >= 0) & on_ts
    n_targets = jnp.sum(tv.astype(I32))
    spread = (n_targets > 1) & ~smart
    # 0 Invulnerability / 1 TryHit: hooks (M3)
    # 2 type immunity
    ign = dex.move_ignore_immunity[move]
    ignore = (ign == 1) | ((ign == 0) & (dex.move_category[move] == T.CAT_STATUS))
    mtype = jnp.where(is_struggle, T.TYPE_TYPELESS, dex.move_type[move])
    t_types = mget(m.types, targets)                                              # [3, 2]
    flying = jnp.any(t_types == T.TYPE_FLYING, axis=-1)
    grounded = ~flying & (mget(m.ability, targets) != T.ABILITY_LEVITATE)
    typed = jax.vmap(lambda tt_: TC.immune(dex, mtype, tt_))(t_types)
    imm = ~ignore & jnp.where(mtype == T.TYPE_GROUND, ~grounded, typed) & (mtype != T.TYPE_TYPELESS)
    # runImmunity(move, !smartTarget) messages (PS: sim/pokemon.ts:2276-2286): isGrounded null -> Levitate
    lev = (mtype == T.TYPE_GROUND) & ~flying
    for j in range(NT):
        st = EV.emit(st, tv[j] & imm[j] & ~smart, EV.EV_IMMUNE, actor=targets[j],
                     flags=jnp.where(lev[j], EV.F_LEVITATE, 0))
    fail_any = jnp.any(tv & imm)
    tv = tv & ~imm
    # 3 TryImmunity: powder vs Grass (getImmunity('powder')), move onTryImmunity (M3), Prankster vs Dark (M3)
    powder = mflag(dex, move, T.MF_POWDER)
    pimm = powder & (targets != u) & jax.vmap(
        lambda tt_: TC.status_immune(dex, ids.IMMUNITY_KEYS.index("powder"), tt_))(t_types)
    for j in range(NT):
        st = EV.emit(st, tv[j] & pimm[j], EV.EV_IMMUNE, actor=targets[j])            # PS :731
    fail_any = fail_any | jnp.any(tv & pimm)
    tv = tv & ~pimm
    # 4 accuracy
    spread_hit = ((n_targets > 1) & ~smart) | (dex.move_spread_hit[move] == 1)        # PS :613
    st, acc_ok, rng = accuracy_step(st, dex, rng, u, move, targets, tv, on_ts & jnp.any(tv), smart=smart,
                                    spread=spread_hit)
    acc_ok = jnp.where(tv, acc_ok, False)
    fail_any = fail_any | jnp.any(tv & ~acc_ok)
    smart = smart & ~jnp.any(tv & ~acc_ok)
    tv = tv & acc_ok
    on_hl = on_ts & jnp.any(tv)
    # ---- hit-loop setup (ch/scripts.ts:432-449)
    mh = dex.move_multihit[move]
    is_range = (mh[0] == 2) & (mh[1] == 5)
    mkey = C.dkey(st, ids.DK_MULTIHIT_COUNT, actor=u, subject=C.eff_move(move))
    idx, rng = R.sample(rng, mkey, 20, active=on_hl & is_range)
    hits = jnp.where(is_range, MULTIHIT_TABLE[idx], jnp.where(mh[0] > 0, mh[0], 1)).astype(I32)
    full = lambda v: jnp.full(NT, v, I32)  # noqa: E731
    new = S.Exec(in_move=jnp.asarray(True), u=u, move=move, targets=targets.astype(I32), tv=tv, spread=spread,
                 smart=smart, spread_hit=spread_hit, on_rm=jnp.asarray(True), go_rm=go_rm, go_um=go_um, has_t=has_t,
                 side_field=side_field, on_ts=on_ts, on_hl=on_hl, fail_any=fail_any, hits=hits,
                 multiacc=dex.move_multiaccuracy[move] == 1, single=jnp.sum(tv.astype(I32)) == 1, h=jnp.int32(1),
                 hit_done=jnp.int32(1), stopped=~on_hl, null_damage=jnp.asarray(True), dt=full(RS.NUM), dv=full(0),
                 lt=full(RS.UNDEF), lv=full(0), total=jnp.int32(0))
    return st.replace(exec=_exec_where(on, new, st.exec)), rng


def move_hit(st, dex, rng, cfg):
    """One iteration of the Champions hit loop (ch/scripts.ts:462-527) for the move in st.exec. Returns (st, rng)."""
    ex = st.exec
    live = ex.in_move & ~ex.stopped
    u, move, targets, tv, smart, h = ex.u, ex.move, ex.targets, ex.tv, ex.smart, ex.h
    m = st.mons
    go = live & (h <= ex.hits)
    any_false = jnp.any(RS.is_false((ex.dt, ex.dv)))
    alive = (targets >= 0) & (mget(m.hp, jnp.clip(targets, 0, 7)) > 0) & tv
    go = go & ~any_false & jnp.any(alive)
    stopped = ex.stopped | (live & ~go)
    js = jnp.arange(NT)
    cur_tv = jnp.where(smart, (js == h - 1) & tv, tv)            # smartTarget: hit h uses targets[h-1]
    # multiaccuracy re-roll for hits >= 2: float stage table -> exact rational bound
    t0 = jnp.where(smart, targets[jnp.clip(h - 1, 0, NT - 1)], targets[0])
    a = dex.move_accuracy[move].astype(I32)
    bu = jnp.clip(mget(m.boosts, u)[5].astype(I32), -6, 6)
    be = jnp.clip(mget(m.boosts, t0)[6].astype(I32), -6, 6)
    be = jnp.where(dex.move_ignore_evasion[move] == 1, 0, be)
    num = a * jnp.where(bu > 0, 3 + bu, 3) * jnp.where(be < 0, 3 - be, 3)
    den = jnp.where(bu > 0, 3, 3 - bu) * jnp.where(be > 0, 3 + be, 3)
    akey = C.dkey(st, ids.DK_ACCURACY, actor=u, target=t0, subject=C.eff_move(move), hit=h)
    bound = (num + den - 1) // den   # r < num/den  <=>  r < ceil(num/den) for integer r
    need_acc = go & ex.multiacc & (a > 0) & (h > 1)
    ahit, rng = R.chance(rng, akey, bound, 100, active=need_acc)
    miss = need_acc & ~ahit
    stopped = stopped | miss
    go = go & ~miss
    # [C] scripts.ts:476-482: boolean smartTarget moves retarget the move line on hit 1 (-anim for later hits)
    st = EV.retarget_last_move(st, go & (h == 1) & (dex.move_smart_target[move] == 1) & (t0 >= 0), t0)
    st, rng, (mt, mv) = spread_move_hit(st, dex, rng, cfg, u, move, targets, cur_tv, ex.spread, h, go)
    all_false = ~jnp.any(~RS.is_false((mt, mv)) & cur_tv)
    stopped = stopped | (go & all_false)
    go = go & ~all_false
    cnt = go & cur_tv
    v = jnp.where(RS.truthy((mt, mv)) & RS.is_num((mt, mv)), mv, 0)
    user_dead = mget(st.mons.hp, u) <= 0
    stopped = stopped | (go & user_dead & ex.single) | (live & (h >= ex.hits))
    new = ex.replace(dt=jnp.where(cnt, RS.NUM, ex.dt), dv=jnp.where(cnt, v, ex.dv),
                     total=ex.total + jnp.sum(jnp.where(cnt, v, 0)), null_damage=ex.null_damage & ~go,
                     lt=jnp.where(cnt, mt, ex.lt), lv=jnp.where(cnt, mv, ex.lv), hit_done=ex.hit_done + go.astype(I32),
                     stopped=stopped, h=h + 1)
    return st.replace(exec=_exec_where(live, new, ex)), rng


def move_end(st, dex, rng, cfg):
    """The tails, in PS order, of the move in st.exec once its hits are done: hit loop (ch/scripts.ts:528-588),
    trySpreadMoveHit (:676-682), useMoveInner (:583-610), runMove (:381-418). Returns (st, rng)."""
    ex = st.exec
    end = ex.in_move & ex.stopped
    u, move, targets, tv = ex.u, ex.move, ex.targets, ex.tv
    # ---- hit loop tail
    on = end & ex.on_hl
    hit_count = ex.hit_done - 1
    all_fail = on & ((ex.hit_done == 1) | ex.null_damage)
    dt = jnp.where(all_fail, RS.BOOL, ex.dt)
    dv = jnp.where(all_fail, 0, ex.dv)
    run_rest = on & ~(ex.hit_done == 1)
    st = F.resolve_faints(st, dex, run_rest, do_check_win=mget(st.mons.hp, u) <= 0)
    # [C] scripts.ts:545-549: '-hitcount' for multihit moves without a boolean smartTarget
    mh = dex.move_multihit[move]
    st = EV.emit(st, run_rest & (mh[0] > 0) & (dex.move_smart_target[move] != 1), EV.EV_HITCOUNT, actor=targets[0],
                 v1=hit_count)
    st = apply_recoil(st, dex, u, move, ex.total, run_rest & (ex.total > 0))
    # gotAttacked / timesAttacked
    m = st.mons
    num = run_rest & tv & (targets != u) & (ex.lt == RS.NUM)
    ta = mset_many(m.times_attacked, targets, mget(m.times_attacked, targets) + jnp.where(ex.smart, 1, hit_count), num)
    st = st.replace(mons=m.replace(times_attacked=ta))
    # [C] scripts.ts:569
    st = EV.emit(st, run_rest & (dex.move_ohko[move] > 0) & (mget(st.mons.hp, targets[0]) <= 0), EV.EV_OHKO)
    # ---- trySpreadMoveHit tail
    on_ts = end & ex.on_ts
    kept = RS.hit_or_zero((dt, dv)) & tv
    fail_any = ex.fail_any | jnp.any(tv & RS.is_false((dt, dv)))
    res = jnp.any(kept)
    set_null = on_ts & ~res & ~fail_any
    # PS :679-680: attrLastMove('[spread] ' + hit slots)
    slots = jnp.int32(0)
    for j in range(NT):
        t = targets[j]
        slots = slots | jnp.where(kept[j], 1 << jnp.clip(side_of(t) * 2 + position(st, t), 0, 3), 0)
    st = EV.attr_last_move(st, on_ts & ex.spread_hit, EV.F_SPREAD, v1=slots)
    # ---- useMoveInner tail (side/field moves: tryMoveHit, M3)
    go = end & ex.go_um
    result = go & ((ex.has_t & ~ex.side_field & res) | ex.side_field)
    # selfBoost (Scale Shot) on success
    st, _ = boost(st, u, dex.move_selfboost[move], result & jnp.any(dex.move_selfboost[move] != 0))
    st = F.faint(st, u, go & (mget(st.mons.hp, u) <= 0))
    # MoveFail / AfterMoveSecondarySelf / EmergencyExit: M3
    set_null = set_null & go & ex.has_t
    # ---- runMove tail
    go_rm = end & ex.go_rm
    m = st.mons
    mr = jnp.where(set_null, S.MR_NULL, jnp.where(result, S.MR_TRUE, S.MR_FALSE))
    st = st.replace(mons=m.replace(move_this_turn_result=mset(m.move_this_turn_result, u, mr, go_rm)),
                    meta=st.meta.replace(last_successful_move=jnp.where(
                        go_rm, jnp.where(result, move, 0), st.meta.last_successful_move).astype(I16)))
    # HOOK AfterMove; then resolveFaints() + checkWin()
    on_rm = end & ex.on_rm
    st = F.resolve_faints(st, dex, on_rm)
    st = F.check_win(st, -1, on_rm)
    return st.replace(exec=ex.replace(in_move=ex.in_move & ~end)), rng


def apply_recoil(st, dex, u, move, total, on):
    """PS: sim/battle-actions.ts:1435-1453 applyRecoilDamage."""
    m = st.mons
    struggle = dex.move_struggle_recoil[move] == 1
    rn = dex.move_recoil[move, 0].astype(I32)
    rd = jnp.maximum(dex.move_recoil[move, 1].astype(I32), 1)
    maxhp = mget(m.maxhp, u).astype(I32)
    rec = jnp.where(struggle, jnp.maximum((2 * maxhp + 4) // 8, 1), jnp.maximum((2 * total * rn + rd) // (2 * rd), 1))
    has = struggle | (rn > 0)
    on = jnp.asarray(on) & has & (mget(m.hp, u) > 0)
    # struggle: directDamage (no Damage event); recoil: battle.damage(..., 'recoil') -> spreadDamage (HOOK H21)
    on = on & mget(m.is_active, u)
    st, dealt = F.damage(st, u, rec, on)
    st = EV.emit_hp(st, on, EV.EV_DAMAGE, u, flags=EV.F_RECOIL)        # '[from] Recoil' / '[from] recoil'
    m = st.mons
    return st.replace(mons=m.replace(hurt_this_turn=mset(m.hurt_this_turn, u, mget(m.hp, u), on & ~struggle & (dealt != 0))))
