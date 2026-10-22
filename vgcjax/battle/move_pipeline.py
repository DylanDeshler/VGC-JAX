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
from vgcjax.battle.util import mget, mset, side_of
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
def boost(st, target, deltas7, on, order=None, primary=False):
    """battle.boost with no hooks (ChangeBoost / TryBoost / AfterEachBoost join in M3). Returns (st, result).
    Messages (PS :2053-2095): per stat in the boosts dict's key order (`order`, default canonical), -boost/-unboost
    with the applied amount; a zero change is shown only for primary effects (not isSecondary / isSelf)."""
    m = st.mons
    on = jnp.asarray(on) & (jnp.asarray(target) >= 0)
    hp = mget(m.hp, target)
    active = mget(m.is_active, target)
    foe_left = st.sides.pokemon_left[1 - side_of(target)] > 0
    cur = mget(m.boosts, target).astype(I32)
    d = jnp.asarray(deltas7, I32)
    capped = jnp.clip(cur + d, -6, 6) - cur
    capped = jnp.where(d != 0, capped, 0)
    go = on & (hp > 0) & active & foe_left
    success = jnp.any(capped != 0)
    order = jnp.arange(7) if order is None else jnp.asarray(order, I32)
    for k in range(7):
        j = order[k]
        dj, cj = d[j], capped[j]
        unboost = (dj < 0) | (cur[j] + cj == -6)
        st = EV.emit(st, go & (dj != 0) & ((cj != 0) | primary), EV.EV_BOOST, actor=target, v1=j, v2=jnp.abs(cj),
                     flags=jnp.where(unboost, EV.F_UNBOOST, 0))
    m = m.replace(boosts=mset(m.boosts, target, cur + capped, go),
                  stats_raised_this_turn=mset(m.stats_raised_this_turn, target, True, go & success & jnp.any(capped > 0)),
                  stats_lowered_this_turn=mset(m.stats_lowered_this_turn, target, True, go & success & jnp.any(capped < 0)))
    res = jnp.where(hp <= 0, 0, 0)
    r = RS.where(hp <= 0, RS.R(RS.NUM, 0), RS.where(~active | ~foe_left, RS.R(RS.BOOL, 0),
                                                    RS.where(success, RS.R(RS.BOOL, 1), RS.R(RS.NULL))))
    del res
    return st.replace(mons=m), r


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
    """hitStepAccuracy for each target (fori over target slots). Returns (st, ok[3], rng).
    Messages: '-immune' target '[ohko]' (PS :768); a miss adds '-miss' user target and marks the move line [miss] when
    not spreadHit (PS :800-806); smartTarget misses are silent."""
    m = st.mons
    acc_base = dex.move_accuracy[move].astype(I32)        # 0 = true
    ohko = dex.move_ohko[move]
    cat = dex.move_category[move]
    tt = dex.move_target[move]
    u_types = mget(m.types, u)
    targets = jnp.asarray(targets, I32)

    def body(j, carry):
        st, oks, rng = carry
        t = targets[j]
        live = jnp.asarray(on) & (t >= 0) & tv[j]
        t_types = mget(m.types, t)
        # OHKO: 30 (Sheer Cold 20 for non-Ice users), Sheer Cold fails on Ice targets
        ohko_acc = jnp.where((ohko == 2) & ~jnp.any(u_types == T.TYPE_ICE), 20, 30)
        ohko_immune = (ohko == 2) & jnp.any(t_types == T.TYPE_ICE)
        # stages (HOOK H11 ModifyAccuracy / ModifyBoost = identity)
        b = jnp.clip(mget(m.boosts, u)[5].astype(I32), -6, 6)
        b = jnp.where(dex.move_ignore_evasion[move] == 1, b, jnp.clip(b - mget(m.boosts, t)[6].astype(I32), -6, 6))
        staged = jnp.where(b > 0, acc_base * (3 + b) // 3, jnp.where(b < 0, acc_base * 3 // (3 - b), acc_base))
        acc = jnp.where(ohko > 0, ohko_acc, staged)
        always = ((acc_base == 0) & (ohko == 0)) | ((tt == T.TARGET_SELF) & (cat == T.CAT_STATUS))
        key = C.dkey(st, ids.DK_ACCURACY, actor=u, target=t, subject=C.eff_move(move), hit=0)
        hit, rng = R.chance(rng, key, acc, 100, active=live & ~always & ~((ohko > 0) & ohko_immune))
        ok = jnp.where(always, True, hit & ~((ohko > 0) & ohko_immune))
        oimm = live & ~always & (ohko > 0) & ohko_immune
        st = EV.emit(st, oimm, EV.EV_IMMUNE, actor=t, flags=EV.F_OHKO)
        miss = live & ~ok & ~oimm & ~smart
        st = EV.attr_last_move(st, miss & ~spread, EV.F_MISS)
        st = EV.emit(st, miss, EV.EV_MISS, actor=u, target=t)
        return st, oks.at[j].set(ok), rng

    return jax.lax.fori_loop(0, NT, body, (st, jnp.zeros(NT, bool), rng))


# ------------------------------------------------------------------ damage for one target
def target_damage(st, dex, rng, cfg, u, move, t, spread, on):
    """getDamage(source, target, move) with CRIT and DAMAGE_ROLL draws. Returns (rs-result, rng)."""
    m = st.mons
    hit = jnp.int32(1)  # set by caller via key fields; see spread_move_hit
    del hit
    return None, rng


def get_damage(st, dex, rng, cfg, u, move, t, spread, hit_no, on):
    """PS getDamage for target t: returns (st, (tag, val), rng). Struggle is typeless via its onModifyMove."""
    m = st.mons
    ab_t = mget(m.ability, t)
    crit_immune = _in(ab_t, (T.ABILITY_BATTLEARMOR, T.ABILITY_SHELLARMOR))   # HOOK(H13) CriticalHit
    crit_ratio = jnp.clip(dex.move_crit_ratio[move].astype(I32), 0, 4)
    will_crit = dex.move_will_crit[move] == 1
    ckey = C.dkey(st, ids.DK_CRIT, actor=u, target=t, subject=C.eff_move(move), hit=hit_no)
    bp = dex.move_bp[move].astype(I32)
    fixed = dex.move_fixed_damage[move]
    ohko = dex.move_ohko[move] > 0
    rolls_damage = jnp.asarray(on) & (bp > 0) & ~ohko & (fixed == 0)
    crit_draw, rng = R.chance(rng, ckey, 1, D.crit_denominator(crit_ratio),
                              active=rolls_damage & ~will_crit & (crit_ratio > 0))
    rkey = C.dkey(st, ids.DK_DAMAGE_ROLL, actor=u, target=t, subject=C.eff_move(move), hit=hit_no)
    roll, rng = R.randint(rng, rkey, 0, 16, active=rolls_damage)
    grounded = ~jnp.any(mget(m.types, t) == T.TYPE_FLYING) & (ab_t != T.ABILITY_LEVITATE)
    is_struggle = move == T.MOVE_STRUGGLE
    # Struggle: onModifyMove type '???' (data/moves.ts struggle) -> treat as typeless via a typeless attacker types
    dmg, kind, crit = D.get_damage_core(
        dex, move, atk_stats5=mget(m.stats, u), atk_boosts7=mget(m.boosts, u),
        atk_types=jnp.where(is_struggle, jnp.array([-1, -1], I8), mget(m.types, u)),
        atk_burned=mget(m.status, u) == T.COND_BRN, def_stats5=mget(m.stats, t), def_boosts7=mget(m.boosts, t),
        def_types=jnp.where(is_struggle, jnp.array([-1, -1], I8), mget(m.types, t)), def_maxhp=mget(m.maxhp, t),
        def_grounded=grounded, def_crit_immune=crit_immune, crit_draw=crit_draw, roll=roll, spread=spread,
        canary_crit=cfg.flags.canary_crit_2048)
    # [C] modifyDamage messages (data/mods/champions/scripts.ts:266-284): effectiveness then crit
    formula = rolls_damage & (kind == D.DMG_NUMBER)
    tm = jnp.clip(jnp.where(is_struggle, 0, TC.effectiveness(dex, dex.move_type[move], mget(m.types, t))), -6, 6)
    st = EV.emit(st, formula & (tm > 0), EV.EV_SUPER, actor=t, v1=jnp.minimum(tm, 2))
    st = EV.emit(st, formula & (tm < 0), EV.EV_RESIST, actor=t, v1=jnp.minimum(-tm, 2))
    st = EV.emit(st, formula & crit, EV.EV_CRIT, actor=t)
    r = jnp.where(kind == D.DMG_NUMBER, 1, 0)
    res = RS.where(kind == D.DMG_NUMBER, RS.R(RS.NUM, dmg), RS.where(kind == D.DMG_FALSE, RS.R(RS.BOOL, 0),
                                                                     RS.where(kind == D.DMG_UNDEFINED, RS.R(RS.UNDEF),
                                                                              RS.R(RS.NUM, 0))))
    del r
    return st, res, rng


# ------------------------------------------------------------------ spreadMoveHit (ch/scripts.ts:314-423), core
def _rget(tags, vals, j):
    return (tags[j], vals[j])


def _rset(tags, vals, j, r, cond=True):
    return tags.at[j].set(jnp.where(cond, r[0], tags[j])), vals.at[j].set(jnp.where(cond, r[1], vals[j]))


def spread_move_hit(st, dex, rng, cfg, u, move, targets, tvalid, spread, hit_no, on):
    """One hit on the current targets (ch/scripts.ts:314-423, core). Returns (st, rng, (tags[3], vals[3]))."""
    on = jnp.asarray(on)
    targets = jnp.asarray(targets, I32)
    tags = jnp.full(NT, RS.BOOL, I32)
    vals = jnp.ones(NT, I32)
    tv = jnp.asarray(tvalid) & on
    # TryHit (move onTryHit; none in core) / TryPrimaryHit (Substitute; HOOK H12) -> true

    # 1. getSpreadDamage
    def s1(j, c):
        st, rng, tags, vals, tv = c
        st, res, rng = get_damage(st, dex, rng, cfg, u, move, targets[j], spread, hit_no, tv[j])
        tags, vals = _rset(tags, vals, j, res, tv[j])
        tv = tv.at[j].set(tv[j] & ~RS.is_false(_rget(tags, vals, j)))
        return st, rng, tags, vals, tv
    st, rng, tags, vals, tv = jax.lax.fori_loop(0, NT, s1, (st, rng, tags, vals, tv))

    # 2. spreadDamage (PS: sim/battle.ts:2108-2210) + drain
    drain_n = dex.move_drain[move, 0].astype(I32)
    drain_d = jnp.maximum(dex.move_drain[move, 1].astype(I32), 1)

    def s2(j, c):
        st, tags, vals, tv = c
        t = targets[j]
        m = st.mons
        live = tv[j]
        cur = _rget(tags, vals, j)
        numeric = RS.is_num(cur)
        hp0 = mget(m.hp, t) <= 0
        inactive = ~mget(m.is_active, t)
        go = live & numeric & ~hp0 & ~inactive
        amt = jnp.where(cur[1] != 0, jnp.maximum(cur[1], 1), 0)            # HOOK(H21) Damage event
        st, dealt = F.damage(st, t, amt, go)
        st = EV.emit_hp(st, go, EV.EV_DAMAGE, t)                             # PS: sim/battle.ts:2171
        m = st.mons
        st = st.replace(mons=m.replace(hurt_this_turn=mset(m.hurt_this_turn, t, mget(m.hp, t), go & (dealt != 0)),
                                       last_damage=mset(m.last_damage, u, dealt, go)))
        newr = RS.where(hp0, RS.R(RS.NUM, 0), RS.where(inactive, RS.R(RS.BOOL, 0), RS.R(RS.NUM, dealt)))
        tags, vals = _rset(tags, vals, j, newr, live & numeric)
        drain_amt = (2 * dealt * drain_n + drain_d) // (2 * drain_d)          # Math.round
        st, _ = battle_heal(st, u, drain_amt, go & (dealt > 0) & (drain_n > 0), drain_of=t)
        tv = tv.at[j].set(tv[j] & ~RS.is_false(_rget(tags, vals, j)))
        return st, tags, vals, tv
    st, tags, vals, tv = jax.lax.fori_loop(0, NT, s2, (st, tags, vals, tv))

    # 3. runMoveEffects (PS: sim/battle-actions.ts:1241-1372)
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

    def s3(j, c):
        st, tags, vals, da_t, da_v = c
        da = (da_t, da_v)
        t = targets[j]
        live = tv[j]
        m = st.mons
        did = RS.R(RS.UNDEF)
        not_fainted = ~mget(m.fainted, t)
        st, br = boost(st, t, dex.move_boosts[move], live & has_boosts & not_fainted, order=dex.move_boost_order[move],
                       primary=True)
        did = RS.where(live & has_boosts & not_fainted, RS.combine(did, br), did)
        # heal
        m = st.mons
        do_heal = live & (heal_n > 0) & not_fainted
        full = mget(m.hp, t) >= mget(m.maxhp, t)
        heal_amt = (2 * mget(m.maxhp, t).astype(I32) * heal_n + heal_d) // (2 * heal_d)
        st, hr = battle_heal(st, t, heal_amt, do_heal & ~full)
        heal_fail = do_heal & (full | (~RS.truthy(hr) & ~((hr[0] == RS.NUM) & (hr[1] == 0))))
        st = EV.emit(st, do_heal & full, EV.EV_FAIL, actor=t, flags=EV.F_HEAL)   # PS :1257-1259
        st = EV.attr_last_move(st, do_heal & full, EV.F_STILL)
        cur = _rget(tags, vals, j)
        tags, vals = _rset(tags, vals, j, RS.combine(cur, RS.R(RS.BOOL, 0)), heal_fail)
        da = RS.where(heal_fail, RS.combine(da, RS.R(RS.NULL)), da)
        cont = live & ~heal_fail
        did = RS.where(do_heal & ~heal_fail, RS.R(RS.BOOL, 1), did)
        # status / volatile / side / slot / weather / terrain / pseudoWeather: M3+
        cs = F.can_switch(st, side_of(t)) > 0
        did = RS.where(cont & force_sw, RS.combine(did, RS.R(RS.BOOL, cs.astype(I32))), did)
        # HOOK(H22) onHit / Hit
        cur = _rget(tags, vals, j)
        st = F.faint(st, u, cont & selfdestruct_ifhit & ~RS.is_false(cur))   # selfdestruct 'ifHit' (Memento)
        css = F.can_switch(st, side_of(u)) > 0
        did = RS.where(cont & self_switch, RS.where(css, RS.R(RS.BOOL, 1), RS.combine(did, RS.R(RS.BOOL, 0))), did)
        did = RS.where(did[0] == RS.UNDEF, RS.R(RS.BOOL, 1), did)
        tags, vals = _rset(tags, vals, j, RS.combine(cur, RS.null_to_false(did)), cont)
        da = RS.where(cont, RS.combine(da, did), da)
        return st, tags, vals, da[0], da[1]
    st, tags, vals, da_t, da_v = jax.lax.fori_loop(0, NT, s3, (st, tags, vals, da[0], da[1]))
    da = (da_t, da_v)
    m = st.mons
    has_self = jnp.any(dex.move_self_boosts[move] != 0) | (dex.move_self_volatile[move] > 0)
    # PS :1352-1358: a primary effect that did nothing (didAnything === false) -> '-fail' source [still]
    fail_msg = on & (da[0] == RS.BOOL) & (da[1] == 0) & ~has_self & (dex.move_self_chance[move] < 0) & \
        (dex.move_selfdestruct[move] == 0)
    st = EV.emit(st, fail_msg, EV.EV_FAIL, actor=u)
    st = EV.attr_last_move(st, fail_msg, EV.F_STILL)
    m = st.mons
    did_nothing = ~RS.truthy(da) & ~((da[0] == RS.NUM) & (da[1] == 0)) & ~has_self & (dex.move_selfdestruct[move] == 0)
    set_flag = on & ~did_nothing & self_switch & (mget(m.hp, u) > 0) & (F.can_switch(st, side_of(u)) > 0)
    st = st.replace(mons=m.replace(switch_flag=mset(m.switch_flag, u, move + 2, set_flag)))
    tv = tv & jnp.stack([RS.hit_or_zero((tags[j], vals[j])) for j in range(NT)])

    # 4. selfDrops (self.boosts): SELF_EFFECT_ROLL only when self.chance is defined (contract §6)
    self_b = dex.move_self_boosts[move]
    has_self_boosts = jnp.any(self_b != 0)
    any_target = jnp.any(tv)
    self_chance = dex.move_self_chance[move].astype(I32)
    skey = C.dkey(st, ids.DK_SELF_EFFECT_ROLL, actor=u, subject=C.eff_move(move), hit=hit_no)
    sroll, rng = R.randint(rng, skey, 0, 100, active=any_target & has_self_boosts & (self_chance >= 0))
    apply_self = any_target & has_self_boosts & ((self_chance < 0) | (sroll < self_chance))
    st, _ = boost(st, u, self_b, apply_self)

    # 5. secondaries (per target, per secondary): SECONDARY draws only for numeric chance
    nsec = dex.move_nsec[move]

    def s5(i, c):
        st, rng = c
        j = i // 2
        k = i % 2
        t = targets[j]
        live = tv[j] & (k < nsec)
        chance = dex.move_sec_chance[move, k].astype(I32)
        key = C.dkey(st, ids.DK_SECONDARY, actor=u, target=t, subject=C.eff_move(move), hit=hit_no, ord=k)
        roll, rng = R.randint(rng, key, 0, 100, active=live & (chance >= 0))
        ok = live & ((chance < 0) | (roll < chance))
        sb = dex.move_sec_boosts[move, k]
        st, _ = boost(st, t, sb, ok & jnp.any(sb != 0) & ~mget(st.mons.fainted, t))
        ssb = dex.move_sec_self_boosts[move, k]
        st, _ = boost(st, u, ssb, ok & jnp.any(ssb != 0))
        # secondary status / volatile: M3
        return st, rng
    st, rng = jax.lax.fori_loop(0, 2 * NT, s5, (st, rng))

    # 6. forceSwitch -> forceSwitchFlag (HOOK DragOut)
    def s6(j, st):
        t = targets[j]
        m = st.mons
        go = tv[j] & force_sw & (mget(m.hp, t) > 0) & (mget(m.hp, u) > 0) & (F.can_switch(st, side_of(t)) > 0)
        return st.replace(mons=m.replace(force_switch_flag=mset(m.force_switch_flag, t, True, go)))
    st = jax.lax.fori_loop(0, NT, s6, st)
    # DamagingHit / AfterHit / EmergencyExit: M3
    return st, rng, (tags, vals)


# ------------------------------------------------------------------ trySpreadMoveHit + hit loop
def try_spread_move_hit(st, dex, rng, cfg, u, move, targets, smart, on):
    """PS: sim/battle-actions.ts:612-682 + ch/scripts.ts:426-588. Returns (st, rng, move_result, set_null)."""
    m = st.mons
    on = jnp.asarray(on)
    tv = jnp.stack([targets[j] >= 0 for j in range(NT)]) & on
    n_targets = jnp.sum(tv.astype(I32))
    spread = (n_targets > 1) & ~smart
    fail_any = jnp.zeros((), bool)
    # 0 Invulnerability / 1 TryHit: hooks (M3)
    # 2 type immunity
    ign = dex.move_ignore_immunity[move]
    ignore = (ign == 1) | ((ign == 0) & (dex.move_category[move] == T.CAT_STATUS))
    mtype = jnp.where(move == T.MOVE_STRUGGLE, T.TYPE_TYPELESS, dex.move_type[move])
    imm = []
    for j in range(NT):
        t = targets[j]
        flying = jnp.any(mget(m.types, t) == T.TYPE_FLYING)
        grounded = ~flying & (mget(m.ability, t) != T.ABILITY_LEVITATE)
        immune = jnp.where(mtype == T.TYPE_GROUND, ~grounded, TC.immune(dex, mtype, mget(m.types, t)))
        imm_j = ~ignore & immune & (mtype != T.TYPE_TYPELESS)
        # runImmunity(move, !smartTarget) messages (PS: sim/pokemon.ts:2276-2286): isGrounded null -> Levitate
        lev = (mtype == T.TYPE_GROUND) & ~flying
        st = EV.emit(st, tv[j] & imm_j & ~smart, EV.EV_IMMUNE, actor=t, flags=jnp.where(lev, EV.F_LEVITATE, 0))
        imm.append(imm_j)
    imm = jnp.stack(imm)
    fail_any = fail_any | jnp.any(tv & imm)
    tv = tv & ~imm
    # 3 TryImmunity: powder vs Grass (getImmunity('powder')), move onTryImmunity (M3), Prankster vs Dark (M3)
    powder = mflag(dex, move, T.MF_POWDER)
    pimm = []
    for j in range(NT):
        t = targets[j]
        pimm.append(powder & (t != u) & TC.status_immune(dex, ids.IMMUNITY_KEYS.index("powder"), mget(m.types, t)))
    pimm = jnp.stack(pimm)
    for j in range(NT):
        st = EV.emit(st, tv[j] & pimm[j], EV.EV_IMMUNE, actor=targets[j])            # PS :731
    fail_any = fail_any | jnp.any(tv & pimm)
    tv = tv & ~pimm
    # 4 accuracy
    spread_hit = ((n_targets > 1) & ~smart) | (dex.move_spread_hit[move] == 1)        # PS :613
    st, acc_ok, rng = accuracy_step(st, dex, rng, u, move, targets, tv, on & jnp.any(tv), smart=smart, spread=spread_hit)
    acc_ok = jnp.where(tv, acc_ok, False)
    fail_any = fail_any | jnp.any(tv & ~acc_ok)
    smart = smart & ~jnp.any(tv & ~acc_ok)
    tv = tv & acc_ok
    any_left = jnp.any(tv)
    # 7 hit loop
    st, rng, (dtag, dval) = hit_loop(st, dex, rng, cfg, u, move, targets, tv, spread, smart, on & any_left)
    kept = RS.hit_or_zero((dtag, dval)) & tv
    fail_any = fail_any | jnp.any(tv & RS.is_false((dtag, dval)))
    result = jnp.any(kept)
    set_null = on & ~result & ~fail_any
    # PS :679-680: attrLastMove('[spread] ' + hit slots)
    slots = jnp.int32(0)
    for j in range(NT):
        t = targets[j]
        slots = slots | jnp.where(kept[j], 1 << jnp.clip(side_of(t) * 2 + position(st, t), 0, 3), 0)
    st = EV.attr_last_move(st, on & spread_hit, EV.F_SPREAD, v1=slots)
    return st, rng, result, set_null


def hit_loop(st, dex, rng, cfg, u, move, targets, tv, spread, smart, on):
    """Champions hitStepMoveHitLoop (ch/scripts.ts:426-588) as a fori over hits. Returns (st, rng, (tags, vals))."""
    on = jnp.asarray(on)
    targets = jnp.asarray(targets, I32)
    tv = jnp.asarray(tv)
    mh = dex.move_multihit[move]
    is_range = (mh[0] == 2) & (mh[1] == 5)
    mkey = C.dkey(st, ids.DK_MULTIHIT_COUNT, actor=u, subject=C.eff_move(move))
    idx, rng = R.sample(rng, mkey, 20, active=on & is_range)
    hits = jnp.where(is_range, MULTIHIT_TABLE[idx], jnp.where(mh[0] > 0, mh[0], 1)).astype(I32)
    multiacc = dex.move_multiaccuracy[move] == 1
    single = jnp.sum(tv.astype(I32)) == 1

    def body(h, c):
        st, rng, dt, dv, total, null_damage, hit_done, stopped, lt, lv = c
        m = st.mons
        go = ~stopped & (h <= hits)
        any_false = jnp.any(RS.is_false((dt, dv)))
        alive = (targets >= 0) & (mget(m.hp, jnp.clip(targets, 0, 7)) > 0) & tv
        go = go & ~any_false & jnp.any(alive)
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
        need_acc = go & multiacc & (a > 0) & (h > 1)
        ahit, rng = R.chance(rng, akey, bound, 100, active=need_acc)
        miss = need_acc & ~ahit
        stopped = stopped | miss
        go = go & ~miss
        # [C] scripts.ts:476-482: boolean smartTarget moves retarget the move line on hit 1 (-anim for later hits)
        st = EV.retarget_last_move(st, go & (h == 1) & (dex.move_smart_target[move] == 1) & (t0 >= 0), t0)
        st, rng, (mt, mv) = spread_move_hit(st, dex, rng, cfg, u, move, targets, cur_tv, spread, h, go)
        all_false = ~jnp.any(~RS.is_false((mt, mv)) & cur_tv)
        stopped = stopped | (go & all_false)
        go = go & ~all_false
        null_damage = null_damage & ~go
        cnt = go & cur_tv
        v = jnp.where(RS.truthy((mt, mv)) & RS.is_num((mt, mv)), mv, 0)
        dt = jnp.where(cnt, RS.NUM, dt)
        dv = jnp.where(cnt, v, dv)
        total = total + jnp.sum(jnp.where(cnt, v, 0))
        lt = jnp.where(cnt, mt, lt)
        lv = jnp.where(cnt, mv, lv)
        hit_done = hit_done + go.astype(I32)
        user_dead = mget(st.mons.hp, u) <= 0
        stopped = stopped | (go & user_dead & single) | (h >= hits)
        return st, rng, dt, dv, total, null_damage, hit_done, stopped, lt, lv

    z = jnp.zeros(NT, I32)
    init = (st, rng, jnp.full(NT, RS.NUM, I32), z, jnp.int32(0), jnp.asarray(True), jnp.int32(1), ~on,
            jnp.full(NT, RS.UNDEF, I32), z)
    st, rng, dt, dv, total, null_damage, hit_done, stopped, lt, lv = jax.lax.fori_loop(1, 11, body, init)
    hit_count = hit_done - 1
    all_fail = on & ((hit_done == 1) | null_damage)
    dt = jnp.where(all_fail, RS.BOOL, dt)
    dv = jnp.where(all_fail, 0, dv)
    run_rest = on & ~(hit_done == 1)
    st = F.resolve_faints(st, dex, run_rest, do_check_win=mget(st.mons.hp, u) <= 0, clear_volatile_fn=_cv())
    # [C] scripts.ts:545-549: '-hitcount' for multihit moves without a boolean smartTarget
    st = EV.emit(st, run_rest & (mh[0] > 0) & (dex.move_smart_target[move] != 1), EV.EV_HITCOUNT, actor=targets[0],
                 v1=hit_count)
    st = apply_recoil(st, dex, u, move, total, run_rest & (total > 0))
    # gotAttacked / timesAttacked
    m = st.mons
    ta = m.times_attacked
    for j in range(NT):
        t = targets[j]
        num = run_rest & tv[j] & (t != u) & (lt[j] == RS.NUM)
        ta = mset(ta, t, mget(ta, t) + jnp.where(smart, 1, hit_count), num)
    st = st.replace(mons=m.replace(times_attacked=ta))
    # [C] scripts.ts:569
    st = EV.emit(st, run_rest & (dex.move_ohko[move] > 0) & (mget(st.mons.hp, targets[0]) <= 0), EV.EV_OHKO)
    return st, rng, (dt, dv)


def _cv():
    from vgcjax.battle.switching import clear_volatile
    return clear_volatile


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


# ------------------------------------------------------------------ runMove / useMove
def run_move(st, dex, rng, cfg, side, b, move_slot, move_id, target_loc, orig, on):
    """PS: sim/battle-actions.ts:267-418 runMove (core). Returns (st, rng)."""
    on = jnp.asarray(on)
    u = jnp.asarray(side, I32) * 4 + jnp.asarray(b, I32)
    move = jnp.asarray(move_id, I32)
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
    go = on & ~nopp
    m = m.replace(move_this_turn_result=mset(m.move_this_turn_result, u, S.MR_FALSE, nopp),
                  last_move=mset(m.last_move, u, move, go),
                  last_move_target_loc=mset(m.last_move_target_loc, u, target_loc, go),
                  move_this_turn=mset(m.move_this_turn, u, move, go))
    st = st.replace(mons=m)
    st, rng, result, set_null = use_move(st, dex, rng, cfg, u, move, target, go)
    m = st.mons
    mr = jnp.where(set_null, S.MR_NULL, jnp.where(result, S.MR_TRUE, S.MR_FALSE))
    st = st.replace(mons=m.replace(move_this_turn_result=mset(m.move_this_turn_result, u, mr, go)),
                    meta=st.meta.replace(last_successful_move=jnp.where(go, jnp.where(result, move, 0), st.meta.last_successful_move).astype(I16)))
    # HOOK AfterMove; then resolveFaints() + checkWin()
    from vgcjax.battle.switching import clear_volatile
    st = F.resolve_faints(st, dex, on, clear_volatile_fn=clear_volatile)
    st = F.check_win(st, -1, on)
    return st, rng


def use_move(st, dex, rng, cfg, u, move, target, on):
    """PS: sim/battle-actions.ts:429-610 useMove/useMoveInner (core). Returns (st, rng, result, set_null)."""
    m = st.mons
    st = st.replace(mons=m.replace(move_this_turn_result=mset(m.move_this_turn_result, u, S.MR_UNDEF, on),
                                   last_move_used=mset(m.last_move_used, u, move, on)))
    tt = dex.move_target[move]
    target = jnp.where(_in(tt, (T.TARGET_SELF, T.TARGET_ALLIES)), u, target)
    # HOOK(H3) ModifyType / ModifyMove
    m = st.mons
    alive = ~mget(m.fainted, u)
    # ModifyMove: Struggle adds '-activate' pokemon 'move: Struggle' (data/moves.ts:18220-18223)
    st = EV.emit(st, jnp.asarray(on) & (move == T.MOVE_STRUGGLE), EV.EV_ACTIVATE, actor=u, move=move)
    # PS :519-528: addMove('move', pokemon, move, target); no target -> [notarget] + '-fail'
    st = EV.emit_move(st, jnp.asarray(on) & alive, u, target, move)
    no_target = jnp.asarray(on) & alive & (target < 0)
    st = EV.attr_last_move(st, no_target, EV.F_NOTARGET)
    st = EV.emit(st, no_target, EV.EV_FAIL, actor=u)
    go = jnp.asarray(on) & alive & (target >= 0)
    targets, smart, rng = get_move_targets(st, dex, rng, u, move, target, go)
    # HOOK(H5) Pressure / DeductPP; HOOK(H6) TryMove
    side_field = _in(tt, (T.TARGET_ALL, T.TARGET_FOESIDE, T.TARGET_ALLYSIDE, T.TARGET_ALLYTEAM))
    has_t = targets[0] >= 0
    st = F.faint(st, u, go & (dex.move_selfdestruct[move] == 1))
    empty = go & ~side_field & ~has_t                                   # PS :573-577
    st = EV.attr_last_move(st, empty, EV.F_NOTARGET)
    st = EV.emit(st, empty, EV.EV_FAIL, actor=u)
    st, rng, res, set_null = try_spread_move_hit(st, dex, rng, cfg, u, move, targets, smart, go & has_t & ~side_field)
    # side/field moves (tryMoveHit): M3
    result = go & ((has_t & ~side_field & res) | side_field)
    # selfBoost (Scale Shot) on success
    st, _ = boost(st, u, dex.move_selfboost[move], result & jnp.any(dex.move_selfboost[move] != 0))
    m = st.mons
    st = F.faint(st, u, go & (mget(m.hp, u) <= 0))
    # MoveFail / AfterMoveSecondarySelf / EmergencyExit: M3
    return st, rng, result, set_null & go & has_t
