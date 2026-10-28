"""Switching (04 §10): clearVolatile, switchIn, batched runSwitch, dragIn."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import faint as F
from vgcjax.battle import queue as Q
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset, mset_many
from vgcjax.core import rng as R
from vgcjax.core import stats as ST
from vgcjax.data import ids

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32


def set_species_many(st: S.State, dex, refs, species, on) -> S.State:
    """PS: sim/pokemon.ts:1408-1441 setSpecies for distinct refs [T]: species, types, weight, storedStats; speed =
    stored Spe."""
    m = st.mons
    refs = jnp.asarray(refs, I32)
    sp = jnp.asarray(species, I32)
    nat = mget(m.nature, refs).astype(I32)
    _, stats = ST.calc_stats(dex.species_base_stats[sp], mget(m.sp, refs).astype(I32), dex.nature_plus[nat],
                             dex.nature_minus[nat])
    m = m.replace(species=mset_many(m.species, refs, sp, on), types=mset_many(m.types, refs, dex.species_types[sp], on),
                  added_type=mset_many(m.added_type, refs, 0, on),
                  weight_hg=mset_many(m.weight_hg, refs, dex.species_weight_hg[sp], on),
                  stats=mset_many(m.stats, refs, stats, on), speed=mset_many(m.speed, refs, stats[:, 4], on))
    return st.replace(mons=m)


def set_species(st: S.State, dex, ref, species, on) -> S.State:
    return set_species_many(st, dex, jnp.asarray(ref, I32)[None], jnp.asarray(species, I32)[None],
                            jnp.asarray(on)[None])


def clear_volatile_many(st: S.State, dex, refs, include_switch_flags, on) -> S.State:
    """[C] PS: data/mods/champions/scripts.ts:123-176 clearVolatile (also resets timesAttacked) for distinct refs [T]."""
    m = st.mons
    refs = jnp.asarray(refs, I32)
    T = refs.shape[0]
    on = jnp.broadcast_to(jnp.asarray(on), (T,)) & (refs >= 0)
    sw = on & jnp.asarray(include_switch_flags)
    w = lambda arr, val, mask=on: mset_many(arr, refs, val, mask)  # noqa: E731
    m = m.replace(
        boosts=w(m.boosts, 0), ability=w(m.ability, mget(m.base_ability, refs)), vol_bits=w(m.vol_bits, 0),
        vol_dur=w(m.vol_dur, 0), vol_fx=w(m.vol_fx, 0), switch_flag=w(m.switch_flag, 0, sw), force_switch_flag=w(m.force_switch_flag, False, sw),
        last_move=w(m.last_move, 0), last_move_used=w(m.last_move_used, 0), move_this_turn=w(m.move_this_turn, 0),
        move_last_turn_result=w(m.move_last_turn_result, S.MR_UNDEF),
        move_this_turn_result=w(m.move_this_turn_result, S.MR_UNDEF),
        last_damage=w(m.last_damage, 0), hurt_this_turn=w(m.hurt_this_turn, -1), newly_switched=w(m.newly_switched, True),
        being_called_back=w(m.being_called_back, False), times_attacked=w(m.times_attacked, 0),
    )
    st = st.replace(mons=m)
    return set_species_many(st, dex, refs, mget(m.base_species, refs), on)


def clear_volatile(st: S.State, dex, ref, include_switch_flags, on) -> S.State:
    return clear_volatile_many(st, dex, jnp.asarray(ref, I32)[None], include_switch_flags, jnp.asarray(on)[None])


def switch_in(st: S.State, dex, rng, side, pos, b_in, on, source_effect=0, is_drag=False):
    """PS: sim/battle-actions.ts:79-215 switchIn(pokemon, pos, sourceEffect, isDrag). Returns (st, rng). `is_drag` may
    be traced; a drag's inline runSwitch (PS :208-210) is run by the caller through the single run_switch site."""
    is_drag = jnp.asarray(is_drag)
    side = jnp.asarray(side, I32)
    pos = jnp.asarray(pos, I32)
    b_in = jnp.asarray(b_in, I32)
    r_in = side * 4 + b_in
    m = st.mons
    on = jnp.asarray(on) & (b_in >= 0) & ~mget(m.is_active, r_in)
    b_old = st.sides.active[side, pos].astype(I32)
    r_old = jnp.where(b_old >= 0, side * 4 + b_old, -1)
    has_old = on & (b_old >= 0)
    unfainted = has_old & (mget(m.hp, r_old) > 0)
    # switch-out of a live occupant: SwitchOut / ability & item End hooks (M3), cancelAction, clearVolatile
    m = m.replace(being_called_back=mset(m.being_called_back, r_old, True, unfainted))
    st = st.replace(mons=m)
    # PS :120-121: singleEvent('End', ability) (item End: M4)
    from vgcjax.effects import abilities as AB
    from vgcjax.effects import ability_reactions as AR
    st, rng = AR.switch_out(st, dex, rng, r_old, unfainted)              # runEvent('SwitchOut') (PS :113)
    st = AB.end_vec(st, r_old[None], unfainted[None])
    st = Q.cancel_action(st, r_old, unfainted)
    st = clear_volatile(st, dex, r_old, True, unfainted)
    m = st.mons
    # old occupant leaves (live or fainted)
    m = m.replace(is_active=mset(m.is_active, r_old, False, has_old), is_started=mset(m.is_started, r_old, False, has_old),
                  used_item_this_turn=mset(m.used_item_this_turn, r_old, False, has_old),
                  stats_raised_this_turn=mset(m.stats_raised_this_turn, r_old, False, has_old),
                  stats_lowered_this_turn=mset(m.stats_lowered_this_turn, r_old, False, has_old),
                  status=mset(m.status, r_old, 0, has_old & mget(m.fainted, r_old)))
    # party order swap: incoming takes the outgoing's index (PS :137-154)
    order = st.sides.ps_order[side].astype(I32)
    p_in = jnp.argmax(order == b_in)
    new_order = jnp.where(has_old, order.at[pos].set(b_in).at[p_in].set(b_old), order)
    sides = st.sides.replace(ps_order=st.sides.ps_order.at[side].set(jnp.where(on, new_order, order).astype(I8)),
                             active=st.sides.active.at[side, pos].set(jnp.where(on, b_in, b_old).astype(I8)))
    # incoming
    meta = st.meta
    seq = meta.effect_seq
    has_ab = on & (mget(m.ability, r_in) > 0)
    ab_seq = jnp.where(has_ab, seq, 0)
    seq = seq + has_ab.astype(I32)
    has_it = on & (mget(m.item, r_in) > 0)
    it_seq = jnp.where(has_it, seq, 0)
    seq = seq + has_it.astype(I32)
    m = m.replace(is_active=mset(m.is_active, r_in, True, on), active_turns=mset(m.active_turns, r_in, 0, on),
                  active_move_actions=mset(m.active_move_actions, r_in, 0, on),
                  move_used=mset(m.move_used, r_in, jnp.zeros(4, bool), on),
                  ability_state=mset(m.ability_state, r_in, jnp.zeros(4, I16), on),
                  item_state=mset(m.item_state, r_in, jnp.zeros(2, I16), on),
                  ability_seq=mset(m.ability_seq, r_in, ab_seq, on), item_seq=mset(m.item_seq, r_in, it_seq, on),
                  previously_switched_in=mset(m.previously_switched_in, r_in, mget(m.previously_switched_in, r_in) + 1, on))
    st = st.replace(mons=m, sides=sides, meta=meta.replace(effect_seq=jnp.where(on, seq, meta.effect_seq)))
    # PS :186-192: add(isDrag ? 'drag' : 'switch', pokemon, details, [from] sourceEffect)
    st = EV.emit_hp(st, on, EV.EV_SWITCH, r_in, move=jnp.where(is_drag, 0, source_effect),
                    flags=jnp.where(is_drag, EV.F_DRAG, 0))
    # BeforeSwitchIn hooks (M3); a drag runs runSwitch inline (caller), a switch inserts it:
    # insertChoice({choice: 'runSwitch', pokemon}): updateSpeed then insert by key (RUNSWITCH_INSERT_TIE)
    ins = on & ~is_drag
    st = C.update_speed_one(st, dex, r_in, ins)
    st = C.mul_tick(st, 2, ins)             # pokemon.updateSpeed() + resolveAction getActionSpeed
    st, rng = Q.insert_run_switch(st, rng, ins, side, b_in, mget(st.mons.speed, r_in))
    return st, rng


def slot_speed_sort(st: S.State, rng: R.Rng, occ, kind, sub, on):
    """speedSort of the slot occupants `occ` [4] (slot k = side*2 + pos) by cached pokemon.speed, ties drawn as one
    record of `kind` per tie group (contract §4: sub, ord = group index, members in slot order p1a, p1b, p2a, p2b; 4
    occupants form at most 2 groups of size >= 2). Returns (rng, order [4] slots in execution order (unoccupied last),
    refs [4])."""
    act = st.sides.active.reshape(4).astype(I32)
    occ = jnp.asarray(occ) & (act >= 0)
    refs = jnp.where(act >= 0, (jnp.arange(4) // 2) * 4 + act, -1)
    spd = jnp.where(occ, mget(st.mons.speed, jnp.clip(refs, 0, 7)).astype(I32), -(1 << 20))
    count = jnp.sum((spd[:, None] == spd[None, :]) & occ[None, :] & occ[:, None], axis=1)  # same-speed occupants
    js = jnp.arange(4)
    earlier_same = (spd[None, :] == spd[:, None]) & occ[None, :] & (js[None, :] < js[:, None])
    grp_first = occ & (count >= 2) & ~jnp.any(earlier_same, axis=1)
    # group index = number of tie groups with strictly higher speed
    gord = jnp.sum(((spd[None, :] > spd[:, None]) & grp_first[None, :]).astype(I32), axis=1)
    rank_in_group = jnp.zeros(4, I32)
    for g in range(2):
        sel = grp_first & (gord == g)
        k = jnp.argmax(sel)
        mem = occ & (spd == spd[k])
        torder, rng = R.tie_order(rng, _tkey(st, kind, sub, g), jnp.clip(refs, 0, 7), mem,
                                  active=jnp.asarray(on) & jnp.any(sel))
        pos_of = jnp.argsort(torder)                       # position of member k in the drawn order
        rank_in_group = jnp.where(jnp.any(sel) & mem, pos_of, rank_in_group)
    key = jnp.where(occ, -spd * 8 + rank_in_group, 1 << 28)
    return rng, jnp.argsort(key, stable=True), refs


def switchin_ties(st: S.State, rng: R.Rng, on):
    """runSwitch's speedSort of all slot occupants (incl. fainted) by cached speed: one SPEED_TIE_SWITCHIN record per
    tie group (contract §4; sub = runswitch_ord)."""
    occ = st.sides.active.reshape(4) >= 0
    rng, order, _ = slot_speed_sort(st, rng, occ, ids.DK_SPEED_TIE_SWITCHIN, st.scratch.runswitch_ord, on)
    # battle.speedOrder = getAllActive(true) sorted (PS: sim/battle-actions.ts:234-236): field position values
    # (side + 2 * pos) of the occupied slots by speed, ties in the drawn order; -1 pads empty slots
    fieldpos = (order // 2) + 2 * (order % 2)
    so = jnp.where(occ[order], fieldpos, -1).astype(st.meta.speed_order.dtype)
    return rng, so


def _tkey(st, kind, sub, ord_):
    return C.tkey(st, kind, sub, ord_)


def run_switch(st: S.State, dex, rng, refs, on):
    """PS: sim/battle-actions.ts:228-249 runSwitch. `refs` = switchers already known (inline / first entry); queued
    runSwitch entries at the head are drained into the batch. Returns (st, rng)."""
    on = jnp.asarray(on)
    switchers = jnp.full(4, -1, I32)
    for i, r in enumerate(refs):
        switchers = switchers.at[i].set(jnp.where(on, r, -1))
    n = len(refs)
    if n == 1:  # drain queued runSwitch entries at the head (batching)
        for i in range(1, 4):
            q = st.queue
            idx, kind, anyv = Q.head(q)
            take = on & anyv & (kind == S.Q_RUN_SWITCH)
            r = jnp.where(take, Q.entry_ref(q)[idx], -1)
            switchers = switchers.at[i].set(r)
            st = st.replace(queue=q.replace(valid=q.valid.at[idx].set(jnp.where(take, False, q.valid[idx]))))
    rng, so = switchin_ties(st, rng, on)
    st = st.replace(meta=st.meta.replace(speed_order=jnp.where(on, so, st.meta.speed_order)))
    sc = st.scratch
    st = st.replace(scratch=sc.replace(runswitch_ord=sc.runswitch_ord + on.astype(I16)))
    # fieldEvent('SwitchIn', switchers): status / volatiles / abilities / items / hazards (effects/fieldevent.py)
    from vgcjax.effects import fieldevent as FE
    st, rng = FE.switch_in(st, dex, rng, switchers, on)
    m = st.mons
    started = m.is_started
    for i in range(4):
        r = switchers[i]
        started = mset(started, r, True, on & (r >= 0) & (mget(m.hp, r) > 0))
    return st.replace(mons=m.replace(is_started=started)), rng


def drag_pick(st: S.State, rng, side, pos, on):
    """PS: sim/battle-actions.ts:216-227 dragIn up to switchIn: FORCED_SWITCH_PICK over possible switches (party
    order); no candidate or a fainted occupant -> no switch. Returns (brought index, ok, rng)."""
    side = jnp.asarray(side, I32)
    poss = F.possible_switches(st, side)                     # bool[4] by brought index
    order = st.sides.ps_order[side].astype(I32)
    cand_b = order                                           # party order (PS iterates side.pokemon)
    cand_ok = poss[cand_b]
    b_old = st.sides.active[side, pos].astype(I32)
    r_old = side * 4 + b_old
    key = C.dkey(st, ids.DK_FORCED_SWITCH_PICK, actor=r_old, subject=st.scratch.step_subject)
    pick, rng = R.sample_ref(rng, key, cand_b, cand_ok, active=jnp.asarray(on) & jnp.any(cand_ok))
    ok = jnp.asarray(on) & jnp.any(cand_ok) & (mget(st.mons.hp, r_old) > 0)
    # DragOut hooks (M3)
    return pick, ok, rng
