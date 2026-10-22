"""Switching (04 §10): clearVolatile, switchIn, batched runSwitch, dragIn."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import faint as F
from vgcjax.battle import queue as Q
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset
from vgcjax.core import rng as R
from vgcjax.core import stats as ST
from vgcjax.data import ids

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32


def set_species(st: S.State, dex, ref, species, on) -> S.State:
    """PS: sim/pokemon.ts:1408-1441 setSpecies: species, types, weight, storedStats; speed = stored Spe."""
    m = st.mons
    sp = jnp.asarray(species, I32)
    nat = mget(m.nature, ref).astype(I32)
    _, stats = ST.calc_stats(dex.species_base_stats[sp], mget(m.sp, ref).astype(I32), dex.nature_plus[nat],
                             dex.nature_minus[nat])
    m = m.replace(species=mset(m.species, ref, sp, on), types=mset(m.types, ref, dex.species_types[sp], on),
                  added_type=mset(m.added_type, ref, 0, on), weight_hg=mset(m.weight_hg, ref, dex.species_weight_hg[sp], on),
                  stats=mset(m.stats, ref, stats, on), speed=mset(m.speed, ref, stats[4], on))
    return st.replace(mons=m)


def clear_volatile(st: S.State, dex, ref, include_switch_flags, on) -> S.State:
    """[C] PS: data/mods/champions/scripts.ts:123-176 clearVolatile (also resets timesAttacked)."""
    m = st.mons
    on = jnp.asarray(on) & (jnp.asarray(ref) >= 0)
    sw = on & jnp.asarray(include_switch_flags)
    m = m.replace(
        boosts=mset(m.boosts, ref, jnp.zeros(7, I8), on),
        ability=mset(m.ability, ref, mget(m.base_ability, ref), on),
        vol_bits=mset(m.vol_bits, ref, jnp.zeros(S.WV, jnp.uint32), on),
        vol_state=mset(m.vol_state, ref, jnp.zeros((S.NV, 6), I16), on),
        switch_flag=mset(m.switch_flag, ref, 0, sw), force_switch_flag=mset(m.force_switch_flag, ref, False, sw),
        last_move=mset(m.last_move, ref, 0, on), last_move_used=mset(m.last_move_used, ref, 0, on),
        move_this_turn=mset(m.move_this_turn, ref, 0, on),
        move_last_turn_result=mset(m.move_last_turn_result, ref, S.MR_UNDEF, on),
        move_this_turn_result=mset(m.move_this_turn_result, ref, S.MR_UNDEF, on),
        last_damage=mset(m.last_damage, ref, 0, on), hurt_this_turn=mset(m.hurt_this_turn, ref, -1, on),
        newly_switched=mset(m.newly_switched, ref, True, on), being_called_back=mset(m.being_called_back, ref, False, on),
        times_attacked=mset(m.times_attacked, ref, 0, on),
    )
    st = st.replace(mons=m)
    return set_species(st, dex, ref, mget(m.base_species, ref), on)


def switch_in(st: S.State, dex, rng, side, pos, b_in, on, source_effect=0, is_drag=False):
    """PS: sim/battle-actions.ts:79-215 switchIn(pokemon, pos, sourceEffect, isDrag). Returns (st, rng)."""
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
    # BeforeSwitchIn hooks (M3)
    if is_drag:
        st, rng = run_switch(st, dex, rng, [r_in], on)
    else:
        # insertChoice({choice: 'runSwitch', pokemon}): updateSpeed then insert by key (ADR-101 order)
        st = C.update_speed_one(st, dex, r_in, on)
        st = Q.add_entry(st, on, kind=S.Q_RUN_SWITCH, side=side, b=b_in, speed=mget(st.mons.speed, r_in))
    return st, rng


def switchin_ties(st: S.State, rng: R.Rng, on):
    """runSwitch's speedSort of all slot occupants (incl. fainted) by cached speed: one SPEED_TIE_SWITCHIN record per
    tie group (contract §4; members in slot order p1a, p1b, p2a, p2b; sub = runswitch_ord; ord = group index)."""
    act = st.sides.active.reshape(4).astype(I32)          # slot k = side*2 + pos
    occ = act >= 0
    refs = jnp.where(occ, (jnp.arange(4) // 2) * 4 + act, -1)
    spd = jnp.where(occ, mget(st.mons.speed, jnp.clip(refs, 0, 7)).astype(I32), -(1 << 20))
    sub = st.scratch.runswitch_ord
    count = jnp.sum((spd[:, None] == spd[None, :]) & occ[None, :] & occ[:, None], axis=1)  # same-speed occupants
    js = jnp.arange(4)
    earlier_same = (spd[None, :] == spd[:, None]) & occ[None, :] & (js[None, :] < js[:, None])
    grp_first = occ & (count >= 2) & ~jnp.any(earlier_same, axis=1)

    def body(k, rng):
        higher = (spd > spd[k]) & grp_first                  # tie groups with strictly higher speed
        ord_k = jnp.sum(higher.astype(I32))
        mem = occ & (spd == spd[k])
        _, rng = R.tie_order(rng, _tkey(st, ids.DK_SPEED_TIE_SWITCHIN, sub, ord_k), jnp.clip(refs, 0, 7), mem,
                             active=jnp.asarray(on) & grp_first[k])
        return rng
    return jax.lax.fori_loop(0, 4, body, rng)


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
    rng = switchin_ties(st, rng, on)
    sc = st.scratch
    st = st.replace(scratch=sc.replace(runswitch_ord=sc.runswitch_ord + on.astype(I16),
                                       fieldevent_ord=sc.fieldevent_ord + on.astype(I16)))
    # fieldEvent('SwitchIn', switchers): hazards / abilities / items (M3)
    m = st.mons
    started = m.is_started
    for i in range(4):
        r = switchers[i]
        started = mset(started, r, True, on & (r >= 0) & (mget(m.hp, r) > 0))
    return st.replace(mons=m.replace(is_started=started)), rng


def drag_in(st: S.State, dex, rng, side, pos, on):
    """PS: sim/battle-actions.ts:216-227 dragIn: FORCED_SWITCH_PICK over possible switches (party order), then
    switchIn(isDrag) with an inline runSwitch. Returns (st, rng)."""
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
    return switch_in(st, dex, rng, side, pos, pick, ok, is_drag=True)
