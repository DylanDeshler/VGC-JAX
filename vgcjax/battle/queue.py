"""Action queue (04 §5): fixed QMAX entries, PS order classes, re-keying, selection with SPEED_TIE_SELECT.

PS sorts the queue and shifts the head; the engine keeps entries unsorted and selects the argmax of
(prepend, -order, priority10, speed). Exact key ties among live actors are resolved by the forced/natural
SPEED_TIE_SELECT record (contract §4); other ties go to insertion order (ADR-101: inserted actions go after the
equal-key actions already queued).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import errors as E
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, or_flag, ref_of
from vgcjax.core import rng as R
from vgcjax.data import ids

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32
SELECTABLE = (S.Q_SWITCH, S.Q_INSTASWITCH, S.Q_MEGA, S.Q_BEFORE_TURN_MOVE, S.Q_PRIORITY_CHARGE, S.Q_REVIVAL, S.Q_MOVE)
ACTOR_CHECKED = (S.Q_MOVE, S.Q_BEFORE_TURN_MOVE, S.Q_PRIORITY_CHARGE)  # early return if actor inactive/fainted


def is_kind(kind, kinds):
    k = jnp.asarray(kind)
    out = jnp.zeros(k.shape, bool)
    for x in kinds:
        out = out | (k == x)
    return out


def add_entry(st: S.State, on, *, kind, side=0, b=-1, order=None, priority10=0, frac10=0, speed=0, move_slot=-1,
              move_id=0, target_loc=0, original_target=-1, mega=False, switch_target_b=-1, source_effect=0,
              prepend=False) -> S.State:
    """Write one entry into the first free slot (PS addChoice / insertChoice after resolveAction)."""
    q = st.queue
    on = jnp.asarray(on)
    free = ~q.valid
    has_free = jnp.any(free)
    i = jnp.argmax(free)
    ok = on & has_free
    st = or_flag(st, E.ERR_OVERFLOW_QUEUE, on & ~has_free)
    order = S.ORDER[kind] if order is None and isinstance(kind, int) else order
    vals = dict(valid=True, prepend=prepend, kind=kind, order=order, priority10=priority10, frac10=frac10, speed=speed,
                side=side, actor_b=b, move_slot=move_slot, move_id=move_id, target_loc=target_loc,
                original_target=original_target, mega=mega, switch_target_b=switch_target_b,
                source_effect=source_effect, seq=st.scratch.queue_seq)
    upd = {}
    for k, v in vals.items():
        arr = getattr(q, k)
        upd[k] = arr.at[i].set(jnp.where(ok, jnp.asarray(v, arr.dtype), arr[i]))
    sc = st.scratch
    return st.replace(queue=q.replace(**upd), scratch=sc.replace(queue_seq=sc.queue_seq + ok.astype(I16)))


def entry_ref(q: S.Queue):
    return jnp.where(q.actor_b >= 0, ref_of(q.side, q.actor_b), -1)


def move_priority(st: S.State, dex, ref, move_id):
    """Base priority + ModifyPriority (hooks in M3). PS: sim/battle.ts:2695-2702 getActionSpeed for moves."""
    return dex.move_priority[jnp.asarray(move_id, I32)].astype(I32)


def frac_priority(st: S.State, dex, rng, ref, move_id, on):
    """runEvent('FractionalPriority') at resolve, x10 (Stall -0.1; Quick Claw / Quick Draw join with items/abilities).
    PS: sim/battle-queue.ts:249, data/abilities.ts:4498 (Stall)."""
    stall = mget(st.mons.ability, ref) == ids.ABILITY_STALL
    return jnp.where(stall, -1, 0), rng


def rekey(st: S.State, dex, on) -> S.State:
    """Before a move is selected: updateSpeed, then recompute priority and speed of every queued action with a pokemon
    and clear prepend (PS: sim/battle.ts:2975-2984; 04 §5.4)."""
    st = C.update_speed(st, dex, on)
    q = st.queue
    refs = entry_ref(q)
    has = q.valid & (refs >= 0)
    speeds = jnp.stack([C.action_speed(st, dex, r) for r in range(8)])
    new_speed = speeds[jnp.clip(refs, 0, 7)]
    prio = 10 * move_priority(st, dex, refs, q.move_id) + q.frac10.astype(I32)
    is_move = q.kind == S.Q_MOVE
    on = jnp.asarray(on)
    q = q.replace(speed=jnp.where(on & has, new_speed, q.speed).astype(I16),
                  priority10=jnp.where(on & has & is_move, prio, q.priority10).astype(I16),
                  prepend=jnp.where(on, False, q.prepend))
    return st.replace(queue=q)


def best_mask(q: S.Queue):
    """Entries with the maximal selection key (prepend, -order, priority10, speed)."""
    v = q.valid
    big = jnp.int32(1 << 30)
    m = v & (q.prepend == jnp.any(v & q.prepend))
    o = jnp.min(jnp.where(m, q.order.astype(I32), big))
    m = m & (q.order == o)
    p = jnp.max(jnp.where(m, q.priority10.astype(I32), -big))
    m = m & (q.priority10 == p)
    s = jnp.max(jnp.where(m, q.speed.astype(I32), -big))
    return m & (q.speed == s)


def head(q: S.Queue):
    """(index, kind, any) of the canonical head (lowest seq among the best entries), without tie draws."""
    m = best_mask(q)
    i = jnp.argmin(jnp.where(m, q.seq.astype(I32), 1 << 30))
    return i, jnp.where(jnp.any(q.valid), q.kind[i], S.Q_NONE), jnp.any(q.valid)


def select(st: S.State, dex, rng: R.Rng, on):
    """Pick the next entry. Returns (st, rng, index, passes) where `passes` = the action does not early-return.

    Ties among >= 2 entries whose actor is active and not fainted draw SPEED_TIE_SELECT with sub = select_ord
    (contract §4). select_ord advances for executed selectable actions only.
    """
    q = st.queue
    on = jnp.asarray(on)
    m = best_mask(q)
    refs = entry_ref(q)
    lv = jnp.stack([C.live(st, r) for r in range(8)])
    entry_live = m & (refs >= 0) & lv[jnp.clip(refs, 0, 7)]
    nlive = jnp.sum(entry_live.astype(I32))
    canon = jnp.argmin(jnp.where(m, q.seq.astype(I32), 1 << 30))
    # tie among live actors -> SPEED_TIE_SELECT over members = refs (sorted)
    members = jnp.arange(8, dtype=I32)
    mem_valid = jnp.stack([jnp.any(entry_live & (refs == r)) for r in range(8)])
    sc = st.scratch
    # only selectable kinds take part in SPEED_TIE_SELECT; other ties (runSwitch) go to insertion order (ADR-101)
    tie = (nlive >= 2) & is_kind(q.kind[canon], SELECTABLE)
    order, rng = R.tie_order(rng, C.tkey(st, ids.DK_SPEED_TIE_SELECT, sc.select_ord), members, mem_valid,
                             active=on & tie, select=True)
    win_ref = members[order[0]]
    win_idx = jnp.argmin(jnp.where(entry_live & (refs == win_ref), q.seq.astype(I32), 1 << 30))
    idx = jnp.where(tie, win_idx, canon)
    kind = q.kind[idx]
    r = refs[idx]
    passes = ~is_kind(kind, ACTOR_CHECKED) | ((r >= 0) & lv[jnp.clip(r, 0, 7)])
    selectable = is_kind(kind, SELECTABLE)
    adv = on & selectable & passes
    st = st.replace(scratch=sc.replace(cur_select_ord=jnp.where(adv, sc.select_ord, sc.cur_select_ord),
                                       select_ord=sc.select_ord + adv.astype(I16)))
    # remove the entry (PS queue.shift)
    q = q.replace(valid=q.valid.at[idx].set(jnp.where(on, False, q.valid[idx])))
    return st.replace(queue=q), rng, idx, passes


def cancel_action(st: S.State, ref, on) -> S.State:
    """PS: sim/battle-queue.ts:334-343 cancelAction(p): remove all of p's actions except runSwitch."""
    q = st.queue
    kill = jnp.asarray(on) & q.valid & (entry_ref(q) == ref) & (q.kind != S.Q_RUN_SWITCH)
    return st.replace(queue=q.replace(valid=q.valid & ~kill))


def peek_kind(q: S.Queue):
    """Kind of the entry PS would `peek()` next (canonical head)."""
    return head(q)[1]
