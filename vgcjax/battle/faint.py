"""Faint queue, win check, end-of-queue faint flags (04 §11.1)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import errors as E
from vgcjax.battle import events as EV
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset, mset_many, or_flag, side_of
from vgcjax.data import ids

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32


def faint(st: S.State, ref, on, src=-1) -> S.State:
    """PS: sim/pokemon.ts:1602-1614 faint(source, effect): queue it (hp = 0, switchFlag = false). `src`: the source
    ref when the effect is a Move (faintData for Moxie's AfterFaint), else -1."""
    m = st.mons
    on = jnp.asarray(on) & (jnp.asarray(ref) >= 0) & ~mget(m.fainted, ref) & ~mget(m.faint_queued, ref)
    m = m.replace(hp=mset(m.hp, ref, 0, on), switch_flag=mset(m.switch_flag, ref, 0, on),
                  faint_queued=mset(m.faint_queued, ref, True, on))
    meta = st.meta
    n = meta.faint_queue_len.astype(I32)
    full = n >= S.FAINTQ
    i = jnp.clip(n, 0, S.FAINTQ - 1)
    fq = meta.faint_queue.at[i].set(jnp.where(on & ~full, jnp.asarray(ref, I8), meta.faint_queue[i]))
    fs = meta.faint_src.at[i].set(jnp.where(on & ~full, jnp.asarray(src, I8), meta.faint_src[i]))
    st = st.replace(mons=m, meta=meta.replace(faint_queue=fq, faint_src=fs,
                                              faint_queue_len=(n + (on & ~full)).astype(I8)))
    return or_flag(st, E.ERR_OVERFLOW_FAINT_QUEUE, on & full)


def faint_many(st: S.State, refs, on, src=-1) -> S.State:
    """faint() for distinct refs in index order (queue appends keep that order), as one batched write."""
    m = st.mons
    refs = jnp.asarray(refs, I32)
    on = jnp.asarray(on) & (refs >= 0) & ~mget(m.fainted, refs) & ~mget(m.faint_queued, refs)
    m = m.replace(hp=mset_many(m.hp, refs, 0, on), switch_flag=mset_many(m.switch_flag, refs, 0, on),
                  faint_queued=mset_many(m.faint_queued, refs, True, on))
    meta = st.meta
    n = meta.faint_queue_len.astype(I32)
    pos = n + jnp.cumsum(on.astype(I32)) - 1
    put = on & (pos < S.FAINTQ)
    hot = (pos[:, None] == jnp.arange(S.FAINTQ)[None, :]) & put[:, None]
    fq = jnp.where(jnp.any(hot, axis=0), jnp.sum(jnp.where(hot, refs[:, None], 0), axis=0).astype(I8), meta.faint_queue)
    fs = jnp.where(jnp.any(hot, axis=0), jnp.asarray(src, I8), meta.faint_src)
    st = st.replace(mons=m, meta=meta.replace(faint_queue=fq, faint_src=fs,
                                              faint_queue_len=(n + jnp.sum(put.astype(I32))).astype(I8)))
    return or_flag(st, E.ERR_OVERFLOW_FAINT_QUEUE, jnp.any(on & ~put))


def damage(st: S.State, ref, d, on):
    """PS: sim/pokemon.ts:1616-1626 pokemon.damage(d): returns (st, dealt). d is a non-negative int."""
    m = st.mons
    hp = mget(m.hp, ref).astype(I32)
    d = jnp.asarray(d, I32)
    on = jnp.asarray(on) & (hp > 0) & (d > 0)
    dealt = jnp.where(on, jnp.minimum(d, hp), 0)
    st = st.replace(mons=m.replace(hp=mset(m.hp, ref, hp - dealt, on)))
    st = faint(st, ref, on & (hp - d <= 0))
    return st, dealt


def heal(st: S.State, ref, h, on):
    """Raw HP gain capped at maxhp (PS pokemon.heal without events). Returns (st, healed)."""
    m = st.mons
    hp = mget(m.hp, ref).astype(I32)
    mx = mget(m.maxhp, ref).astype(I32)
    h = jnp.asarray(h, I32)
    on = jnp.asarray(on) & (hp > 0) & (h > 0) & (hp < mx)
    new = jnp.minimum(mx, hp + h)
    return st.replace(mons=m.replace(hp=mset(m.hp, ref, new, on))), jnp.where(on, new - hp, 0)


def check_win(st: S.State, last_ref, on) -> S.State:
    """PS: sim/battle.ts:2659-2670 checkWin(faintData): both sides out -> side of the last processed faint wins."""
    meta = st.meta
    left = st.sides.pokemon_left.astype(I32)
    on = jnp.asarray(on) & (meta.winner < 0)
    both = (left[0] == 0) & (left[1] == 0)
    w = jnp.where(both, jnp.where(jnp.asarray(last_ref) >= 0, side_of(last_ref), 2),
                  jnp.where(left[1] == 0, 0, jnp.where(left[0] == 0, 1, -1)))
    ended = on & (both | (left[0] == 0) | (left[1] == 0))
    meta = meta.replace(winner=jnp.where(ended, w, meta.winner).astype(I8),
                        phase=jnp.where(ended, S.DONE, meta.phase).astype(I8),
                        ended_reason=jnp.where(ended, S.END_KO, meta.ended_reason).astype(I8))
    return st.replace(meta=meta)


def resolve_faints(st: S.State, dex, on, check_win_if_no_faints=False, do_check_win=True, move_src=None) -> S.State:
    """PS: sim/battle.ts:2561-2657 resolveFaints (gen 9 path) over the whole faint queue at once: queued refs are
    distinct, the per-side counters are additive, and 'faint' messages keep queue order.

    `move_src` (the move user, for the hit-loop tail's resolveFaints, ch/scripts.ts:545): the queued faints carry it
    as faintData.source with its move as the effect, so runEvent('Faint') runs Destiny Bond (effects/batch_utility.py):
    its source.faint() is appended to the queue and processed by the same loop (one more pass), and checkWin is
    forced (PS :2626-2631)."""
    on = jnp.asarray(on) & (st.meta.phase != S.DONE)
    n0 = st.meta.faint_queue_len.astype(I32)
    st, last, db, last_src, n_done = _faint_pass(st, dex, on, move_src)
    if move_src is not None:
        st = faint(st, move_src, db)
        st, last2, _, src2, n2 = _faint_pass(st, dex, on & db, None)
        last = jnp.where(db, last2, last)
        last_src = jnp.where(db, src2, last_src)
        n_done = n_done + jnp.where(db, n2, 0)
        do_check_win = jnp.asarray(do_check_win) | db
    had = n0 > 0
    cw = on & jnp.where(had, do_check_win, check_win_if_no_faints)
    st = check_win(st, jnp.where(had, last, -1), cw)
    # PS :2645-2647: unless checkWin ended the battle, runEvent('AfterFaint', last target, source, effect, length)
    from vgcjax.effects import batch_reactive as BR
    st = BR.moxie(st, last_src, n_done, on & had & (st.meta.phase != S.DONE))
    from vgcjax.effects import batch7a as B7                   # Eelevate onSourceAfterFaint (same event as Moxie)
    return B7.eelevate(st, last_src, n_done, on & had & (st.meta.phase != S.DONE))


def _faint_pass(st: S.State, dex, on, move_src):
    """One batched pass of resolveFaints' loop over the queued faints; empties the queue. Returns (st, last ref,
    Destiny Bond activated)."""
    from vgcjax.battle.switching import clear_volatile_many
    meta = st.meta
    n0 = meta.faint_queue_len.astype(I32)
    refs = meta.faint_queue.astype(I32)
    live = on & (jnp.arange(S.FAINTQ) < n0) & (refs >= 0)
    do = live & ~mget(st.mons.fainted, refs)             # BeforeFaint: no handlers yet
    db = jnp.zeros((), bool)
    for k in range(S.FAINTQ):
        st = EV.emit(st, do[k], EV.EV_FAINT, actor=refs[k])     # PS: sim/battle.ts:2598 add("faint", pokemon)
        if move_src is not None:                                 # runEvent('Faint') (PS :2602-2604): Destiny Bond
            from vgcjax.effects import batch_utility as BU
            st, act = BU.destinybond_faint(st, refs[k], move_src, do[k] & ~db)
            db = db | act
        from vgcjax.effects import batch_forme as BF            # Illusion onFaint / PS :2613 illusion = null
        st = BF.illusion_clear(st, refs[k], do[k])
    side = side_of(jnp.clip(refs, 0, 7))
    cnt = jnp.stack([jnp.sum((do & (side == s)).astype(I32)) for s in range(2)])
    sides = st.sides
    st = st.replace(sides=sides.replace(
        pokemon_left=jnp.maximum(sides.pokemon_left.astype(I32) - cnt, 0).astype(sides.pokemon_left.dtype),
        total_fainted=jnp.minimum(sides.total_fainted.astype(I32) + cnt, 100).astype(sides.total_fainted.dtype),
        fainted_this_turn=sides.fainted_this_turn | (cnt > 0)))
    # Faint event (M4), singleEvent('End', ability) (PS :2604; no messages yet, so after all 'faint' lines), item End
    from vgcjax.effects import abilities as AB
    from vgcjax.effects import batch_reactive as BR
    st = BR.mold_faint(st, refs, do)                 # Mold Breaker: End is not suppressed; the mover's faint ends it
    st = AB.end_vec(st, refs, do)
    st = clear_volatile_many(st, dex, refs, False, do)
    m = st.mons
    st = st.replace(mons=m.replace(fainted=mset_many(m.fainted, refs, True, do),
                                   is_active=mset_many(m.is_active, refs, False, do),
                                   is_started=mset_many(m.is_started, refs, False, do)))
    last = jnp.where(n0 > 0, refs[jnp.clip(n0 - 1, 0, S.FAINTQ - 1)], -1)
    last_src = jnp.where(n0 > 0, meta.faint_src.astype(I32)[jnp.clip(n0 - 1, 0, S.FAINTQ - 1)], -1)
    meta = st.meta
    st = st.replace(meta=meta.replace(faint_queue=jnp.where(on, jnp.full(S.FAINTQ, -1, I8), meta.faint_queue),
                                      faint_src=jnp.where(on, jnp.full(S.FAINTQ, -1, I8), meta.faint_src),
                                      faint_queue_len=jnp.where(on, 0, meta.faint_queue_len).astype(I8)))
    return st, last, db, last_src, n0


def possible_switches(st: S.State, side):
    """PS: sim/battle.ts:1589-1603: bench mons (party index >= 2) not fainted and not targeted by a queued
    switch/instaswitch. Returns bool[4] over brought index."""
    q = st.queue
    side = jnp.asarray(side, I32)
    queued = q.valid & ((q.kind == S.Q_SWITCH) | (q.kind == S.Q_INSTASWITCH)) & (q.side == side)
    out = []
    order = st.sides.ps_order[side].astype(I32)
    for b in range(4):
        pos = jnp.argmax(order == b)
        targeted = jnp.any(queued & (q.switch_target_b == b))
        out.append((pos >= 2) & ~mget(st.mons.fainted, side * 4 + b) & ~targeted & (st.sides.pokemon_left[side] > 0))
    return jnp.stack(out)


def can_switch(st: S.State, side):
    return jnp.sum(possible_switches(st, side).astype(I32))


def check_fainted(st: S.State, on) -> S.State:
    """PS: sim/battle.ts checkFainted(): fainted actives of sides that can switch get status fnt + switchFlag."""
    m = st.mons
    for side in range(2):
        cs = can_switch(st, side) > 0
        for slot in range(2):
            b = st.sides.active[side, slot].astype(I32)
            ref = side * 4 + b
            do = jnp.asarray(on) & cs & (b >= 0) & mget(m.fainted, ref)
            m = m.replace(status=mset(m.status, ref, ids.COND_FNT, do), switch_flag=mset(m.switch_flag, ref, 1, do))
    return st.replace(mons=m)
