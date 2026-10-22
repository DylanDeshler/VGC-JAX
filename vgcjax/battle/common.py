"""Shared engine primitives: EffectIds, DrawKeys from the running step, speed, condition bits."""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import state as S
from vgcjax.battle.util import bit_test, mget, mset
from vgcjax.core import drawkey as dkm
from vgcjax.core import stats as ST
from vgcjax.data import ids

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32


# ------------------------------------------------------------------ EffectIds (03 §1)
def eff_move(m):
    m = jnp.asarray(m, I32)
    return jnp.where(m > 0, ids.MOVE_BASE + m - 1, -1)


def eff_ability(a):
    a = jnp.asarray(a, I32)
    return jnp.where(a > 0, ids.ABILITY_BASE + a - 1, -1)


def eff_item(i):
    i = jnp.asarray(i, I32)
    return jnp.where(i > 0, ids.ITEM_BASE + i - 1, -1)


def eff_cond(c):
    c = jnp.asarray(c, I32)
    return jnp.where(c > 0, ids.COND_BASE + c - 1, -1)


def eff_action(name: str) -> int:
    return ids.ACTION_BASE + ids.ACTION_KINDS.index(name)


# ------------------------------------------------------------------ DrawKeys (contract §3)
def dkey(st: S.State, kind, actor=-1, target=-1, subject=-1, hit=0, ord=0):
    sc = st.scratch
    return dkm.DrawKey(kind=kind, turn=st.meta.turn, step_tag=sc.step_tag, step_actor=sc.step_actor,
                       step_subject=sc.step_subject, actor=actor, target=target, subject=subject, hit=hit, ord=ord,
                       sub=0, action_ord=sc.action_ord)


def tkey(st: S.State, kind, sub, ord=0):
    """Tie-record key (contract §4): only (turn, step_tag, sub, ord) besides the kind; other fields are 0."""
    return dkm.DrawKey(kind=kind, turn=st.meta.turn, step_tag=st.scratch.step_tag, step_actor=0, step_subject=0,
                       actor=0, target=0, subject=0, hit=0, ord=ord, sub=sub, action_ord=0)


def set_step(st: S.State, tag, actor, subject, action_ord=0, on=True) -> S.State:
    sc = st.scratch
    w = lambda new, old: jnp.where(on, jnp.asarray(new, old.dtype), old)  # noqa: E731
    return st.replace(scratch=sc.replace(step_tag=w(tag, sc.step_tag), step_actor=w(actor, sc.step_actor),
                                         step_subject=w(subject, sc.step_subject),
                                         action_ord=w(action_ord, sc.action_ord)))


# ------------------------------------------------------------------ field / condition bits
def pw_active(st: S.State, cid: int):
    return bit_test(st.field.pw_bits, cid - ids.PW_BASE)


def trick_room(st: S.State):
    return pw_active(st, ids.COND_TRICKROOM)


def vol_active(st: S.State, ref, cid: int):
    k = cid - ids.VOL_BASE
    word = mget(st.mons.vol_bits, ref)[k // 32]
    return bit_test(word, k % 32)


# ------------------------------------------------------------------ speed (04 §5.4)
def get_stat(st: S.State, dex, ref, stat_idx):
    """getStat with stages and (core) paralysis; Modify* hook chains join in M3. PS: sim/pokemon.ts:617-660.

    Status handlers only run for active targets (runEvent gathers only field handlers for an inactive target,
    sim/battle.ts:1067)."""
    m = st.mons
    stats = mget(m.stats, ref)
    boosts = mget(m.boosts, ref)
    active = mget(m.is_active, ref)
    par = active & (mget(m.status, ref) == ids.COND_PAR)
    return ST.get_stat_hookless(stats, boosts, stat_idx, par)


def action_speed(st: S.State, dex, ref):
    """[C] getActionSpeed: getStat('spe'), negated under Trick Room. PS: data/mods/champions/scripts.ts:46-54."""
    return ST.action_speed(get_stat(st, dex, ref, 4), trick_room(st))


def update_speed(st: S.State, dex, on=True) -> S.State:
    """PS: sim/battle.ts:392-396 updateSpeed: every non-fainted active gets speed = getActionSpeed()."""
    m = st.mons
    sp = m.speed
    for r in range(8):
        live = mget(m.is_active, r) & ~mget(m.fainted, r)
        sp = mset(sp, r, action_speed(st, dex, r), jnp.asarray(on) & live)
    return st.replace(mons=m.replace(speed=sp))


def update_speed_one(st: S.State, dex, ref, on=True) -> S.State:
    """PS: sim/pokemon.ts:577-579 pokemon.updateSpeed()."""
    m = st.mons
    return st.replace(mons=m.replace(speed=mset(m.speed, ref, action_speed(st, dex, ref), on)))


def live(st: S.State, ref):
    """isActive and not fainted (getAllActive membership)."""
    m = st.mons
    return (jnp.asarray(ref) >= 0) & mget(m.is_active, ref) & ~mget(m.fainted, ref)
