"""Conditions framework (04 §9, M3.3): status, volatiles, side / slot conditions, pseudo-weathers, weather, terrain.

Storage (03 §4): every condition kind keeps a small i16 state vector per instance. Slot 0 is always `duration`
(0 = none); the last slot is a presence bitmask for the other fields (bit k = slot k set), because PS effect states
only contain the keys a handler assigned and the projection (oracle/lib/project.js effState) compares key sets.

  status      mons.status / status_state[4]          (0 duration, 1..2 fields, 3 presence)
  volatile    mons.vol_bits / vol_state[NV, 6]        (0 duration, 1..4 fields, 5 presence)
  side        sides.side_cond_bits / state[NSC, 4]    (0 duration, 1..2 fields, 3 presence)
  slot        sides.slot_cond_bits / state[NSL, 6]    (0 duration, 1..4 fields, 5 presence)
  pseudo      field.pw_bits / pw_state[NPW, 3]        (0 duration, 1 field, 2 presence)
  weather     field.weather / weather_state[3]        (0 duration, 1 field, 2 presence)
  terrain     field.terrain / terrain_state[3]        (same)

Each condition's named fields are declared once with `fields(cond, ...)`; the harness projection
(harness/project.py) reads the same table. Condition callbacks (Start, Restart, End, Residual, durationCallback,
...) are bespoke handlers registered with `effects.hooks.handler` and run through `single_event`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import results as RS
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset
from vgcjax.data import ids
from vgcjax.effects import hooks as H

I16, I32, U32 = jnp.int16, jnp.int32, jnp.uint32
DURATION_CALLBACK = -1          # pseudo hook id for durationCallback handlers (not a PS event)

# ------------------------------------------------------------------ kinds and state layout
STATUS, VOLATILE, SIDE, SLOT, PSEUDO, WEATHER, TERRAIN = range(7)
_LAYOUT = {STATUS: (4, 3), VOLATILE: (6, 5), SIDE: (4, 3), SLOT: (6, 5), PSEUDO: (3, 2), WEATHER: (3, 2),
           TERRAIN: (3, 2)}         # (state length, presence slot)
WEATHERS = {"sunnyday", "raindance", "sandstorm", "snowscape", "desolateland", "primordialsea", "deltastream"}
TERRAINS = {"electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"}


def kind_of(name: str) -> int:
    i = ids.CONDITIONS.index(name)
    if ids.VOL_BASE <= i < ids.VOL_END:
        return VOLATILE
    if ids.SIDE_BASE <= i < ids.SIDE_END:
        return SIDE
    if ids.SLOT_BASE <= i < ids.SLOT_END:
        return SLOT
    if ids.PW_BASE <= i < ids.PW_END:
        return PSEUDO
    if name in WEATHERS:
        return WEATHER
    if name in TERRAINS:
        return TERRAIN
    return STATUS


# field schema: condition name -> [(PS key, slot, kind)]; kind: int | bool | move | cond | mon (ref) | loc
FIELDS: dict[str, list[tuple[str, int, str]]] = {}


def fields(cond: str, *specs: tuple[str, str]):
    """Declare the named state keys of `cond` in slot order (1, 2, ...): fields("slp", ("startTime", "int"), ...)."""
    FIELDS[cond] = [(k, j + 1, kind) for j, (k, kind) in enumerate(specs)]
    n, pres = _LAYOUT[kind_of(cond)]
    assert len(specs) < pres, (cond, specs)


def slot_of(cond: str, key: str) -> int:
    for k, j, _ in FIELDS.get(cond, []):
        if k == key:
            return j
    raise KeyError((cond, key))


def cid(name: str) -> int:
    return ids.CONDITIONS.index(name)


def eff(name: str) -> int:
    """EffectId of a condition (hook registry keys)."""
    return ids.COND_BASE + cid(name) - 1


def next_effect_order(st, on):
    """PS: sim/battle.ts:3380-3390 initEffectState: a new effect state with an id and a target (a Side / Field, or an
    active Pokemon) takes battle.effectOrder++."""
    meta = st.meta
    return st.replace(meta=meta.replace(effect_seq=meta.effect_seq + jnp.asarray(on).astype(meta.effect_seq.dtype)))


# ------------------------------------------------------------------ volatile storage
def has_volatile(st, ref, c):
    k = jnp.asarray(c, I32) - ids.VOL_BASE
    word = mget(st.mons.vol_bits, ref)[jnp.clip(k // 32, 0, S.WV - 1)]
    return ((word >> (k % 32).astype(U32)) & 1) == 1


def vol_state(st, ref, c):
    return mget(st.mons.vol_state, ref)[jnp.asarray(c, I32) - ids.VOL_BASE]


def _set_vol_bit(st, ref, c, val, on):
    m = st.mons
    k = jnp.asarray(c, I32) - ids.VOL_BASE
    words = mget(m.vol_bits, ref)
    w = jnp.clip(k // 32, 0, S.WV - 1)
    bit = U32(1) << (k % 32).astype(U32)
    new = words.at[w].set(jnp.where(val, words[w] | bit, words[w] & ~bit))
    return st.replace(mons=m.replace(vol_bits=mset(m.vol_bits, ref, new, on)))


def set_vol_state(st, ref, c, state6, on):
    m = st.mons
    k = jnp.asarray(c, I32) - ids.VOL_BASE
    cur = mget(m.vol_state, ref)
    new = cur.at[k].set(jnp.asarray(state6, I16))
    return st.replace(mons=m.replace(vol_state=mset(m.vol_state, ref, new, on)))


def set_vol_field(st, ref, c, slot, value, on):
    """effectState[key] = value for a declared key slot (sets its presence bit)."""
    s = vol_state(st, ref, c)
    s = s.at[slot].set(jnp.asarray(value, I16)).at[5].set(s[5] | (1 << slot))
    return set_vol_state(st, ref, c, s, on)


# ------------------------------------------------------------------ side / field storage
def has_side(st, side, c):
    k = jnp.asarray(c, I32) - ids.SIDE_BASE
    return ((st.sides.side_cond_bits[side] >> k.astype(U32)) & 1) == 1


def side_state(st, side, c):
    return st.sides.side_cond_state[side, jnp.asarray(c, I32) - ids.SIDE_BASE]


def set_side(st, side, c, present, state4, on):
    sd = st.sides
    side = jnp.asarray(side, I32)
    k = jnp.asarray(c, I32) - ids.SIDE_BASE
    on = jnp.asarray(on)
    bits = sd.side_cond_bits[side]
    bit = U32(1) << k.astype(U32)
    nb = jnp.where(present, bits | bit, bits & ~bit)
    sd = sd.replace(side_cond_bits=sd.side_cond_bits.at[side].set(jnp.where(on, nb, bits)),
                    side_cond_state=sd.side_cond_state.at[side, k].set(
                        jnp.where(on, jnp.asarray(state4, I16), sd.side_cond_state[side, k])))
    return st.replace(sides=sd)


def has_pseudo(st, c):
    k = jnp.asarray(c, I32) - ids.PW_BASE
    return ((st.field.pw_bits >> k.astype(U32)) & 1) == 1


def set_pseudo(st, c, present, state3, on):
    f = st.field
    k = jnp.asarray(c, I32) - ids.PW_BASE
    on = jnp.asarray(on)
    bit = U32(1) << k.astype(U32)
    nb = jnp.where(present, f.pw_bits | bit, f.pw_bits & ~bit)
    f = f.replace(pw_bits=jnp.where(on, nb, f.pw_bits),
                  pw_state=f.pw_state.at[k].set(jnp.where(on, jnp.asarray(state3, I16), f.pw_state[k])))
    return st.replace(field=f)


# ------------------------------------------------------------------ singleEvent dispatch
def single_event(hook: int, effect, st, dex, rng, hc: H.HookCtx, on, prefix=ids.PFX_ON):
    """PS singleEvent(eventid, effect, state, target, source, sourceEffect) for a traced `effect`: runs the bespoke
    handler registered for (hook, prefix, effect) if any. Returns (hc, st, rng); hc.ret_* holds the handler's return
    (tag UNDEF when none ran)."""
    on = jnp.asarray(on)
    static_prefix = isinstance(prefix, int)
    hs = H.registered(hook, prefix if static_prefix else None)
    hc = hc.replace(cur_effect=jnp.asarray(effect, I32), ret_t=jnp.asarray(RS.UNDEF, I32), ret_v=jnp.zeros((), I32))
    if not hs:
        return hc, st, rng
    idx = jnp.zeros((), I32)
    for i, h in enumerate(hs):
        hit = jnp.asarray(effect, I32) == h.effect
        if not static_prefix:
            hit = hit & (jnp.asarray(prefix, I32) == h.prefix)
        idx = jnp.where(hit, i + 1, idx)
    idx = jnp.where(on, idx, 0)
    branches = [lambda hc_, st_, rng_: (hc_, st_, rng_)] + [
        (lambda hc_, st_, rng_, f=h.fn: f(hc_, st_, dex, rng_)) for h in hs]
    hc, st, rng = jax.lax.switch(idx, branches, hc, st, rng)
    return jax.lax.optimization_barrier((hc, st, rng))


def handler_present(dex, effect, hook: int, prefix: int = ids.PFX_ON):
    """Does `effect` define the callback (PS getCallback !== undefined)? From the generated handler mask."""
    return ((dex.handler_mask[jnp.maximum(jnp.asarray(effect, I32), 0), hook].astype(I32) >> prefix) & 1) == 1


# ------------------------------------------------------------------ volatiles: add / remove (PS: sim/pokemon.ts)
def add_volatile(st, dex, rng, ref, c, on, source=-1, source_effect=-1):
    """PS: sim/pokemon.ts:1990-2054 addVolatile (no linkedStatus): hp gate, Restart if present, status immunity
    (runStatusImmunity: none for core volatiles), TryAddVolatile (M3: hooks), duration + durationCallback, Start
    (falsy cancels). Returns (st, rng, result (tag, val))."""
    on = jnp.asarray(on)
    ref = jnp.asarray(ref, I32)
    c = jnp.asarray(c, I32)
    e = ids.COND_BASE + c - 1
    m = st.mons
    alive = mget(m.hp, ref) > 0          # affectsFainted: false for every v0 volatile
    present = has_volatile(st, ref, c)
    src = jnp.where(jnp.asarray(source, I32) >= 0, source, ref)
    hc = H.HookCtx.make(ref, src, source_effect)
    # restart
    restart = on & alive & present
    hc, st, rng = single_event(ids.HOOK_RESTART, e, st, dex, rng, hc, restart)
    r_restart = jnp.where(handler_present(dex, e, ids.HOOK_RESTART), hc.ret_t, RS.BOOL), \
        jnp.where(handler_present(dex, e, ids.HOOK_RESTART), hc.ret_v, 0)
    add = on & alive & ~present
    dur = dex.cond_duration[c].astype(I32)
    hc2 = H.HookCtx.make(ref, src, source_effect)
    hc2, st, rng = single_event(DURATION_CALLBACK, e, st, dex, rng, hc2, add)
    dur = jnp.where(hc2.ret_t == RS.NUM, hc2.ret_v, dur)
    st = _set_vol_bit(st, ref, c, True, add)
    st = set_vol_state(st, ref, c, jnp.zeros(6, I16).at[0].set(dur.astype(I16)), add)
    st = next_effect_order(st, add & mget(st.mons.is_active, ref))
    hc3 = H.HookCtx.make(ref, src, source_effect)
    hc3, st, rng = single_event(ids.HOOK_START, e, st, dex, rng, hc3, add)
    started = (hc3.ret_t == RS.UNDEF) | RS.truthy((hc3.ret_t, hc3.ret_v))
    st = _set_vol_bit(st, ref, c, False, add & ~started)
    r_add = RS.where(started, RS.R(RS.BOOL, 1), (hc3.ret_t, hc3.ret_v))
    res = RS.where(restart, r_restart, RS.where(add, r_add, RS.R(RS.BOOL, 0)))
    return st, rng, res


def remove_volatile(st, dex, rng, ref, c, on):
    """PS: sim/pokemon.ts:2056-2067 removeVolatile: hp gate, End, delete. Returns (st, rng, removed)."""
    on = jnp.asarray(on)
    c = jnp.asarray(c, I32)
    go = on & (mget(st.mons.hp, ref) > 0) & has_volatile(st, ref, c)
    hc = H.HookCtx.make(ref, -1, -1)
    hc, st, rng = single_event(ids.HOOK_END, ids.COND_BASE + c - 1, st, dex, rng, hc, go)
    st = _set_vol_bit(st, ref, c, False, go)
    st = set_vol_state(st, ref, c, jnp.zeros(6, I16), go)
    return st, rng, go
