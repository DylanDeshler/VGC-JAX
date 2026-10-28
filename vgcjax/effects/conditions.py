"""Conditions framework (04 §9, M3.3): status, volatiles, side / slot conditions, pseudo-weathers, weather, terrain.

Storage (03 §4): every condition kind keeps a small i16 state vector per instance. Slot 0 is always `duration`
(0 = none); the last slot is a presence bitmask for the other fields (bit k = slot k set), because PS effect states
only contain the keys a handler assigned and the projection (oracle/lib/project.js effState) compares key sets.

  status      mons.status / status_state[4]          (0 duration, 1..2 fields, 3 presence)
  volatile    mons.vol_bits / vol_dur[NV] + vol_fx     (6-vector view: 0 duration, 1..3 fields, 5 presence; only
                                                       the state.VOL_STATEFUL volatiles store fields, ADR-107)
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
    if kind_of(cond) == VOLATILE:                       # compact layout: fields only for the declared volatiles
        assert cond in S.VOL_STATEFUL and len(specs) <= 3, (cond, "add it to battle/state.py VOL_STATEFUL")


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
    """Volatile c on ref (scalar or vector refs; c broadcasts)."""
    k = jnp.asarray(c, I32) - ids.VOL_BASE
    words = mget(st.mons.vol_bits, ref)                                 # [..., WV]
    w = jnp.broadcast_to(jnp.clip(k // 32, 0, S.WV - 1), words.shape[:-1])
    word = jnp.take_along_axis(words, w[..., None], axis=-1)[..., 0]
    return ((word >> (k % 32).astype(U32)) & 1) == 1


def vol_state(st, ref, c):
    """The volatile's 6-vector state [duration, f1, f2, f3, 0, presence] (compact storage: vol_dur + vol_fx)."""
    m = st.mons
    k = jnp.asarray(c, I32) - ids.VOL_BASE
    dur = mget(m.vol_dur, ref)[k]
    fi = jnp.asarray(S.VOL_FIDX)[jnp.clip(k, 0, S.NV - 1)]
    fx = jnp.where(fi >= 0, mget(m.vol_fx, ref)[jnp.clip(fi, 0, S.NVF - 1)], 0)
    z = jnp.zeros((), I16)
    return jnp.stack([dur, fx[0], fx[1], fx[2], z, fx[3]]).astype(I16)


def _set_vol_bit(st, ref, c, val, on):
    m = st.mons
    k = jnp.asarray(c, I32) - ids.VOL_BASE
    words = mget(m.vol_bits, ref)
    w = jnp.clip(k // 32, 0, S.WV - 1)
    bit = U32(1) << (k % 32).astype(U32)
    new = words.at[w].set(jnp.where(val, words[w] | bit, words[w] & ~bit))
    return st.replace(mons=m.replace(vol_bits=mset(m.vol_bits, ref, new, on)))


def set_vol_state(st, ref, c, state6, on):
    """Write a volatile's 6-vector state (slot 4 is never used: at most 3 fields)."""
    m = st.mons
    k = jnp.asarray(c, I32) - ids.VOL_BASE
    s = jnp.asarray(state6, I16)
    d = mget(m.vol_dur, ref).at[k].set(s[0])
    fi = jnp.asarray(S.VOL_FIDX)[jnp.clip(k, 0, S.NV - 1)]
    fx = mget(m.vol_fx, ref).at[jnp.clip(fi, 0, S.NVF - 1)].set(jnp.stack([s[1], s[2], s[3], s[5]]))
    return st.replace(mons=m.replace(vol_dur=mset(m.vol_dur, ref, d, on),
                                     vol_fx=mset(m.vol_fx, ref, fx, jnp.asarray(on) & (fi >= 0))))


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
def single_event(hook: int, effect, st, dex, rng, hc: H.HookCtx, on, prefix=ids.PFX_ON, effect_range=None,
                 exclude_range=None, only=None):
    """PS singleEvent(eventid, effect, state, target, source, sourceEffect) for a traced `effect`: runs the bespoke
    handler registered for (hook, prefix, effect) if any. Returns (hc, st, rng); hc.ret_* holds the handler's return
    (tag UNDEF when none ran)."""
    on = jnp.asarray(on)
    static_prefix = isinstance(prefix, int)
    hs = H.registered(hook, prefix if static_prefix else None)
    if effect_range is not None:                         # static restriction (e.g. volatiles only)
        hs = [h for h in hs if effect_range[0] <= h.effect < effect_range[1]]
    if exclude_range is not None:                        # handled undispatched by the caller
        hs = [h for h in hs if not exclude_range[0] <= h.effect < exclude_range[1]]
    if only is not None:                                 # the effects this call site can name (static set)
        hs = [h for h in hs if h.effect in only]
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


def registered_for(hook: int, effect, prefix: int = ids.PFX_ON, effect_range=None):
    """Does a registered handler exist for (hook, prefix, traced effect)? (PS `if (moveData.onHit)` style guards.)"""
    hs = [h for h in H.registered(hook, prefix) if effect_range is None or effect_range[0] <= h.effect < effect_range[1]]
    e = jnp.asarray(effect, I32)
    out = jnp.zeros((), bool)
    for h in hs:
        out = out | (e == h.effect)
    return out


def event_result(hc):
    """singleEvent's return: the handler's value, or the relay (true) when it returned undefined / none ran."""
    return RS.where(hc.ret_t == RS.UNDEF, RS.R(RS.BOOL, 1), (hc.ret_t, hc.ret_v))


MOVE_RANGE = (ids.MOVE_BASE, ids.MOVE_BASE + len(ids.MOVES) - 1)


def handler_present(dex, effect, hook: int, prefix: int = ids.PFX_ON):
    """Does `effect` define the callback (PS getCallback !== undefined)? From the generated handler mask."""
    return ((dex.handler_mask[jnp.maximum(jnp.asarray(effect, I32), 0), hook].astype(I32) >> prefix) & 1) == 1


# ------------------------------------------------------------------ volatiles: add / remove (PS: sim/pokemon.ts)
def vol_candidates(column: str) -> frozenset:
    """EffectIds of the volatiles a move-data column (move_volatile, move_self_volatile, move_sec_volatile) can name:
    the static dispatch set of an add_volatile call site fed by that column (ADR-106 profiling: a full volatile
    dispatch copies every Start / Restart handler into each call site)."""
    if column not in _VOL_CANDS:
        import numpy as np
        from vgcjax.core.dex import load_tables_np
        vals = np.unique(np.asarray(load_tables_np()[column]))
        _VOL_CANDS[column] = frozenset(int(ids.COND_BASE + v - 1) for v in vals if v > 0)
    return _VOL_CANDS[column]


_VOL_CANDS: dict = {}


def add_volatile(st, dex, rng, ref, c, on, source=-1, source_effect=-1, cands=None):
    """PS: sim/pokemon.ts:1990-2054 addVolatile (no linkedStatus): hp gate, Restart if present, status immunity
    (runStatusImmunity: none for core volatiles), TryAddVolatile (terrains), duration + durationCallback, Start
    (falsy cancels). Returns (st, rng, result (tag, val)). Handlers dispatched: volatile conditions only."""
    on = jnp.asarray(on)
    ref = jnp.asarray(ref, I32)
    # static dispatch set: a python-int condition names only itself; dynamic callers pass `cands`
    only = frozenset({ids.COND_BASE + c - 1}) if isinstance(c, int) else cands
    c = jnp.asarray(c, I32)
    e = ids.COND_BASE + c - 1
    m = st.mons
    alive = mget(m.hp, ref) > 0          # affectsFainted: false for every v0 volatile
    present = has_volatile(st, ref, c)
    src = jnp.where(jnp.asarray(source, I32) >= 0, source, ref)
    hc = H.HookCtx.make(ref, src, source_effect)
    # restart
    restart = on & alive & present
    vr = (ids.COND_BASE + ids.VOL_BASE - 1, ids.COND_BASE + ids.VOL_END - 1)
    hc, st, rng = single_event(ids.HOOK_RESTART, e, st, dex, rng, hc, restart, effect_range=vr, only=only)
    r_restart = jnp.where(handler_present(dex, e, ids.HOOK_RESTART), hc.ret_t, RS.BOOL), \
        jnp.where(handler_present(dex, e, ids.HOOK_RESTART), hc.ret_v, 0)
    add = on & alive & ~present
    # runEvent('TryAddVolatile'): the terrains (abilities: M4); a falsy (null) result returns it
    from vgcjax.effects import terrain as TR
    from vgcjax.battle import events as EV
    from vgcjax.effects import ability_blocks as AXB
    a_b = add & AXB.add_volatile_block(st, ref, c)
    a_spd = mget(st.mons.speed, ref).astype(I32)
    if "yawn" in ids.CONDITIONS:                        # Yawn's ability blocks (messages, ally veils)
        y_b, y_spd, y_blk, y_eff, y_of = AXB.yawn_block(st, ref)
        is_y = c == ids.CONDITIONS.index("yawn")
        a_b = a_b | (add & is_y & y_b)
        a_spd = jnp.where(is_y & y_b, y_spd, a_spd)
    t_b, tmsg = TR.blocks_volatile(st, dex, ref, c, source_effect)
    t_b = add & t_b
    # the field's handlers (subOrder 5, speed 0) run first unless the winning holder's cached speed is > 0
    ab_block = a_b & ((a_spd > 0) | ~t_b)
    if "yawn" in ids.CONDITIONS:
        y_on = ab_block & is_y
        st = EV.emit(st, y_on & ~y_blk, EV.EV_IMMUNE, actor=ref, v1=y_eff)
        st = EV.emit(st, y_on & y_blk, EV.EV_BLOCK, actor=ref, target=y_of, move=y_eff)
    blocked = t_b & ~ab_block
    st = EV.emit(st, blocked & tmsg, EV.EV_ACTIVATE, actor=ref, move=ids.COND_BASE + TR.terrain(st) - 1)
    add = add & ~blocked & ~ab_block
    blocked = blocked | ab_block
    dur = dex.cond_duration[c].astype(I32)
    hc2 = H.HookCtx.make(ref, src, source_effect)
    hc2, st, rng = single_event(DURATION_CALLBACK, e, st, dex, rng, hc2, add, effect_range=vr, only=only)
    dur = jnp.where(hc2.ret_t == RS.NUM, hc2.ret_v, dur)
    st = _set_vol_bit(st, ref, c, True, add)
    st = set_vol_state(st, ref, c, jnp.zeros(6, I16).at[0].set(dur.astype(I16)), add)
    st = next_effect_order(st, add & mget(st.mons.is_active, ref))
    hc3 = H.HookCtx.make(ref, src, source_effect)
    hc3, st, rng = single_event(ids.HOOK_START, e, st, dex, rng, hc3, add, effect_range=vr, only=only)
    started = (hc3.ret_t == RS.UNDEF) | RS.truthy((hc3.ret_t, hc3.ret_v))
    st = _set_vol_bit(st, ref, c, False, add & ~started)
    r_add = RS.where(started, RS.R(RS.BOOL, 1), (hc3.ret_t, hc3.ret_v))
    res = RS.where(restart, r_restart, RS.where(add, r_add, RS.where(blocked, RS.R(RS.NULL), RS.R(RS.BOOL, 0))))
    return st, rng, res


def remove_volatile(st, dex, rng, ref, c, on):
    """PS: sim/pokemon.ts:2056-2067 removeVolatile: hp gate, End, delete. Returns (st, rng, removed)."""
    on = jnp.asarray(on)
    # a static condition dispatches only its own End handler (no other volatile's End is traced)
    vr = (ids.COND_BASE + c - 1, ids.COND_BASE + c) if isinstance(c, int) else \
        (ids.COND_BASE + ids.VOL_BASE - 1, ids.COND_BASE + ids.VOL_END - 1)
    c = jnp.asarray(c, I32)
    go = on & (mget(st.mons.hp, ref) > 0) & has_volatile(st, ref, c)
    hc = H.HookCtx.make(ref, -1, -1)
    hc, st, rng = single_event(ids.HOOK_END, ids.COND_BASE + c - 1, st, dex, rng, hc, go, effect_range=vr)
    st = _set_vol_bit(st, ref, c, False, go)
    st = set_vol_state(st, ref, c, jnp.zeros(6, I16), go)
    return st, rng, go


def is_grounded(st, refs):
    """PS: sim/pokemon.ts isGrounded (negateImmunity false) for refs [...]: Gravity, Ingrain, Smack Down, Iron Ball ->
    true; Flying type -> false; Levitate / Eelevate -> null (falsy); Magnet Rise, Air Balloon -> false
    (ignoringItem / suppressingAbility: M4). Returns bool."""
    refs = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    m = st.mons
    vol = lambda n: has_volatile(st, refs, cid(n)) if n in ids.CONDITIONS else jnp.zeros(refs.shape, bool)  # noqa: E731
    item, ab = mget(m.item, refs), mget(m.ability, refs)
    gravity = has_pseudo(st, cid("gravity")) if "gravity" in ids.CONDITIONS else False
    forced = gravity | vol("ingrain") | vol("smackdown") | (item == getattr(ids, "ITEM_IRONBALL", -1))
    flying = jnp.any(mget(m.types, refs) == ids.TYPE_FLYING, axis=-1)
    lev = (ab == getattr(ids, "ABILITY_LEVITATE", -1)) | (ab == getattr(ids, "ABILITY_EELEVATE", -1))
    air = vol("magnetrise") | vol("telekinesis") | (item == getattr(ids, "ITEM_AIRBALLOON", -1))
    return forced | (~flying & ~lev & ~air)
