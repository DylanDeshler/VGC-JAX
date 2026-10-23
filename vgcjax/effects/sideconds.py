"""Side / slot conditions, pseudo-weathers, weather and terrain: add / remove primitives (04 §9).

PS: sim/side.ts addSideCondition / removeSideCondition / addSlotCondition / removeSlotCondition, sim/field.ts
setWeather / clearWeather / setTerrain / clearTerrain / addPseudoWeather / removePseudoWeather. End callbacks are
SideEnd / FieldEnd (hook End with the Side / Field prefix) or End for slot conditions.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import HookCtx

I16, I32, U32 = jnp.int16, jnp.int32, jnp.uint32


def remove_side_condition(st, dex, rng, side, c, on):
    """PS: sim/side.ts removeSideCondition: SideEnd, delete."""
    on = jnp.asarray(on) & CO.has_side(st, side, c)
    hc = HookCtx.make(-1, -1, -1).replace(holder=8 + jnp.asarray(side, I32))
    hc, st, rng = CO.single_event(ids.HOOK_END, ids.COND_BASE + jnp.asarray(c, I32) - 1, st, dex, rng, hc, on,
                                  prefix=ids.PFX_SIDE)
    st = CO.set_side(st, side, c, False, jnp.zeros(4, I16), on)
    return st, rng


def remove_slot_condition(st, dex, rng, side, pos, c, on):
    """PS: sim/side.ts removeSlotCondition: End, delete."""
    sd = st.sides
    k = jnp.asarray(c, I32) - ids.SLOT_BASE
    side = jnp.asarray(side, I32)
    pos = jnp.asarray(pos, I32)
    bits = sd.slot_cond_bits[side, pos]
    on = jnp.asarray(on) & (((bits >> k.astype(U32)) & 1) == 1)
    hc = HookCtx.make(-1, -1, -1)
    hc, st, rng = CO.single_event(ids.HOOK_END, ids.COND_BASE + jnp.asarray(c, I32) - 1, st, dex, rng, hc, on)
    sd = st.sides
    nb = sd.slot_cond_bits[side, pos] & ~(U32(1) << k.astype(U32))
    sd = sd.replace(slot_cond_bits=sd.slot_cond_bits.at[side, pos].set(jnp.where(on, nb, sd.slot_cond_bits[side, pos])),
                    slot_cond_state=sd.slot_cond_state.at[side, pos, k].set(
                        jnp.where(on, jnp.zeros(6, I16), sd.slot_cond_state[side, pos, k])))
    return st.replace(sides=sd), rng


def remove_pseudo_weather(st, dex, rng, c, on):
    """PS: sim/field.ts removePseudoWeather: FieldEnd, delete."""
    on = jnp.asarray(on) & CO.has_pseudo(st, c)
    hc = HookCtx.make(-1, -1, -1).replace(holder=10)
    hc, st, rng = CO.single_event(ids.HOOK_END, ids.COND_BASE + jnp.asarray(c, I32) - 1, st, dex, rng, hc, on,
                                  prefix=ids.PFX_FIELD)
    return CO.set_pseudo(st, c, False, jnp.zeros(3, I16), on), rng


def clear_weather(st, dex, rng, on):
    """PS: sim/field.ts clearWeather: FieldEnd of the weather, weather = '' (WeatherChange event: M3 weather)."""
    f = st.field
    on = jnp.asarray(on) & (f.weather > 0)
    hc = HookCtx.make(-1, -1, -1).replace(holder=10)
    hc, st, rng = CO.single_event(ids.HOOK_END, ids.COND_BASE + f.weather.astype(I32) - 1, st, dex, rng, hc, on,
                                  prefix=ids.PFX_FIELD)
    f = st.field
    return st.replace(field=f.replace(weather=jnp.where(on, 0, f.weather).astype(f.weather.dtype),
                                      weather_state=jnp.where(on, jnp.zeros(3, I16), f.weather_state))), rng


def clear_terrain(st, dex, rng, on):
    """PS: sim/field.ts clearTerrain: FieldEnd of the terrain, terrain = '' (TerrainChange: M3 terrain)."""
    f = st.field
    on = jnp.asarray(on) & (f.terrain > 0)
    hc = HookCtx.make(-1, -1, -1).replace(holder=10)
    hc, st, rng = CO.single_event(ids.HOOK_END, ids.COND_BASE + f.terrain.astype(I32) - 1, st, dex, rng, hc, on,
                                  prefix=ids.PFX_FIELD)
    f = st.field
    return st.replace(field=f.replace(terrain=jnp.where(on, 0, f.terrain).astype(f.terrain.dtype),
                                      terrain_state=jnp.where(on, jnp.zeros(3, I16), f.terrain_state))), rng


def add_side_condition(st, dex, rng, side, c, on, source=-1, source_effect=-1):
    """PS: sim/side.ts:413-443 addSideCondition: SideRestart if present (false without a handler), else new state
    (duration, durationCallback), SideStart (falsy cancels), SideConditionStart (M4). Returns (st, rng, result)."""
    from vgcjax.battle import results as RS
    on = jnp.asarray(on)
    side = jnp.asarray(side, I32)
    c = jnp.asarray(c, I32)
    e = ids.COND_BASE + c - 1
    present = CO.has_side(st, side, c)
    hc = HookCtx.make(-1, source, source_effect).replace(holder=8 + side)
    restart = on & present
    hc, st, rng = CO.single_event(ids.HOOK_RESTART, e, st, dex, rng, hc, restart, prefix=ids.PFX_SIDE)
    has_restart = CO.handler_present(dex, e, ids.HOOK_RESTART, ids.PFX_SIDE)
    r_restart = RS.where(has_restart, (hc.ret_t, hc.ret_v), RS.R(RS.BOOL, 0))
    add = on & ~present
    dur = dex.cond_duration[c].astype(I32)
    hc2 = HookCtx.make(-1, source, source_effect).replace(holder=8 + side)
    hc2, st, rng = CO.single_event(CO.DURATION_CALLBACK, e, st, dex, rng, hc2, add)
    dur = jnp.where(hc2.ret_t == RS.NUM, hc2.ret_v, dur)
    st = CO.set_side(st, side, c, True, jnp.zeros(4, I16).at[0].set(dur.astype(I16)), add)
    st = CO.next_effect_order(st, add)
    hc3 = HookCtx.make(-1, source, source_effect).replace(holder=8 + side)
    hc3, st, rng = CO.single_event(ids.HOOK_START, e, st, dex, rng, hc3, add, prefix=ids.PFX_SIDE)
    started = (hc3.ret_t == RS.UNDEF) | RS.truthy((hc3.ret_t, hc3.ret_v))
    st = CO.set_side(st, side, c, False, jnp.zeros(4, I16), add & ~started)
    res = RS.where(restart, r_restart, RS.where(add & started, RS.R(RS.BOOL, 1), RS.R(RS.BOOL, 0)))
    return st, rng, res


def add_pseudo_weather(st, dex, rng, c, on, source=-1, source_effect=-1):
    """PS: sim/field.ts:186-216 addPseudoWeather: FieldRestart if present (false without a handler), else new state
    (duration, durationCallback; no target, so no effectOrder), FieldStart (falsy cancels), PseudoWeatherChange (M4).
    Returns (st, rng, result)."""
    from vgcjax.battle import results as RS
    on = jnp.asarray(on)
    c = jnp.asarray(c, I32)
    e = ids.COND_BASE + c - 1
    present = CO.has_pseudo(st, c)
    restart = on & present
    hc = HookCtx.make(-1, source, source_effect).replace(holder=10)
    hc, st, rng = CO.single_event(ids.HOOK_RESTART, e, st, dex, rng, hc, restart, prefix=ids.PFX_FIELD)
    has_restart = CO.handler_present(dex, e, ids.HOOK_RESTART, ids.PFX_FIELD)
    r_restart = RS.where(has_restart, CO.event_result(hc), RS.R(RS.BOOL, 0))
    add = on & ~present
    dur = dex.cond_duration[c].astype(I32)
    hc2 = HookCtx.make(-1, source, source_effect).replace(holder=10)
    hc2, st, rng = CO.single_event(CO.DURATION_CALLBACK, e, st, dex, rng, hc2, add)
    dur = jnp.where(hc2.ret_t == RS.NUM, hc2.ret_v, dur)
    st = CO.set_pseudo(st, c, True, jnp.zeros(3, I16).at[0].set(dur.astype(I16)), add)
    hc3 = HookCtx.make(-1, source, source_effect).replace(holder=10)
    hc3, st, rng = CO.single_event(ids.HOOK_START, e, st, dex, rng, hc3, add, prefix=ids.PFX_FIELD)
    started = (hc3.ret_t == RS.UNDEF) | RS.truthy((hc3.ret_t, hc3.ret_v))
    st = CO.set_pseudo(st, c, False, jnp.zeros(3, I16), add & ~started)
    res = RS.where(restart, r_restart, RS.where(add & started, RS.R(RS.BOOL, 1), RS.R(RS.BOOL, 0)))
    return st, rng, res
