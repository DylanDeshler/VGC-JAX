"""PS fixed-point arithmetic, bit-exact in int32 (ADR-011).

PS computes in float64 and truncates with `num >>> 0` (Dex.trunc). For the non-negative, < 2**31 operands that
mechanics produce, that equals integer floor division; `chain_modify` reproduces JS's int32 `>>` (wrapping) exactly
because jnp int32 multiplication wraps identically.
All functions broadcast over arrays. Modifiers are 4096-scale integers.
Domain: value * m4096 < 2**31 (e.g. value < 65536 with modifiers <= x4); PS's float64 differs beyond that.
"""
from __future__ import annotations

import jax.numpy as jnp

I32 = jnp.int32


def trunc(x):
    """PS: sim/dex.ts:365-368 `trunc(num)` for integer inputs = identity on non-negative int32."""
    return jnp.asarray(x, I32)


def trunc16(x):
    """PS: sim/dex.ts:365-368 `trunc(num, 16)` = (num >>> 0) % 65536 (non-negative inputs)."""
    return jnp.asarray(x, I32) & 0xFFFF


def mod4096(num, den=1):
    """4096-scale modifier of the fraction num/den. PS: sim/battle.ts modify/chainModify `trunc(num * 4096 / den)`."""
    return (jnp.asarray(num, I32) * 4096) // jnp.asarray(den, I32)


def modify_raw(value, m4096):
    """Apply a 4096-scale modifier. PS: sim/battle.ts:2348-2359 `tr((tr(value * modifier) + 2048 - 1) / 4096)`."""
    return (jnp.asarray(value, I32) * jnp.asarray(m4096, I32) + 2047) // 4096


def modify(value, num, den=1):
    """PS: sim/battle.ts:2348-2359 `modify(value, numerator, denominator)`."""
    return modify_raw(value, mod4096(num, den))


def chain(m4096_prev, m4096_next):
    """Combine two 4096-scale modifiers. PS: sim/battle.ts:2337-2346 `((previousMod * nextMod + 2048) >> 12)`."""
    return (jnp.asarray(m4096_prev, I32) * jnp.asarray(m4096_next, I32) + 2048) >> 12


def chain_modify(m4096_prev, num, den=1):
    """PS: sim/battle.ts:2337-2346 `chainModify(numerator, denominator)` on the event modifier."""
    return chain(m4096_prev, mod4096(num, den))


def final_modify(value, m4096):
    """PS: sim/battle.ts:2360-2364 `finalModify(relayVar)` = modify(relayVar, event.modifier)."""
    return modify_raw(value, m4096)


def js_round_frac(x, num, den):
    """JS Math.round(x * num / den) for non-negative ints (round half up): (2*x*num + den) // (2*den) (04 §7.8)."""
    x = jnp.asarray(x, I32)
    return (2 * x * num + den) // (2 * den)
