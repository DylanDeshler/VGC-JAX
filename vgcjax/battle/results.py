"""PS's dynamically typed move results as (tag, value) int pairs.

PS mixes `undefined`, NOT_FAIL (`''`), `null`, booleans and numbers in hit results and combines them with
`combineResults` (PS: sim/battle-actions.ts:1615-1631). Tags follow its priority order.
"""
from __future__ import annotations

import jax.numpy as jnp

I32 = jnp.int32
UNDEF, NOTFAIL, NULL, BOOL, NUM = 0, 1, 2, 3, 4


def R(tag, val=0):
    return (jnp.asarray(tag, I32), jnp.asarray(val, I32))


def truthy(r):
    t, v = r
    return ((t == BOOL) | (t == NUM)) & (v != 0)


def is_false(r):
    t, v = r
    return (t == BOOL) & (v == 0)


def is_num(r):
    return r[0] == NUM


def hit_or_zero(r):
    """PS `x || x === 0` (kept as a target)."""
    t, v = r
    return truthy(r) | ((t == NUM) & (v == 0))


def combine(left, right):
    """PS combineResults(left, right)."""
    lt, lv = left
    rt, rv = right
    right_is_zero = (rt == NUM) & (rv == 0)
    take_left = (lt > rt) | (truthy(left) & ~truthy(right) & ~right_is_zero)
    both_num = (lt == NUM) & (rt == NUM)
    t = jnp.where(take_left, lt, rt)
    v = jnp.where(take_left, lv, jnp.where(both_num, lv + rv, rv))
    return (t, v)


def where(c, a, b):
    return (jnp.where(c, a[0], b[0]), jnp.where(c, a[1], b[1]))


def null_to_false(r):
    """`didSomething === null ? false : didSomething`."""
    return where(r[0] == NULL, R(BOOL, 0), r)
