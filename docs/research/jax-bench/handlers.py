"""Synthetic effect handlers that look like PS ability/item/move callbacks.

Each handler maps an int32 'ctx' tuple (bp, atk, dfn, hp, maxhp, weather, mtype, flags, mult)
to a new ctx. Handlers are built from 6 templates with per-handler constants, so XLA cannot
CSE them away, and each costs ~10-40 integer ops (comparable to PS onBasePower / onModifyAtk /
onResidual callbacks such as Technician, Sand Force, Tough Claws, Blaze, Leftovers).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

FIELDS = ("bp", "atk", "dfn", "hp", "maxhp", "weather", "mtype", "flags", "mult")
NF = len(FIELDS)
N_TEMPLATES = 6


def modify(x, m):
    # PS: tr((tr(value * modifier) + 2048 - 1) / 4096)
    return (x * m + 2047) >> 12


def chain(a, b):
    # PS chainModify: ((a * b) + 2048) >> 12
    return (a * b + 2048) >> 12


def template_eval(t, c, ctx):
    """Evaluate template t (python int) with constants c=(c1..c4) (python ints or traced int32)."""
    bp, atk, dfn, hp, maxhp, weather, mtype, flags, mult = ctx
    c1, c2, c3, c4 = c
    if t == 0:  # Technician-like: low-BP boost
        bp = jnp.where(bp <= c1, modify(bp, c2), bp)
    elif t == 1:  # Sand Force-like: weather + type gated multiplier
        cond = (weather == (c1 % 5)) & ((mtype == (c3 % 18)) | (mtype == (c4 % 18)))
        mult = jnp.where(cond, chain(mult, c2), mult)
    elif t == 2:  # Tough Claws / Iron Fist-like: flag gated
        cond = (flags & ((c1 % 255) + 1)) != 0
        mult = jnp.where(cond, chain(mult, c2), mult)
        bp = jnp.where(cond & (bp > c3 % 120), bp + 1, bp)
    elif t == 3:  # Blaze-like pinch: hp threshold + type
        cond = (hp * 3 <= maxhp) & (mtype == (c1 % 18))
        atk = jnp.where(cond, modify(atk, c2), atk)
        dfn = jnp.where(cond & (flags > c3 % 255), modify(dfn, c4), dfn)
    elif t == 4:  # stat-ratio BP (Gyro Ball / Electro Ball-like)
        bp = jnp.clip((atk * (c1 % 25 + 1)) // jnp.maximum(dfn, 1) + c2 % 40, 1, 250)
        mult = jnp.where(bp > c3 % 150, chain(mult, c4), mult)
    else:  # residual-like (Leftovers / poison / Black Sludge)
        delta = jnp.maximum(maxhp * (c1 % 4 + 1) // 16, 1)
        heal = (mtype == (c2 % 18))
        hp = jnp.where(heal, jnp.minimum(hp + delta, maxhp), jnp.maximum(hp - delta, 0))
        flags = jnp.where(hp == 0, flags | 128, flags)
    return (bp, atk, dfn, hp, maxhp, weather, mtype, flags, mult)


def make_consts(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) % N_TEMPLATES
    c = np.stack([
        rng.integers(30, 90, n),      # c1
        rng.integers(4096, 8192, n),  # c2 (multiplier on 4096 base)
        rng.integers(0, 255, n),      # c3
        rng.integers(2048, 6144, n),  # c4
    ], axis=1)
    return t.astype(np.int32), c.astype(np.int32)


def make_branches(n, seed=0):
    t, c = make_consts(n, seed)
    branches = []
    for k in range(n):
        tk, ck = int(t[k]), tuple(int(v) for v in c[k])

        def h(ctx, tk=tk, ck=ck):
            return template_eval(tk, ck, ctx)
        branches.append(h)
    return branches


def identity(ctx):
    return ctx


def random_ctx(key, B):
    ks = jax.random.split(key, NF)
    ri = lambda k, lo, hi: jax.random.randint(k, (B,), lo, hi, dtype=jnp.int32)
    maxhp = ri(ks[4], 100, 400)
    return (
        ri(ks[0], 20, 150),            # bp
        ri(ks[1], 50, 400),            # atk
        ri(ks[2], 50, 400),            # dfn
        jnp.maximum(1, (maxhp * jax.random.uniform(ks[3], (B,))).astype(jnp.int32)),
        maxhp,
        ri(ks[5], 0, 5),
        ri(ks[6], 0, 18),
        ri(ks[7], 0, 256),
        jnp.full((B,), 4096, jnp.int32),
    )
