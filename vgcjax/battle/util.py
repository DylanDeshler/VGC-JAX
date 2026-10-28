"""Predicated-execution helpers: every branch is computed, then masked (04 §1.2).

Mon arrays are [2, 4, ...]; `mget/mset` address them by ref = side*4 + b (flattened to [8, ...]).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

I32 = jnp.int32


def tree_where(c, a, b):
    """Elementwise select over two pytrees of identical structure (scalar predicate broadcast)."""
    c = jnp.asarray(c)
    return jax.tree_util.tree_map(lambda x, y: jnp.where(c, x, y) if x is not None else None, a, b)


def run_if(pred, fn, carry, *args):
    """carry' = fn(carry, *args) if pred else carry (both computed)."""
    return tree_where(pred, fn(carry, *args), carry)


# ADR-107 experiment: per-mon access as gathers / scatters (default) or as masked selects over the 8 mons
# (VGCJAX_ACCESS=select, read at import). Identical values either way; the cost differs by backend (CPU: selects
# 2.5x slower, they touch all 8 rows).
_SELECT = __import__("os").environ.get("VGCJAX_ACCESS", "") == "select"


def mget(arr, ref):
    """arr[side, b, ...] for ref = side*4 + b (ref < 0 reads ref 0; callers mask)."""
    flat = arr.reshape((8,) + arr.shape[2:])
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    if not _SELECT:
        return flat[r]
    rr = r.reshape(r.shape + (1,) * (flat.ndim - 1))
    out = jnp.broadcast_to(flat[0], r.shape + flat.shape[1:])
    for i in range(1, 8):
        out = jnp.where(rr == i, flat[i], out)
    return out


def mset(arr, ref, val, mask=True):
    """Masked write arr[ref] = val (no-op when mask is False or ref < 0)."""
    ref = jnp.asarray(ref, I32)
    ok = jnp.asarray(mask) & (ref >= 0)
    flat = arr.reshape((8,) + arr.shape[2:])
    r = jnp.clip(ref, 0, 7)
    if _SELECT and ref.ndim == 0 and ok.ndim == 0:
        h = ((jnp.arange(8) == r) & ok).reshape((8,) + (1,) * (flat.ndim - 1))
        v = jnp.broadcast_to(jnp.asarray(val, arr.dtype), flat.shape[1:])
        return jnp.where(h, v[None], flat).reshape(arr.shape)
    new = jnp.where(ok, jnp.asarray(val, arr.dtype), flat[r])
    return flat.at[r].set(new).reshape(arr.shape)


def madd(arr, ref, delta, mask=True):
    return mset(arr, ref, mget(arr, ref) + jnp.asarray(delta, arr.dtype), mask)


def side_of(ref):
    return jnp.asarray(ref, I32) // 4


def b_of(ref):
    return jnp.asarray(ref, I32) % 4


def ref_of(side, b):
    return jnp.asarray(side, I32) * 4 + jnp.asarray(b, I32)


def slot_ref(st, side, slot):
    """ref of the mon occupying (side, slot), -1 if the slot is empty."""
    b = jnp.asarray(st.sides.active[jnp.asarray(side, I32), jnp.asarray(slot, I32)], I32)
    return jnp.where(b >= 0, ref_of(side, b), -1)


def or_flag(st, bit, cond):
    """Set an error bit when cond."""
    meta = st.meta
    return st.replace(meta=meta.replace(error_flags=meta.error_flags | jnp.where(cond, jnp.uint32(bit), jnp.uint32(0))))


def bit_test(word, k):
    return ((jnp.asarray(word, jnp.uint32) >> jnp.asarray(k, jnp.uint32)) & 1) == 1


def mset_many(arr, refs, vals, mask=True):
    """arr[refs[j]] = vals[j] where mask[j] (refs pairwise distinct where masked) as one masked select over the 8 mons:
    elementwise, so it fuses (one batched write instead of len(refs) scatters; 09 J5)."""
    refs = jnp.asarray(refs, I32)
    T = refs.shape[0]
    flat = arr.reshape((8,) + arr.shape[2:])
    tail = flat.ndim - 1
    ok = jnp.broadcast_to(jnp.asarray(mask), (T,)) & (refs >= 0)
    hot = (jnp.clip(refs, 0, 7)[:, None] == jnp.arange(8)[None, :]) & ok[:, None]          # [T, 8]
    vals = jnp.broadcast_to(jnp.asarray(vals, arr.dtype), (T,) + flat.shape[1:])   # scalar or [T, *tail]
    h = hot.reshape(hot.shape + (1,) * tail)
    if arr.dtype == jnp.bool_:
        new = jnp.any(h & vals[:, None], axis=0)
    else:
        new = jnp.sum(jnp.where(h, vals[:, None], jnp.zeros((), arr.dtype)), axis=0, dtype=arr.dtype)
    anyh = jnp.any(hot, axis=0).reshape((8,) + (1,) * tail)
    return jnp.where(anyh, new, flat).reshape(arr.shape)
