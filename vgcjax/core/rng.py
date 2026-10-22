"""Counter-based, semantically keyed RNG with forced-outcome and draw-log modes (ADR-012, contract/drawkey.md).

Every random decision is `hash32(w0, w1, pack(DrawKey))` where (w0, w1) = bits(fold_in(key, step_counter)). Draws
are never positional, so masked (not executed) branches cannot shift later draws. Each draw takes an `active`
predicate: inactive draws return a value but are neither logged nor checked against the forced table.

Forced mode: an exact match of the packed key in `ForcedOutcomes` overrides the natural value; a non-degenerate
active draw with no match sets ERR_FORCED_MISSING and uses the natural value. Degenerate draws (contract §7) never
consult the table or the log.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from vgcjax.battle import errors as E
from vgcjax.core import drawkey as dk

U32 = jnp.uint32
I32 = jnp.int32
M1, M2, GOLD = 0x85EBCA6B, 0xC2B2AE35, 0x9E3779B9

LOG_USED_FORCED = 1
LOG_IS_TIE = 2


@struct.dataclass
class ForcedOutcomes:
    keys: jax.Array    # u32[KF, 3] packed DrawKeys (contract §2)
    value: jax.Array   # i32[KF] value in the engine domain
    valid: jax.Array   # bool[KF]

    @staticmethod
    def empty(kf: int) -> "ForcedOutcomes":
        return ForcedOutcomes(jnp.zeros((kf, 3), U32), jnp.zeros(kf, I32), jnp.zeros(kf, bool))


@struct.dataclass
class DrawLog:
    keys: jax.Array    # u32[L, 3]
    args: jax.Array    # i32[L, 2] distribution arguments (e.g. numerator, denominator / n options)
    value: jax.Array   # i32[L]
    flags: jax.Array   # i32[L]: bit0 used_forced, bit1 is_tie, bits 8..15 tied-member mask
    count: jax.Array   # i32

    @staticmethod
    def empty(n: int) -> "DrawLog":
        return DrawLog(jnp.zeros((n, 3), U32), jnp.zeros((n, 2), I32), jnp.zeros(n, I32), jnp.zeros(n, I32),
                       jnp.zeros((), I32))


@struct.dataclass
class Rng:
    w: jax.Array                       # u32[2]
    forced: ForcedOutcomes | None
    log: DrawLog | None
    err: jax.Array                     # u32 error bits raised by draws


def make_rng(key, step_counter, forced: ForcedOutcomes | None = None, log: DrawLog | None = None) -> Rng:
    """contract §2: w0, w1 = jax.random.bits(fold_in(key, step_counter), (2,), uint32)."""
    w = jax.random.bits(jax.random.fold_in(key, step_counter), (2,), U32)
    return Rng(w=w, forced=forced, log=log, err=jnp.zeros((), U32))


def fmix32(h):
    h = jnp.asarray(h, U32)
    h = h ^ (h >> 16)
    h = h * U32(M1)
    h = h ^ (h >> 13)
    h = h * U32(M2)
    return h ^ (h >> 16)


def hash32(w, c):
    """contract §2 hash of packed key words c = (c0, c1, c2)."""
    a = fmix32(w[0] ^ c[0])
    b = fmix32(a ^ w[1] ^ (c[1] * U32(M1)))
    return fmix32(b ^ (c[2] * U32(M2)))


def _pack(key: dk.DrawKey):
    return dk.pack(key, jnp)


def _lookup(rng: Rng, c):
    f = rng.forced
    if f is None:
        return jnp.zeros((), bool), jnp.zeros((), I32)
    m = f.valid & (f.keys[:, 0] == c[0]) & (f.keys[:, 1] == c[1]) & (f.keys[:, 2] == c[2])
    return jnp.any(m), f.value[jnp.argmax(m)]


def _log(rng: Rng, c, a0, a1, value, flags, active) -> Rng:
    log = rng.log
    if log is None:
        return rng
    n = log.keys.shape[0]
    ok = active & (log.count < n)
    i = jnp.minimum(log.count, n - 1)
    kw = jnp.stack(c)
    log = DrawLog(
        keys=log.keys.at[i].set(jnp.where(ok, kw, log.keys[i])),
        args=log.args.at[i].set(jnp.where(ok, jnp.stack([jnp.asarray(a0, I32), jnp.asarray(a1, I32)]), log.args[i])),
        value=log.value.at[i].set(jnp.where(ok, jnp.asarray(value, I32), log.value[i])),
        flags=log.flags.at[i].set(jnp.where(ok, jnp.asarray(flags, I32), log.flags[i])),
        count=log.count + jnp.asarray(active, I32),
    )
    err = rng.err | jnp.where(active & ~ok, U32(E.ERR_DRAWLOG_OVERFLOW), U32(0))
    return rng.replace(log=log, err=err)


def _resolve(rng: Rng, key: dk.DrawKey, natural, degenerate, degen_value, a0, a1, active):
    """Common path: forced override, missing-key flag, logging. Returns (value, rng)."""
    c = _pack(key)
    live = jnp.asarray(active) & ~jnp.asarray(degenerate)
    hit, fval = _lookup(rng, c)
    use_forced = live & hit
    value = jnp.where(degenerate, degen_value, jnp.where(use_forced, fval, natural))
    if rng.forced is not None:
        rng = rng.replace(err=rng.err | jnp.where(live & ~hit, U32(E.ERR_FORCED_MISSING), U32(0)))
    rng = _log(rng, c, a0, a1, value, jnp.where(use_forced, LOG_USED_FORCED, 0), live)
    return value, rng


def chance(rng: Rng, key: dk.DrawKey, n, d, active=True):
    """PS randomChance(n, d): hash % d < n. Degenerate when n >= d or n <= 0. -> (bool, rng)"""
    n = jnp.asarray(n, I32)
    d = jnp.asarray(d, I32)
    h = hash32(rng.w, _pack(key))
    natural = (h % jnp.maximum(d, 1).astype(U32)).astype(I32) < n
    degen = (n >= d) | (n <= 0)
    v, rng = _resolve(rng, key, natural.astype(I32), degen, (n >= d).astype(I32), n, d, active)
    return v.astype(bool), rng


def randint(rng: Rng, key: dk.DrawKey, a, b, active=True):
    """PS random(a, b): a + hash % (b - a), value in [a, b). Degenerate when b - a <= 1. -> (int, rng)"""
    a = jnp.asarray(a, I32)
    b = jnp.asarray(b, I32)
    span = jnp.maximum(b - a, 1)
    h = hash32(rng.w, _pack(key))
    natural = a + (h % span.astype(U32)).astype(I32)
    return _resolve(rng, key, natural, b - a <= 1, a, a, b, active)


def sample(rng: Rng, key: dk.DrawKey, n, active=True):
    """PS sample(list) as an index: hash % n. Degenerate when n <= 1. -> (index, rng)"""
    n = jnp.asarray(n, I32)
    h = hash32(rng.w, _pack(key))
    natural = (h % jnp.maximum(n, 1).astype(U32)).astype(I32)
    return _resolve(rng, key, natural, n <= 1, jnp.zeros((), I32), n, 0, active)


def sample_ref(rng: Rng, key: dk.DrawKey, refs, valid, active=True):
    """Sample a member of `refs[valid]` (in the given order). Natural: the (hash % count)-th valid ref; forced: the
    forced value *is* the ref (engine domain, e.g. side*4+b). -> (ref, rng)"""
    refs = jnp.asarray(refs, I32)
    valid = jnp.asarray(valid, bool)
    count = jnp.sum(valid.astype(I32))
    h = hash32(rng.w, _pack(key))
    k = (h % jnp.maximum(count, 1).astype(U32)).astype(I32)
    rank = jnp.cumsum(valid.astype(I32)) - 1
    pick = jnp.argmax(valid & (rank == k))
    first = jnp.argmax(valid)
    natural = refs[pick]
    return _resolve(rng, key, natural, count <= 1, refs[first], count, 0, active)


# ------------------------------------------------------------------ tie records (contract §4)
def _perm_rank(order, valid):
    """Lexicographic rank of the valid members' resulting order (positions index the canonical member list)."""
    m = order.shape[0]
    vo = valid[order]
    fact = jnp.cumprod(jnp.concatenate([jnp.ones(1, I32), jnp.arange(1, m, dtype=I32)]))
    nvalid = jnp.sum(valid.astype(I32))
    # position p of the result (only the first nvalid positions are valid members)
    rank = jnp.zeros((), I32)
    for p in range(m):
        later = jnp.arange(m) > p
        smaller = jnp.sum((later & vo & (order < order[p])).astype(I32))
        remaining = nvalid - 1 - p
        f = fact[jnp.clip(remaining, 0, m - 1)]
        rank = rank + jnp.where(vo[p] & (remaining >= 0), smaller * f, 0)
    return rank


def _perm_unrank(rank, valid):
    """Order (over canonical member indices, valid members first) with lexicographic rank `rank`."""
    m = valid.shape[0]
    fact = jnp.cumprod(jnp.concatenate([jnp.ones(1, I32), jnp.arange(1, m, dtype=I32)]))
    nvalid = jnp.sum(valid.astype(I32))
    pool = valid
    order = jnp.zeros(m, I32)
    r = jnp.asarray(rank, I32)
    for p in range(m):
        remaining = nvalid - 1 - p
        f = fact[jnp.clip(remaining, 0, m - 1)]
        q = jnp.where(remaining >= 0, r // f, 0)
        r = jnp.where(remaining >= 0, r % f, r)
        cum = jnp.cumsum(pool.astype(I32)) - 1
        pick = jnp.argmax(pool & (cum == q))
        order = order.at[p].set(jnp.where(remaining >= 0, pick, 0))
        pool = pool & ~((jnp.arange(m) == pick) & (remaining >= 0))
    # invalid members after the valid ones, in index order
    idx = jnp.arange(m)
    inv_first = jnp.argsort(jnp.where(valid, m + idx, idx))
    order = jnp.where(idx < nvalid, order, inv_first[jnp.clip(idx - nvalid, 0, m - 1)])
    return order


def tie_order(rng: Rng, key: dk.DrawKey, members, valid, active=True, select=False):
    """Resolve a tie record (contract §4) over `members` (canonical member order, static length M <= 8).

    Natural: members ordered by descending per-member hash. Forced: `value` is the winner's member id (select=True,
    SPEED_TIE_SELECT) or the permutation rank over the valid members (other tie kinds).
    Returns (order: i32[M] indices into `members`, valid ones first; rng). Logged only when >= 2 members are valid.
    """
    members = jnp.asarray(members, I32)
    valid = jnp.asarray(valid, bool)
    m = members.shape[0]
    c = _pack(key)
    h0 = hash32(rng.w, c)
    hs = fmix32(h0 ^ (jnp.arange(m, dtype=U32) * U32(GOLD) + U32(1)))
    nat_order = jnp.argsort(jnp.where(valid, -(hs >> 1).astype(I32), jnp.iinfo(I32).max - jnp.arange(m)))
    nvalid = jnp.sum(valid.astype(I32))
    live = jnp.asarray(active) & (nvalid >= 2)
    hit, fval = _lookup(rng, c)
    use_forced = live & hit
    if select:
        win = jnp.argmax(valid & (members == fval))
        forced_ok = jnp.any(valid & (members == fval))
        winner = jnp.where(use_forced & forced_ok, win, nat_order[0])
        rest = jnp.argsort(jnp.where(jnp.arange(m) == winner, -1, jnp.where(valid, 0, 1)) * m + jnp.arange(m))
        order = rest
        value = members[winner]
    else:
        f_order = _perm_unrank(fval, valid)
        order = jnp.where(use_forced, f_order, nat_order)
        value = _perm_rank(order, valid)
    if rng.forced is not None:
        rng = rng.replace(err=rng.err | jnp.where(live & ~hit, U32(E.ERR_FORCED_MISSING), U32(0)))
    tied_mask = jnp.sum(jnp.where(valid, 1 << jnp.clip(members, 0, 7), 0)) if select else jnp.sum(
        jnp.where(valid, 1 << jnp.arange(m), 0))
    flags = LOG_IS_TIE | jnp.where(use_forced, LOG_USED_FORCED, 0) | (tied_mask << 8)
    rng = _log(rng, c, nvalid, 0, value, flags, live)
    return order, rng
