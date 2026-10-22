"""DrawKey packing (contract/drawkey.md §2-§3), shared by the engine (jnp) and the harness (numpy/int).

c0 = turn(11) | step_tag(3) | kind(8) | actor(4) | target(4)          (LSB first, 30 bits)
c1 = hit(4) | ord(4) | sub(8) | subject(12) | action_ord(2)            (30 bits)
c2 = step_actor(4) | step_subject(12)                                   (16 bits)
-1 fields are all-ones in their width. Works on Python ints, numpy and jax arrays alike.
"""
from __future__ import annotations

from typing import NamedTuple

FIELDS = ("kind", "turn", "step_tag", "step_actor", "step_subject", "actor", "target", "subject", "hit", "ord", "sub",
          "action_ord")
# (word, shift, width)
LAYOUT = {
    "turn": (0, 0, 11), "step_tag": (0, 11, 3), "kind": (0, 14, 8), "actor": (0, 22, 4), "target": (0, 26, 4),
    "hit": (1, 0, 4), "ord": (1, 4, 4), "sub": (1, 8, 8), "subject": (1, 16, 12), "action_ord": (1, 28, 2),
    "step_actor": (2, 0, 4), "step_subject": (2, 4, 12),
}


class DrawKey(NamedTuple):
    """Semantic key of one random decision; each field an int32 scalar/array (or Python int)."""
    kind: object
    turn: object
    step_tag: object
    step_actor: object
    step_subject: object
    actor: object
    target: object
    subject: object
    hit: object = 0
    ord: object = 0
    sub: object = 0
    action_ord: object = 0


def pack(key, xp=None):
    """-> (c0, c1, c2) as uint32. `xp` = numpy or jax.numpy (None: Python ints)."""
    words = [0, 0, 0]
    for name, (w, shift, width) in LAYOUT.items():
        v = getattr(key, name) if not isinstance(key, dict) else key[name]
        mask = (1 << width) - 1
        if xp is None:
            words[w] |= (int(v) & mask) << shift
        else:
            words[w] = words[w] | ((xp.asarray(v).astype(xp.uint32) & xp.uint32(mask)) << xp.uint32(shift))
    if xp is None:
        return tuple(words)
    return tuple(xp.asarray(x, dtype=xp.uint32) for x in words)


def unpack(c0, c1, c2) -> dict:
    """Inverse of pack for Python ints; fields with all-ones are returned as -1 for actor/target/step_actor."""
    words = (int(c0), int(c1), int(c2))
    out = {}
    for name, (w, shift, width) in LAYOUT.items():
        v = (words[w] >> shift) & ((1 << width) - 1)
        if name in ("actor", "target", "step_actor") and v == (1 << width) - 1:
            v = -1
        out[name] = v
    return out


def fmix32_int(h: int) -> int:
    """MurmurHash3 finaliser on Python ints (reference implementation for tests and the harness)."""
    h &= 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def hash32_int(w0: int, w1: int, c0: int, c1: int, c2: int) -> int:
    """contract §2: fmix32(fmix32(fmix32(w0 ^ c0) ^ w1 ^ (c1 * 0x85EBCA6B)) ^ (c2 * 0xC2B2AE35))."""
    a = fmix32_int(w0 ^ c0)
    b = fmix32_int(a ^ w1 ^ ((c1 * 0x85EBCA6B) & 0xFFFFFFFF))
    return fmix32_int(b ^ ((c2 * 0xC2B2AE35) & 0xFFFFFFFF))
