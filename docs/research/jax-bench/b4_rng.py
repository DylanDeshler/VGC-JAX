"""Benchmark 4: random numbers per event under vmap.

One turn needs E = 8 actions x 3 targets x 6 events = 144 uniforms per battle.
  foldin   : k_turn = fold_in(key, turn); k_a = fold_in(k_turn, a); u = uniform(fold_in(k_a, 8t+e))   (144 fold_ins)
  split    : keys = split(k_turn, 144); u = vmap(uniform)(keys)
  bulk     : u = uniform(fold_in(key, turn), (8, 3, 6))                                                (1 fold_in)
  hash     : custom counter-based hash (murmur3 finalizer x2) of (seed, turn, a, t, e) -> 24-bit uniform
for impl in threefry2x32 / rbg / unsafe_rbg (typed keys with explicit impl).
"""
from __future__ import annotations

import json
import subprocess
import sys

import jax
import jax.numpy as jnp

from common import bench, md_table, save

B = 16384
A, T, E = 8, 3, 6


def draws_foldin(key, turn):
    kt = jax.random.fold_in(key, turn)
    out = []
    for a in range(A):
        ka = jax.random.fold_in(kt, a)
        for t in range(T):
            for e in range(E):
                out.append(jax.random.uniform(jax.random.fold_in(ka, t * 8 + e)))
    return jnp.stack(out).reshape(A, T, E)


def draws_split(key, turn):
    ks = jax.random.split(jax.random.fold_in(key, turn), A * T * E)
    return jax.vmap(jax.random.uniform)(ks).reshape(A, T, E)


def draws_bulk(key, turn):
    return jax.random.uniform(jax.random.fold_in(key, turn), (A, T, E))


def _mix(h):
    # murmur3 fmix32
    h = h ^ (h >> 16)
    h = h * jnp.uint32(0x85EBCA6B)
    h = h ^ (h >> 13)
    h = h * jnp.uint32(0xC2B2AE35)
    return h ^ (h >> 16)


def draws_hash(seed_u32, turn):
    a = jnp.arange(A, dtype=jnp.uint32)[:, None, None]
    t = jnp.arange(T, dtype=jnp.uint32)[None, :, None]
    e = jnp.arange(E, dtype=jnp.uint32)[None, None, :]
    ctr = (turn.astype(jnp.uint32) << 16) | (a << 8) | (t << 4) | e
    h = _mix(_mix(seed_u32 ^ jnp.uint32(0x9E3779B9)) ^ ctr)
    return (h >> 8).astype(jnp.float32) * (1.0 / (1 << 24))


def consume(u):
    # a little downstream use so XLA can't drop anything
    return (u < 0.5).sum(axis=(-1, -2, -3)) + (u * 100).astype(jnp.int32).sum(axis=(-1, -2, -3))


def run_one(impl, name):
    fn = {"foldin x144": draws_foldin, "split 144": draws_split, "bulk (8,3,6)": draws_bulk}[name]
    turn = jnp.zeros((B,), jnp.int32) + 3
    keys = jax.random.split(jax.random.key(0, impl=impl), B)
    r = bench(lambda k, t, fn=fn: consume(jax.vmap(fn)(k, t)), keys, turn, B=B)
    r["ns_per_draw"] = r["ns_per_elem"] / (A * T * E)
    print("RESULT " + json.dumps(r))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        return run_one(sys.argv[2], sys.argv[3])
    rows = []
    turn = jnp.zeros((B,), jnp.int32) + 3
    r = bench(lambda t: consume(jnp.zeros((B, A, T, E)) + t[:, None, None, None].astype(jnp.float32)), turn, B=B)
    r.update(impl="-", method="baseline (no RNG)")
    rows.append(r)
    for impl in ("threefry2x32", "philox4x32", "philox2x32", "rbg", "unsafe_rbg"):
        for name in ("foldin x144", "split 144", "bulk (8,3,6)"):
            try:
                p = subprocess.run([sys.executable, __file__, "--one", impl, name], capture_output=True, text=True,
                                   timeout=150)
                line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                r = json.loads(line[0][7:]) if line else {"error": p.stderr[-200:]}
            except subprocess.TimeoutExpired:
                r = {"error": "compile+run > 150 s (timeout)"}
            r.update(impl=impl, method=name)
            rows.append(r)
            print(f"4 {impl:12s} {name:14s} compile={r.get('compile_s', float('nan')):.2f}s "
                  f"{r.get('ns_per_elem', float('nan')):.1f} ns/elem  {r.get('ns_per_draw', float('nan')):.2f} ns/draw "
                  f"{r.get('error', '')}", flush=True)
    seeds = jax.random.bits(jax.random.key(1), (B,), jnp.uint32)
    r = bench(lambda s, t: consume(jax.vmap(draws_hash)(s, t)), seeds, turn, B=B)
    r["ns_per_draw"] = r["ns_per_elem"] / (A * T * E)
    r.update(impl="custom u32 hash", method="hash(seed,turn,a,t,e)")
    rows.append(r)
    print(f"4 hash {r['ns_per_elem']:.1f} ns/elem {r['ns_per_draw']:.2f} ns/draw", flush=True)

    # Determinism / branch-independence check for foldin: the draw for (a=5,t=2,e=3) doesn't depend on others
    k = jax.random.key(42)
    u1 = draws_foldin(k, jnp.int32(7))[5, 2, 3]
    u2 = jax.random.uniform(jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(k, 7), 5), 2 * 8 + 3))
    print("foldin draw reproducible standalone:", bool(u1 == u2))
    # rbg vmap peculiarity: per-battle streams are not independent of batch composition
    for impl in ("threefry2x32", "philox4x32", "rbg", "unsafe_rbg"):
        ks = jax.random.split(jax.random.key(7, impl=impl), 8)
        batched = jax.vmap(jax.random.uniform)(ks)
        single = jnp.stack([jax.random.uniform(k) for k in ks])
        print(f"{impl}: vmap(uniform)(keys) == [uniform(k) for k in keys]: {bool(jnp.all(batched == single))}")
        rows.append(dict(impl=impl, method="vmap-exact check", vmap_exact=bool(jnp.all(batched == single))))
    print(md_table(rows, ["impl", "method", "compile_s", "ns_per_elem", "ns_per_draw"],
                   {"compile_s": ".2f", "ns_per_elem": ".1f", "ns_per_draw": ".2f"}))
    save("b4_rng", {"rows": rows, "B": B, "draws_per_turn": A * T * E})


if __name__ == "__main__":
    sys.exit(main())
