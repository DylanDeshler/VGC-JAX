"""Benchmark 3: state representation.

3A  tree_where(mask, new, old) over a ~2 KB/battle state split into F leaves (F = 1..1024), standalone.
3B  the same inside a step that only modifies k leaves (XLA folds select(p, x, x) -> x for untouched leaves).
3C  jit dispatch overhead vs number of leaves (B=1, Python-side pytree flatten / arg handling).
3D  dtype: storage/compute in int8/int16/int32/float32 for (i) a bandwidth-bound masked copy and
    (ii) a compute-bound ~200-op damage-calc chain.
3E  volatile flags: bool[64] vs int8[64] vs uint32[2] bitfield (test batched bit, set static bit, clear).
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from common import bench, compile_fn, md_table, save
from toy_battle import damage_calc

B = 16384
STATE_BYTES = 2048


def make_state(F, Bn, dtype=jnp.int16):
    per = STATE_BYTES // jnp.dtype(dtype).itemsize // F
    keys = jax.random.split(jax.random.PRNGKey(F), F)
    return {f"f{i:04d}": jax.random.randint(keys[i], (Bn, per), 0, 100).astype(dtype) for i in range(F)}


def tree_where(m, a, b):
    return jax.tree.map(lambda x, y: jnp.where(m.reshape(m.shape + (1,) * (x.ndim - 1)), x, y), a, b)


def part_a(rows):
    m = jax.random.bernoulli(jax.random.PRNGKey(9), 0.5, (B,))
    for F in (1, 4, 16, 64, 256, 1024):
        s1, s2 = make_state(F, B), make_state(F, B)
        s2 = jax.tree.map(lambda x: x + 1, s2)
        r = bench(lambda m, a, b: tree_where(m, a, b), m, s1, s2, B=B)
        gbs = 3 * STATE_BYTES * B / r["run_s"] / 1e9
        r.update(test="3A tree_where standalone", F=F, GBps=gbs)
        rows.append(r)
        print(f"3A F={F:5d} lower={r['lower_s']:.3f}s compile={r['compile_s']:.3f}s {r['ns_per_elem']:.1f} ns/elem "
              f"{gbs:.1f} GB/s", flush=True)


def part_b(rows):
    """Step modifies k of F leaves, then state = tree_where(mask, new, state) over all F leaves."""
    m = jax.random.bernoulli(jax.random.PRNGKey(9), 0.5, (B,))
    F = 256
    s = make_state(F, B)
    names = sorted(s)
    for k in (1, 8, 64, 256):
        def f(m, s, k=k):
            new = dict(s)
            for n in names[:k]:
                new[n] = s[n] * 3 + 1
            return tree_where(m, new, s)
        r = bench(f, m, s, B=B, jit_kwargs={"donate_argnums": (1,)} if False else None)
        r.update(test="3B step+tree_where (F=256)", F=F, k_modified=k)
        rows.append(r)
        print(f"3B F={F} k={k:4d} compile={r['compile_s']:.3f}s {r['ns_per_elem']:.1f} ns/elem", flush=True)


def part_c(rows):
    for F in (1, 16, 64, 256, 1024):
        s = make_state(F, 1)
        f = jax.jit(lambda s: jax.tree.map(lambda x: x + 1, s))
        jax.block_until_ready(f(s))
        n = 2000 if F <= 256 else 300
        t = time.perf_counter()
        for _ in range(n):
            out = f(s)
        jax.block_until_ready(out)
        us = (time.perf_counter() - t) / n * 1e6
        rows.append({"test": "3C jit dispatch, B=1", "F": F, "us_per_call": us})
        print(f"3C F={F:5d} {us:.1f} us/call", flush=True)


def part_d(rows):
    n_el = 1024  # elements per battle
    m = jax.random.bernoulli(jax.random.PRNGKey(9), 0.5, (B, 1))
    for dt in (jnp.int8, jnp.int16, jnp.int32, jnp.float32):
        x = jax.random.randint(jax.random.PRNGKey(1), (B, n_el), 0, 100).astype(dt)
        y = (x + 1).astype(dt)
        r = bench(lambda m, x, y: jnp.where(m, x + 1, y), m, x, y, B=B)
        r.update(test="3D masked copy [B,1024]", dtype=jnp.dtype(dt).name, compute="native",
                 GBps=3 * n_el * jnp.dtype(dt).itemsize * B / r["run_s"] / 1e9)
        rows.append(r)
        print(f"3D copy {jnp.dtype(dt).name:8s} {r['ns_per_elem']:.1f} ns/elem {r['GBps']:.1f} GB/s", flush=True)
    # compute-bound damage chain; inputs [B, 64] (64 independent calcs per battle)
    Bc, W = B, 64
    ks = jax.random.split(jax.random.PRNGKey(3), 8)
    base = dict(
        a=jax.random.randint(ks[0], (Bc, W), 60, 120), ab=jax.random.randint(ks[1], (Bc, W), -2, 3),
        d=jax.random.randint(ks[2], (Bc, W), 60, 120), db=jax.random.randint(ks[3], (Bc, W), -2, 3),
        bp=jax.random.randint(ks[4], (Bc, W), 40, 120), mt=jax.random.randint(ks[5], (Bc, W), 0, 18),
        t0=jax.random.randint(ks[6], (Bc, W), 0, 18), t1=jax.random.randint(ks[7], (Bc, W), 0, 18),
    )
    u = jax.random.uniform(ks[0], (Bc, W))
    for dt in (jnp.int8, jnp.int16, jnp.int32, jnp.float32):
        inp = {k: v.astype(dt) for k, v in base.items()}
        for compute in ("int32", "native"):
            if compute == "native" and dt in (jnp.int32, jnp.float32):  # float indices are illegal; int32 native == int32
                continue
            def f(inp, u, compute=compute, dt=dt):
                c = (lambda v: v.astype(jnp.int32)) if compute == "int32" else (lambda v: v)
                def one(a, ab, d, db, bp, mt, t0, t1, uu):
                    return damage_calc(c(a), c(ab), c(d), c(db), c(bp), c(mt).astype(jnp.int32),
                                       jnp.stack([c(t0), c(t1)]).astype(jnp.int32), jnp.stack([c(t1), c(t0)]).astype(jnp.int32),
                                       uu, uu < 0.04, False, 1, False, False)
                out = jax.vmap(jax.vmap(one))(inp["a"], inp["ab"], inp["d"], inp["db"], inp["bp"], inp["mt"], inp["t0"], inp["t1"], u)
                return out.astype(dt)
            r = bench(f, inp, u, B=Bc * W)
            r.update(test="3D damage chain (~200 ops)", dtype=jnp.dtype(dt).name, compute=compute)
            rows.append(r)
            print(f"3D dmg  {jnp.dtype(dt).name:8s} compute={compute:6s} {r['ns_per_elem']:.2f} ns/calc", flush=True)


def part_e(rows):
    Bm = B * 12  # every mon
    kk = jax.random.randint(jax.random.PRNGKey(4), (Bm,), 0, 64)
    reps = {
        "bool[64]": jax.random.bernoulli(jax.random.PRNGKey(5), 0.1, (Bm, 64)),
        "int8[64]": jax.random.bernoulli(jax.random.PRNGKey(5), 0.1, (Bm, 64)).astype(jnp.int8),
        "uint32[2]": jax.random.bits(jax.random.PRNGKey(5), (Bm, 2), jnp.uint32),
    }
    for name, v in reps.items():
        def f(v, kk, name=name):
            if name == "uint32[2]":
                word = jnp.take_along_axis(v, (kk // 32)[:, None], 1)[:, 0]
                test = (word >> (kk % 32).astype(jnp.uint32)) & 1
                v = v.at[:, 0].set(v[:, 0] | jnp.uint32(1 << 5))              # set static bit 5 (e.g. PROTECT)
                v = jnp.where(test[:, None] == 1, v & jnp.uint32(0xFFFF0000), v)  # clear a group of flags
                return v, test.astype(jnp.int32)
            test = jnp.take_along_axis(v, kk[:, None], 1)[:, 0].astype(jnp.int32)
            v = v.at[:, 5].set(True if v.dtype == jnp.bool_ else 1)
            v = jnp.where(test[:, None] == 1, v.at[:, :16].set(0 if v.dtype != jnp.bool_ else False), v)
            return v, test
        r = bench(f, v, kk, B=Bm)
        r.update(test="3E volatile flags per mon", repr=name, bytes_per_mon=int(v[0].nbytes))
        rows.append(r)
        print(f"3E {name:10s} {r['ns_per_elem']:.2f} ns/mon", flush=True)


def part_f(rows):
    """Container type: NamedTuple vs flax.struct.dataclass vs chex.dataclass vs dict (64 leaves, B=1)."""
    import collections
    import dataclasses
    import chex
    from flax import struct
    names = [f"f{i:02d}" for i in range(64)]
    vals = {n: jnp.zeros((2, 6), jnp.int16) for n in names}
    NT = collections.namedtuple("NT", names)
    FS = struct.dataclass(type("FS", (), {"__annotations__": {n: jax.Array for n in names}}))
    CD = chex.dataclass(type("CD", (), {"__annotations__": {n: jax.Array for n in names}}))
    RD = jax.tree_util.register_dataclass(dataclasses.make_dataclass("RD", [(n, jax.Array) for n in names], frozen=True),
                                          data_fields=names, meta_fields=[])
    objs = {"dict": dict(vals), "NamedTuple": NT(**vals), "flax.struct.dataclass": FS(**vals),
            "chex.dataclass": CD(**vals), "jax register_dataclass": RD(**vals)}
    for name, o in objs.items():
        n = 3000
        t = time.perf_counter()
        for _ in range(n):
            leaves, td = jax.tree.flatten(o)
            jax.tree.unflatten(td, leaves)
        flat_us = (time.perf_counter() - t) / n * 1e6
        f = jax.jit(lambda s: jax.tree.map(lambda x: x + 1, s))
        jax.block_until_ready(f(o))
        t = time.perf_counter()
        for _ in range(1000):
            out = f(o)
        jax.block_until_ready(out)
        call_us = (time.perf_counter() - t) / 1000 * 1e6
        rows.append({"test": "3F container type (64 leaves, B=1)", "container": name, "flatten_unflatten_us": flat_us,
                     "jit_call_us": call_us})
        print(f"3F {name:24s} flatten+unflatten {flat_us:.1f} us  jit call {call_us:.1f} us", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="ABCDEF")
    args = ap.parse_args()
    rows = []
    for p, fn in zip("ABCDEF", (part_a, part_b, part_c, part_d, part_e, part_f)):
        if p in args.part:
            fn(rows)
    save(f"b3_state_{args.part}", {"rows": rows, "B": B})


if __name__ == "__main__":
    sys.exit(main())
