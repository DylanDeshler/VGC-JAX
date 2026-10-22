"""Benchmark 2: control-flow structure for the per-turn action loop (8) and the target loop (3).

2A  synthetic: 8 actions x 3 targets x ~200-op damage calc; outer in {fori, scan, unroll, while},
    inner in {fori, scan, unroll, vector(vmap over the 3 targets)}; hp update via scatter or one-hot.
2B  the toy battle step (6-entry queue, dynamic speed, 64-way effect switch): fori vs unrolled queue,
    vector vs unrolled target axis, gather/scatter vs one-hot indexing.
"""
from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
from jax import lax

from common import bench, md_table, save
from toy_battle import damage_calc, init, step

B = 16384


def make_inputs(key, Bn):
    ks = jax.random.split(key, 10)
    ri = lambda k, shape, lo, hi: jax.random.randint(k, (Bn,) + shape, lo, hi, dtype=jnp.int32)
    st = dict(
        hp=ri(ks[0], (4,), 150, 300), stats=ri(ks[1], (4, 6), 60, 300), boosts=ri(ks[2], (4, 7), -2, 3),
        types=ri(ks[3], (4, 2), 0, 18),
    )
    q = dict(
        actor=ri(ks[4], (8,), 0, 4), bp=ri(ks[5], (8,), 40, 130), mtype=ri(ks[6], (8,), 0, 18),
        tgt=ri(ks[7], (8, 3), 0, 4), tmask=jax.random.bernoulli(ks[8], 0.6, (Bn, 8, 3)),
        u=jax.random.uniform(ks[9], (Bn, 8, 3, 3)), n_act=ri(ks[9], (), 4, 9),
    )
    return st, q


def per_target(st, q, a, t, hp):
    actor = q["actor"][a]
    tg = q["tgt"][a, t]
    spread = q["tmask"][a].sum() > 1
    dmg = damage_calc(st["stats"][actor, 1], st["boosts"][actor, 0], st["stats"][tg, 2], st["boosts"][tg, 1],
                      q["bp"][a], q["mtype"][a], st["types"][actor], st["types"][tg], q["u"][a, t, 0],
                      q["u"][a, t, 1] < 0.04, spread, 1, False, False)
    alive = hp[actor] > 0
    return jnp.where(q["tmask"][a, t] & alive & (q["u"][a, t, 2] < 0.9), dmg, 0), tg


def apply_dmg(hp, tg, dmg, index_mode):
    if index_mode == "scatter":
        return jnp.maximum(hp.at[tg].add(-dmg), 0)
    oh = (jnp.arange(4) == jnp.atleast_1d(tg)[..., None])
    return jnp.maximum(hp - (oh * jnp.atleast_1d(dmg)[..., None]).sum(0), 0)


def make_fn(outer, inner, index_mode="scatter"):
    def action(st, q, a, hp):
        if inner == "vector":
            dmg, tg = jax.vmap(lambda t: per_target(st, q, a, t, hp))(jnp.arange(3))
            return apply_dmg(hp, tg, dmg, index_mode)
        if inner == "unroll":
            for t in range(3):
                dmg, tg = per_target(st, q, a, t, hp)
                hp = apply_dmg(hp, tg, dmg, index_mode)
            return hp
        if inner == "fori":
            def tb(t, hp):
                dmg, tg = per_target(st, q, a, t, hp)
                return apply_dmg(hp, tg, dmg, index_mode)
            return lax.fori_loop(0, 3, tb, hp)
        if inner == "scan":
            def tb(hp, t):
                dmg, tg = per_target(st, q, a, t, hp)
                return apply_dmg(hp, tg, dmg, index_mode), None
            return lax.scan(tb, hp, jnp.arange(3))[0]
        raise ValueError(inner)

    def one(st, q):
        hp = st["hp"]
        if outer == "fori":
            hp = lax.fori_loop(0, 8, lambda a, hp: jnp.where(a < q["n_act"], action(st, q, a, hp), hp), hp)
        elif outer == "fori_nomask":
            hp = lax.fori_loop(0, 8, lambda a, hp: action(st, q, a, hp), hp)
        elif outer == "scan":
            hp = lax.scan(lambda hp, a: (jnp.where(a < q["n_act"], action(st, q, a, hp), hp), None), hp, jnp.arange(8))[0]
        elif outer == "unroll":
            for a in range(8):
                hp = jnp.where(a < q["n_act"], action(st, q, a, hp), hp)
        elif outer == "while":  # variable trip count per element -> max trip count under vmap
            hp = lax.while_loop(lambda c: c[0] < q["n_act"], lambda c: (c[0] + 1, action(st, q, c[0], c[1])),
                                (jnp.int32(0), hp))[1]
        else:
            raise ValueError(outer)
        return hp

    def f(st, q):
        return jax.vmap(one)(st, q)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="AB")
    args = ap.parse_args()
    fmt = {"compile_s": ".2f", "lower_s": ".2f", "ns_per_elem": ".1f", "temp_MB": ".1f"}
    res = {}
    if "A" in args.part:
        st, q = make_inputs(jax.random.PRNGKey(0), B)
        rows = []
        combos = [(o, i, "scatter") for o in ("fori", "scan", "unroll", "while") for i in ("fori", "unroll", "vector")]
        combos += [("fori", "scan", "scatter"), ("fori_nomask", "vector", "scatter"),
                   ("fori", "vector", "onehot"), ("unroll", "vector", "onehot"), ("fori", "unroll", "onehot")]
        for outer, inner, im in combos:
            r = bench(make_fn(outer, inner, im), st, q, B=B)
            r.update(outer=outer, inner=inner, index=im, temp_MB=r.get("temp_bytes", 0) / 2**20)
            rows.append(r)
            print(f"2A {outer:12s} {inner:7s} {im:8s} lower={r['lower_s']:.2f}s compile={r['compile_s']:.2f}s "
                  f"{r['ns_per_elem']:.1f} ns/elem", flush=True)
        res["2A"] = rows
        print(md_table(rows, ["outer", "inner", "index", "lower_s", "compile_s", "ns_per_elem", "temp_MB"], fmt))
    if "B" in args.part:
        Bb = 8192
        states = jax.vmap(init)(jax.random.split(jax.random.PRNGKey(1), Bb))
        acts = jax.random.randint(jax.random.PRNGKey(2), (Bb, 2, 2), 0, 12, dtype=jnp.int32)
        rows = []
        for loop_mode in ("fori", "unroll"):
            for tm in ("vector", "unroll"):
                for im in ("gather", "onehot"):
                    f = lambda s, a, lm=loop_mode, tm=tm, im=im: jax.vmap(
                        lambda s1, a1: step(s1, a1, loop_mode=lm, target_mode=tm, index_mode=im))(s, a)
                    r = bench(f, states, acts, B=Bb)
                    r.update(queue_loop=loop_mode, targets=tm, index=im, temp_MB=r.get("temp_bytes", 0) / 2**20)
                    rows.append(r)
                    print(f"2B {loop_mode:6s} {tm:6s} {im:6s} lower={r['lower_s']:.2f}s compile={r['compile_s']:.2f}s "
                          f"{r['ns_per_elem']:.1f} ns/elem", flush=True)
        res["2B"] = rows
        print(md_table(rows, ["queue_loop", "targets", "index", "lower_s", "compile_s", "ns_per_elem", "temp_MB"], fmt))
    save(f"b2_control_{args.part}", res)


if __name__ == "__main__":
    sys.exit(main())
