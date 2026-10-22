"""Benchmark/check 6: testing and debugging ergonomics.

6A  jax.disable_jit(): eager step time; does lax.cond / lax.switch run only the taken branch? fori_loop?
6B  checkify: overhead of user checks + index checks inside jit(vmap(scan(step)))) and the error message
6C  jax.debug.print: overhead of one host callback per scan step
6D  eager (disable_jit) vs jit outputs agree bit-for-bit (differential-test baseline)
"""
from __future__ import annotations

import contextlib
import io
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import checkify

from common import bench, save
from toy_battle import init, step

rows = []


def part_a():
    s = init(jax.random.PRNGKey(0))
    a = jnp.array([[0, 4], [7, 11]], jnp.int32)
    calls = {"cond_true": 0, "cond_false": 0, "switch": [0] * 4, "fori": 0}

    def t_branch(x):
        calls["cond_true"] += 1
        return x + 1

    def f_branch(x):
        calls["cond_false"] += 1
        return x - 1

    def sw(k):
        def b(x):
            calls["switch"][k] += 1
            return x * k
        return b

    with jax.disable_jit():
        lax.cond(jnp.int32(3) > 2, t_branch, f_branch, jnp.int32(1))
        lax.switch(jnp.int32(2), [sw(k) for k in range(4)], jnp.int32(5))

        def body(i, x):
            calls["fori"] += 1
            return x + i
        lax.fori_loop(0, 8, body, jnp.int32(0))
        step(s, a)  # warm
        t = time.perf_counter()
        n = 3
        for _ in range(n):
            s2 = step(s, a)
        jax.block_until_ready(s2)
        eager_ms = (time.perf_counter() - t) / n * 1e3
    jstep = jax.jit(step)
    jax.block_until_ready(jstep(s, a))
    t = time.perf_counter()
    for _ in range(200):
        s3 = jstep(s, a)
    jax.block_until_ready(s3)
    jit_us = (time.perf_counter() - t) / 200 * 1e6
    print("6A disable_jit branch calls:", calls)
    print(f"6A eager step (disable_jit, B=1): {eager_ms:.1f} ms; jitted step B=1: {jit_us:.1f} us/call")
    rows.append(dict(test="6A", calls=str(calls), eager_ms=eager_ms, jit_us_B1=jit_us))


def checked_step(s, a):
    checkify.check(jnp.all(s.hp >= 0), "negative hp {x}", x=s.hp.min())
    checkify.check(jnp.all((s.boosts >= -6) & (s.boosts <= 6)), "boost out of range")
    return step(s, a)


def part_b():
    B, T = 8192, 16
    st = jax.vmap(init)(jax.random.split(jax.random.PRNGKey(0), B))
    acts = jax.random.randint(jax.random.PRNGKey(1), (T, B, 2, 2), 0, 12, dtype=jnp.int32)

    def roll(step_fn):
        def f(s, acts):
            return lax.scan(lambda s, a: (jax.vmap(step_fn)(s, a), None), s, acts)[0]
        return f

    r0 = bench(roll(step), st, acts, B=B * T)
    r0.update(test="6B no checks")
    rows.append(r0)
    for name, errs in (("user_checks", checkify.user_checks), ("user+index", checkify.user_checks | checkify.index_checks),
                       ("all (user+index+nan+div)", checkify.all_checks)):
        for where in ("outside vmap", "inside vmap"):
            if where == "outside vmap":
                f = checkify.checkify(roll(checked_step), errors=errs)
            else:  # checkify the per-battle step, then vmap it (error object is batched)
                inner = checkify.checkify(checked_step, errors=errs)
                def f(s, acts, inner=inner):
                    def body(s, a):
                        err, s = jax.vmap(inner)(s, a)
                        return s, err
                    s, errs_t = lax.scan(body, s, acts)
                    return errs_t, s
            try:
                r = bench(f, st, acts, B=B * T)
            except Exception as ex:  # noqa: BLE001
                r = {"error": f"{type(ex).__name__}: {str(ex)[:150]}"}
            r.update(test=f"6B checkify {name} ({where})")
            rows.append(r)
            print(f"6B {name:26s} {where:12s} compile={r.get('compile_s', float('nan')):.2f}s "
                  f"{r.get('ns_per_elem', float('nan')):.1f} ns/step  {r.get('error', '')} "
                  f"(no checks: compile={r0['compile_s']:.2f}s {r0['ns_per_elem']:.1f})", flush=True)
    # error message demo: corrupt one battle
    bad = st._replace(hp=st.hp.at[1234, 1, 3].set(-5))
    err, _ = jax.jit(checkify.checkify(roll(checked_step), errors=checkify.user_checks))(bad, acts)
    msg = err.get()
    print("6B error message:", msg.splitlines()[0] if msg else None)
    rows.append(dict(test="6B error message", msg=(msg or "")[:300]))


def part_c():
    B, T = 8192, 16
    st = jax.vmap(init)(jax.random.split(jax.random.PRNGKey(0), B))
    acts = jax.random.randint(jax.random.PRNGKey(1), (T, B, 2, 2), 0, 12, dtype=jnp.int32)

    def f(s, acts):
        def body(s, a):
            s = jax.vmap(step)(s, a)
            jax.debug.print("turn {t} alive {h}", t=s.turn[0], h=(s.hp > 0).sum())
            return s, None
        return lax.scan(body, s, acts)[0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = bench(f, st, acts, B=B * T)
    r.update(test="6C debug.print once per scan step")
    rows.append(r)
    print(f"6C debug.print per step: {r['ns_per_elem']:.1f} ns/step (vs no-checks row above)")


def part_d():
    s = init(jax.random.PRNGKey(5))
    a = jnp.array([[1, 5], [9, 2]], jnp.int32)
    with jax.disable_jit():
        e = step(s, a)
    j = jax.jit(step)(s, a)
    same = all(bool(np.array_equal(np.asarray(x), np.asarray(y))) for x, y in zip(jax.tree.leaves(e), jax.tree.leaves(j)))
    print("6D eager == jit bit-exact:", same)
    rows.append(dict(test="6D eager==jit", same=same))


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    save("b6_ergonomics", {"rows": rows})
