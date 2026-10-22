"""Benchmark 1: effect dispatch under vmap.

1A  single hook call, N handlers, four dispatch styles
1B  8 hook points per turn: one big switch at every hook vs per-hook switches vs tables
1C  compile time of a switch nested in an 8-action x 3-target loop nest
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from common import bench, md_table, save
from handlers import (N_TEMPLATES, identity, make_branches, make_consts, random_ctx,
                      template_eval)

NS = (8, 32, 128, 512)
B = 16384
N_HOOKS = 8


# ---------------------------------------------------------------- 1A variants
def v_switch(n):
    br = make_branches(n)

    def f(ctx, idx):
        return jax.vmap(lambda c, i: lax.switch(i, br, c))(ctx, idx)
    return f


def _all_outs(br, c):
    outs = [b(c) for b in br]
    return [jnp.stack([o[j] for o in outs]) for j in range(len(c))]


def v_onehot(n):
    br = make_branches(n)

    def one(c, i):
        st = _all_outs(br, c)
        oh = (jnp.arange(n) == i).astype(jnp.int32)
        return tuple(jnp.sum(s * oh) for s in st)

    def f(ctx, idx):
        return jax.vmap(one)(ctx, idx)
    return f


def v_take(n):
    br = make_branches(n)

    def one(c, i):
        st = _all_outs(br, c)
        return tuple(s[i] for s in st)

    def f(ctx, idx):
        return jax.vmap(one)(ctx, idx)
    return f


def _table_eval(tid, cvec, c, with_identity=False):
    """Generic interpreter: evaluate every template with gathered constants, select by tid."""
    cs = (cvec[0], cvec[1], cvec[2], cvec[3])
    tbr = [lambda cc, t=t: template_eval(t, cs, cc) for t in range(N_TEMPLATES)]
    if with_identity:
        tbr = tbr + [identity]
    return lax.switch(tid, tbr, c)


def v_table(n, special_frac=8):
    """Most effects = rows of (template, c1..c4); N//special_frac effects also need a custom branch."""
    t_np, c_np = make_consts(n)
    k = max(1, n // special_frac)
    sp_br = [identity] + make_branches(k, seed=1)
    sp_idx_np = np.zeros(n, np.int32)
    sp_idx_np[np.random.default_rng(2).choice(n, k, replace=False)] = np.arange(1, k + 1)

    def one(c, i):
        T, C, S = jnp.asarray(t_np), jnp.asarray(c_np), jnp.asarray(sp_idx_np)
        c = _table_eval(T[i], C[i], c)
        return lax.switch(S[i], sp_br, c)

    def f(ctx, idx):
        return jax.vmap(one)(ctx, idx)
    return f


# ---------------------------------------------------------------- 1B variants (8 hooks)
def hook_assign(n):
    return (np.random.default_rng(3).permutation(n) % N_HOOKS).astype(np.int32)


def v_naive_hooks(n):
    """At every hook point: switch over all N effect ids (identity for effects without this hook)."""
    br = make_branches(n)
    hk = hook_assign(n)
    per_hook = [[br[k] if hk[k] == h else identity for k in range(n)] for h in range(N_HOOKS)]

    def one(c, i):
        for h in range(N_HOOKS):
            c = lax.switch(i, per_hook[h], c)
        return c

    def f(ctx, idx):
        return jax.vmap(one)(ctx, idx)
    return f


def v_grouped_hooks(n, barrier=False):
    """Per hook: tiny remap table effect->local index (0 = no handler) + switch over that hook's handlers."""
    br = make_branches(n)
    hk = hook_assign(n)
    local = np.zeros((N_HOOKS, n), np.int32)
    per_hook = []
    for h in range(N_HOOKS):
        ks = np.nonzero(hk == h)[0]
        local[h, ks] = np.arange(1, len(ks) + 1)
        per_hook.append([identity] + [br[k] for k in ks])

    def one(c, i):
        L = jnp.asarray(local)
        for h in range(N_HOOKS):
            c = lax.switch(L[h, i], per_hook[h], c)
            if barrier:
                c = lax.optimization_barrier(c)
        return c

    def f(ctx, idx):
        return jax.vmap(one)(ctx, idx)
    return f


def v_table_hooks(n, special_frac=8):
    """Per hook: table row (template id N_TEMPLATES == no-op) + per-hook special switch."""
    t_np, c_np = make_consts(n)
    hk = hook_assign(n)
    T = np.full((N_HOOKS, n), N_TEMPLATES, np.int32)  # no-op by default
    for k in range(n):
        T[hk[k], k] = t_np[k]
    k_sp = max(1, n // special_frac)
    sp_effects = np.random.default_rng(2).choice(n, k_sp, replace=False)
    sp_all = make_branches(k_sp, seed=1)
    S = np.zeros((N_HOOKS, n), np.int32)
    per_hook_sp = [[identity] for _ in range(N_HOOKS)]
    for j, e in enumerate(sp_effects):
        h = hk[e]
        per_hook_sp[h].append(sp_all[j])
        S[h, e] = len(per_hook_sp[h]) - 1

    def one(c, i):
        TT, CC, SS = jnp.asarray(T), jnp.asarray(c_np), jnp.asarray(S)
        for h in range(N_HOOKS):
            c = _table_eval(TT[h, i], CC[i], c, with_identity=True)
            if len(per_hook_sp[h]) > 1:
                c = lax.switch(SS[h, i], per_hook_sp[h], c)
        return c

    def f(ctx, idx):
        return jax.vmap(one)(ctx, idx)
    return f


# ---------------------------------------------------------------- 1C: loop nest compile time
def v_loopnest(n, outer, inner, barrier=False):
    br = make_branches(n)

    def target_body(t, carry, a):
        c, eids = carry
        c = lax.switch(eids[a, t], br, c)
        if barrier:
            c = lax.optimization_barrier(c)
        return (c, eids)

    def action_body(a, carry):
        if inner == "fori":
            return lax.fori_loop(0, 3, lambda t, cr: target_body(t, cr, a), carry)
        for t in range(3):
            carry = target_body(t, carry, a)
        return carry

    def one(c, eids):
        carry = (c, eids)
        if outer == "fori":
            carry = lax.fori_loop(0, 8, action_body, carry)
        else:
            for a in range(8):
                carry = action_body(a, carry)
        return carry[0]

    def f(ctx, eids):
        return jax.vmap(one)(ctx, eids)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="ABC")
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--B", type=int, default=B)
    ap.add_argument("--single", default="")
    ap.add_argument("--timeout", type=int, default=150)
    args = ap.parse_args()
    globals()["B"] = args.B
    ns = [int(x) for x in args.ns.split(",")]
    key = jax.random.PRNGKey(0)
    ctx = random_ctx(key, B)
    res = {}
    fmt = {"compile_s": ".2f", "lower_s": ".2f", "ns_per_elem": ".1f", "temp_MB": ".1f"}

    if "A" in args.part:
        rows = []
        r = bench(lambda c, i: tuple(x + (i & 0) for x in c), ctx, jnp.zeros((B,), jnp.int32), B=B)
        r.update(N=0, variant="baseline (copy)")
        rows.append(r)
        print(f"1A baseline {r['ns_per_elem']:.1f} ns/elem", flush=True)
        for n in ns:
            idx = jax.random.randint(jax.random.PRNGKey(n), (B,), 0, n, dtype=jnp.int32)
            for name, mk in [("switch", v_switch), ("onehot", v_onehot), ("take", v_take), ("table+N/8 sw", v_table)]:
                r = bench(mk(n), ctx, idx, B=B)
                r.update(N=n, variant=name, temp_MB=r.get("temp_bytes", 0) / 2**20)
                rows.append(r)
                print(f"1A N={n:4d} {name:14s} lower={r['lower_s']:.2f}s compile={r['compile_s']:.2f}s "
                      f"{r['ns_per_elem']:.1f} ns/elem temp={r['temp_MB']:.1f}MB", flush=True)
        res["1A"] = rows
        print(md_table(rows, ["N", "variant", "lower_s", "compile_s", "ns_per_elem", "temp_MB"], fmt))

    if "B" in args.part:
        rows = []
        for n in ns:
            idx = jax.random.randint(jax.random.PRNGKey(n), (B,), 0, n, dtype=jnp.int32)
            for name, mk in [("naive N-way x8", v_naive_hooks), ("grouped (N/8)-way x8", v_grouped_hooks),
                             ("grouped + opt_barrier", lambda n: v_grouped_hooks(n, True)),
                             ("table+specials x8", v_table_hooks)]:
                r = bench(mk(n), ctx, idx, B=B)
                r.update(N=n, variant=name, temp_MB=r.get("temp_bytes", 0) / 2**20)
                rows.append(r)
                print(f"1B N={n:4d} {name:22s} lower={r['lower_s']:.2f}s compile={r['compile_s']:.2f}s "
                      f"{r['ns_per_elem']:.1f} ns/elem", flush=True)
        res["1B"] = rows
        print(md_table(rows, ["N", "variant", "lower_s", "compile_s", "ns_per_elem"], fmt))

    if args.single:
        n, outer, inner, bar = args.single.split(",")
        n, bar = int(n), bool(int(bar))
        Bc = 4096
        ctxc = random_ctx(key, Bc)
        eids = jax.random.randint(jax.random.PRNGKey(n), (Bc, 8, 3), 0, n, dtype=jnp.int32)
        r = bench(v_loopnest(n, outer, inner, bar), ctxc, eids, B=Bc)
        print("RESULT " + json.dumps(r))
        return 0

    if "C" in args.part:
        rows = []
        for n in ns:
            for outer, inner in [("fori", "fori"), ("fori", "unroll"), ("unroll", "unroll")]:
                for bar in (0, 1):
                    row = dict(N=n, loops=f"{outer}8/{inner}3", barrier=bool(bar))
                    try:
                        out = subprocess.run([sys.executable, __file__, "--part", "S", "--single", f"{n},{outer},{inner},{bar}"],
                                             capture_output=True, text=True, timeout=args.timeout)
                        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
                        row.update(json.loads(line[0][7:]) if line else {"error": out.stderr[-300:]})
                    except subprocess.TimeoutExpired:
                        row.update(compile_s=f">{args.timeout}s (timeout)", ns_per_elem="-")
                    rows.append(row)
                    print("1C", row, flush=True)
        res["1C"] = rows
        print(md_table(rows, ["N", "loops", "barrier", "lower_s", "compile_s", "ns_per_elem"], fmt))

    res["B"] = B
    save(f"b1_dispatch_{args.part}_B{B}", res)


if __name__ == "__main__":
    sys.exit(main())
