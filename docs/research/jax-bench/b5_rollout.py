"""Benchmark 5: whole-rollout compile + throughput for the toy doubles battle.

rollout = lax.scan over T steps of:
    obs[2] = observe(state, p) ; mask[2] = legal_mask(state, p)
    actions = masked random categorical (stand-in for the policy)
    state  = step(state, actions)  (6-entry queue x 3 targets x ~200-op damage calc + 64-way effect switch
                                    + 16-way item residual switch + faint replacement + terminal check)
    auto-reset: state = tree_where(done, reset_state, state)
reset_state is either a full init() every step (gymnax style) or a gather from a pre-built bank of
initial states (cheap).

Run one config per process (clean RSS):
  python b5_rollout.py --B 8192 [--T 128] [--reset bank|init] [--loop fori|unroll] [--store-obs] [--devices N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8192)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--reset", default="bank")
    ap.add_argument("--loop", default="fori")
    ap.add_argument("--targets", default="vector")
    ap.add_argument("--index", default="gather")
    ap.add_argument("--store-obs", action="store_true")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump-hlo", default="")
    ap.add_argument("--extra-hooks", type=int, default=0)
    ap.add_argument("--barrier", type=int, default=0)
    args = ap.parse_args()
    if args.devices > 1:
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={args.devices}"

    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    if args.cache_dir:
        jax.config.update("jax_compilation_cache_dir", args.cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import peak_rss_mb
    from toy_battle import init, legal_mask, observe, step

    B, T = args.B, args.T
    BANK = 4096

    def env_step(state, actions, bank, key):
        new = jax.vmap(lambda s, a: step(s, a, loop_mode=args.loop, target_mode=args.targets,
                                         index_mode=args.index, extra_hooks=args.extra_hooks,
                                         barrier=bool(args.barrier)))(state, actions)
        if args.reset == "init":
            fresh = jax.vmap(init)(jax.random.split(key, B))
        else:
            idx = jax.random.randint(key, (B,), 0, BANK)
            fresh = jax.tree.map(lambda x: x[idx], bank)
            fresh = fresh._replace(key=jax.random.split(jax.random.fold_in(key, 1), B))
        reward = jnp.where(new.done, jnp.where(new.winner == 0, 1.0, jnp.where(new.winner == 1, -1.0, 0.0)), 0.0)
        done = new.done
        nxt = jax.tree.map(lambda f, n: jnp.where(done.reshape(done.shape + (1,) * (n.ndim - 1)), f, n), fresh, new)
        return nxt, reward, done

    def policy(key, mask):
        # mask [B,2,2,12] -> actions [B,2,2]
        logits = jnp.where(mask, 0.0, -1e9)
        return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)

    def rollout(state, bank, key):
        def body(carry, k):
            state = carry
            k1, k2 = jax.random.split(k)
            obs = jnp.stack([jax.vmap(observe, in_axes=(0, None))(state, p) for p in range(2)], 1)  # [B,2,D]
            mask = jnp.stack([jax.vmap(legal_mask, in_axes=(0, None))(state, p) for p in range(2)], 1)  # [B,2,2,12]
            acts = policy(k1, mask)
            state, rew, done = env_step(state, acts, bank, k2)
            out = (rew.sum(), done.sum())
            if args.store_obs:
                out = out + (obs.astype(jnp.bfloat16), acts.astype(jnp.int8), rew)
            return state, out
        return lax.scan(body, state, jax.random.split(key, T))

    state = jax.vmap(init)(jax.random.split(jax.random.PRNGKey(0), B))
    bank = jax.vmap(init)(jax.random.split(jax.random.PRNGKey(1), BANK))
    key = jax.random.PRNGKey(2)
    if args.devices > 1:
        mesh = Mesh(jax.devices(), ("b",))
        shard = NamedSharding(mesh, P("b"))
        state = jax.device_put(state, shard)
        repl = NamedSharding(mesh, P())
        bank = jax.device_put(bank, repl)

    t0 = time.perf_counter()
    lowered = jax.jit(rollout, donate_argnums=(0,)).lower(state, bank, key)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    try:
        ma = compiled.memory_analysis()
    except Exception:  # noqa: BLE001  (e.g. jax-metal)
        ma = None
    hlo_stats = {}
    if args.dump_hlo:
        with open(args.dump_hlo, "w") as fh:
            fh.write(compiled.as_text())
    try:
        ca = compiled.cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        hlo_stats["flops"] = float(ca.get("flops", -1))
        hlo_stats["bytes_accessed"] = float(ca.get("bytes accessed", -1))
        txt = compiled.as_text()
        hlo_stats["n_fusion_instrs"] = sum(1 for l in txt.splitlines() if " fusion(" in l or "= fusion" in l)
        hlo_stats["n_while"] = txt.count(" while(")
        hlo_stats["n_computations"] = sum(1 for l in txt.splitlines() if l and not l.startswith(" ") and "{" in l)
        hlo_stats["hlo_text_kB"] = len(txt) / 1024
    except Exception as ex:  # noqa: BLE001
        hlo_stats["error"] = repr(ex)[:200]
    times = []
    total_done = 0
    for r in range(args.reps + 1):
        ts = time.perf_counter()
        state, out = compiled(state, bank, jax.random.fold_in(key, r))
        jax.block_until_ready(out)
        if r > 0:
            times.append(time.perf_counter() - ts)
            total_done += int(out[1].sum())
    best = min(times)
    res = dict(extra_hooks=args.extra_hooks, barrier=args.barrier, B=B, T=T, reset=args.reset, loop=args.loop, targets=args.targets, index=args.index,
               store_obs=args.store_obs, devices=args.devices, lower_s=t1 - t0, compile_s=t2 - t1,
               rollout_s=best, steps_per_s=B * T / best, us_per_step_call=best / T * 1e6,
               episodes_per_s=total_done / sum(times),
               temp_MB=ma.temp_size_in_bytes / 2**20 if ma else None,
               arg_MB=ma.argument_size_in_bytes / 2**20 if ma else None,
               out_MB=ma.output_size_in_bytes / 2**20 if ma else None,
               state_bytes_per_battle=sum(x[0].nbytes for x in jax.tree.leaves(state)) if args.devices == 1 else None,
               peak_rss_MB=peak_rss_mb(), tag=args.tag, jax=jax.__version__,
               xla_flags=os.environ.get("XLA_FLAGS", ""), **hlo_stats)
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
