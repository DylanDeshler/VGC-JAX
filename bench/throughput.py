"""Env throughput (09 §1): vmapped auto-reset rollout with uniform legal actions, env-steps/s.
    uv run python bench/throughput.py --stage core --batch 256 --steps 64
"""
import argparse
import time

import jax
import numpy as np

from vgcjax.config import EngineConfig
from vgcjax.core.dex import load_dex
from vgcjax.env import api, teambank as TB
from vgcjax.env.autoreset import AutoResetEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="core")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--bank", default="data/teams/bank_core.npz")
    a = ap.parse_args()
    cfg = EngineConfig(stage=a.stage, max_turns=40)
    dex = load_dex(a.stage)
    env = AutoResetEnv(cfg, dex, TB.load(a.bank))
    B, T = a.batch, a.steps
    es, out = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), B))

    def one(carry, k):
        es, out = carry
        ks = jax.random.split(k, B * 2).reshape(B, 2, -1)
        masks = jax.vmap(lambda s: api.legal_masks(s, cfg, dex))(es.state)
        acts = jax.vmap(api.random_actions)(ks[:, 0], masks, es.state.meta.phase)
        es, out, stats = jax.vmap(env.step)(es, acts, ks[:, 1])
        return (es, out), stats.done

    run = jax.jit(lambda c, ks: jax.lax.scan(one, c, ks))
    t0 = time.time()
    c, d = run((es, out), jax.random.split(jax.random.PRNGKey(1), T))
    jax.block_until_ready(d)
    t1 = time.time()
    c, d = run(c, jax.random.split(jax.random.PRNGKey(2), T))
    jax.block_until_ready(d)
    t2 = time.time()
    steps = B * T
    print(f"compile+first {t1 - t0:.1f}s; {steps} env-steps in {t2 - t1:.2f}s = {steps / (t2 - t1):.0f} steps/s; "
          f"episodes finished {int(np.asarray(d).sum())} (backend {jax.default_backend()}, batch {B})")


if __name__ == "__main__":
    main()
