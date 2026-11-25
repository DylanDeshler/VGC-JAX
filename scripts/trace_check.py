"""Trace-only check of the engine step (no XLA compile): catches Python / shape / dtype errors in new content quickly
and with little memory, so code can be checked without a full replay (which compiles the engine: ~15 GB per worker).

    PYTHONPATH=. uv run python scripts/trace_check.py [--stage impl]
"""
import argparse
import resource
import time

import jax

from vgcjax.config import EngineConfig
from vgcjax.core.dex import load_dex
from vgcjax.env import api, teambank as TB
from vgcjax.env.autoreset import AutoResetEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="impl")
    ap.add_argument("--bank", default="data/teams/bank_core.npz")
    a = ap.parse_args()
    cfg = EngineConfig(stage=a.stage, max_turns=40, events=True)
    dex = load_dex(a.stage)
    env = AutoResetEnv(cfg, dex, TB.load(a.bank))
    t0 = time.time()
    es, out = jax.eval_shape(env.reset, jax.random.PRNGKey(0))

    def step(es, k):
        masks = api.legal_masks(es.state, cfg, dex)
        acts = api.random_actions(k, masks, es.state.meta.phase)
        return env.step(es, acts, k)

    jax.eval_shape(step, es, jax.random.PRNGKey(1))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)
    print(f"trace_check ok: {time.time() - t0:.1f}s, peak rss {rss:.1f} GB")


if __name__ == "__main__":
    main()
