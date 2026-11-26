"""Count primitive ops in the traced engine step (no XLA compile; ~4 GB). Compile memory scales with this count.
    PYTHONPATH=. uv run python scripts/op_count.py [--stage impl]
"""
import argparse, collections
import jax
from vgcjax.config import EngineConfig
from vgcjax.core.dex import load_dex
from vgcjax.env import api, teambank as TB
from vgcjax.env.autoreset import AutoResetEnv


def walk(jaxpr, c):
    for e in jaxpr.eqns:
        c[e.primitive.name] += 1
        for p in e.params.values():
            for sub in (p if isinstance(p, (list, tuple)) else [p]):
                if hasattr(sub, "jaxpr"):
                    walk(sub.jaxpr if hasattr(sub.jaxpr, "eqns") else sub.jaxpr.jaxpr, c)
                elif hasattr(sub, "eqns"):
                    walk(sub, c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="impl"); a = ap.parse_args()
    cfg = EngineConfig(stage=a.stage, max_turns=40, events=True)
    dex = load_dex(a.stage)
    env = AutoResetEnv(cfg, dex, TB.load("data/teams/bank_core.npz"))
    es, _ = jax.eval_shape(env.reset, jax.random.PRNGKey(0))

    def step(es, k):
        masks = api.legal_masks(es.state, cfg, dex)
        return env.step(es, api.random_actions(k, masks, es.state.meta.phase), k)

    c = collections.Counter()
    walk(jax.make_jaxpr(step)(es, jax.random.PRNGKey(1)).jaxpr, c)
    print(f"total ops {sum(c.values())}  cond {c['cond']}  while {c['while']}  scan {c['scan']}  gather {c['gather']}  "
          f"scatter {c['scatter'] + c['scatter-add']}  select_n {c['select_n']}")


if __name__ == "__main__":
    main()
