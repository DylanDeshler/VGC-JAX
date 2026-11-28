"""Synthetic check of the belief filter (vgcjax/agent/belief.py): a true opponent set (prior skeleton k*, random SP)
in a duel; observations generated from the true board by the calculator (exact engine damage); after each update the
true skeleton must keep positive probability and the posterior on the truth should rise.

    MEM_CAP_GB=10 bash scripts/memguard.sh env PYTHONPATH=. uv run python scripts/agent/check_belief.py --n 20
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from vgcjax.agent import belief as BL
from vgcjax.agent import calc as CA
from vgcjax.battle.util import mget
from vgcjax.config import EngineConfig
from vgcjax.core.dex import load_dex
from vgcjax.data import ids
from scripts.agent.check_calc import rand_sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--priors", default="data/priors/set_priors_bo1.npz")
    a = ap.parse_args()
    pri_np = np.load(a.priors)
    pri = BL.load_priors(a.priors)
    dex = load_dex("impl")
    cfg = EngineConfig(stage="impl", events=False)
    rng = np.random.default_rng(a.seed)
    species = [s for s in range(len(ids.SPECIES)) if pri_np["skel_n"][s] >= 4]

    def prior_sp(s):
        """A spread drawn from the species' per-stat SP marginals (realistic, unlike a uniform random spread)."""
        while True:
            sp = np.array([rng.choice(33, p=pri_np["sp_marg"][s, j] / pri_np["sp_marg"][s, j].sum()) for j in range(6)])
            if sp.sum() <= 66:
                return sp

    def set_of(s, k):
        return dict(species=s, item=int(pri_np["skel_item"][s, k]), ability=int(pri_np["skel_ability"][s, k]),
                    nature=int(pri_np["skel_nature"][s, k]), gender=0, moves=pri_np["skel_moves"][s, k].astype(np.int64),
                    sp=prior_sp(s))

    @jax.jit
    def observe_taken(team, move, roll):          # the opponent (ref 4) hits us (ref 0)
        st = CA.duel_state(dex, team)
        vals, _ = CA.damage_rolls(st, dex, cfg, move, False, u=CA.DEF, t=CA.ATK)
        return vals[roll], st

    @jax.jit
    def upd_taken(b, st, move, obs):
        return BL.damage_taken(b, pri, st, dex, cfg, 0, CA.DEF, CA.ATK, move, False, obs)

    stats = dict(trials=0, true_alive=0, true_up=0, sp_true_alive=0)
    for _ in range(a.n):
        so, su = int(rng.choice(species)), int(rng.choice(species))
        k_true = int(rng.integers(0, pri_np["skel_n"][so]))
        opp, us = set_of(so, k_true), set_of(su, 0)
        dmg = [m for m in opp["moves"] if m > 0 and dex.move_bp[m] > 0 and int(dex.move_category[m]) != ids.CAT_STATUS]
        if not dmg:
            continue
        move = int(rng.choice(dmg))
        team = CA.team_arrays(us, opp)
        obs, st = observe_taken(team, jnp.asarray(move), jnp.asarray(int(rng.integers(0, 16))))
        if int(obs) <= 0:
            continue
        b0 = BL.init(pri, jnp.asarray([so] * 6))
        p0 = float(BL.skel_probs(b0)[0, k_true])
        b1 = upd_taken(b0, st, jnp.asarray(move), obs)
        p1 = float(BL.skel_probs(b1)[0, k_true])
        stat = int(BL.off_stat(dex, jnp.asarray(move)))
        sp_true = float(b1.sp[0, stat, int(opp["sp"][stat])])
        stats["trials"] += 1
        stats["true_alive"] += p1 > 0
        stats["true_up"] += p1 >= p0 - 1e-6
        stats["sp_true_alive"] += sp_true > 1e-5
        print(f"{ids.SPECIES[so]} k*={k_true} {ids.MOVES[move]} dmg={int(obs)}: P(k*) {p0:.3f} -> {p1:.3f}, "
              f"P(sp*={int(opp['sp'][stat])}) = {sp_true:.3f}, off_prior={bool(b1.off_prior[0])}", flush=True)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
