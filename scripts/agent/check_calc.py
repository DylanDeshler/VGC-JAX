"""Check the engine-as-calculator (vgcjax/agent/calc.py) against PS's own getDamage via the oracle 'damage' RPC.

Cases are sampled from the Bo1 set priors (data/priors/set_priors_bo1.npz): attacker / defender = a species' prior
skeleton with a random SP spread, move = one of the attacker skeleton's damaging moves, crit on/off; every roll 0..15
is compared. Expected values come from PS only (CLAUDE.md).

    MEM_CAP_GB=10 bash scripts/memguard.sh env PYTHONPATH=. uv run python scripts/agent/check_calc.py --n 50
"""
import argparse
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from harness.oracle_client import OracleClient
from vgcjax.agent import calc as CA
from vgcjax.config import EngineConfig
from vgcjax.core.dex import load_dex
from vgcjax.data import ids


def rand_sp(rng):
    """A random legal Champions spread: <= 32 per stat, <= 66 total."""
    sp = np.zeros(6, np.int64)
    budget = int(rng.integers(0, 67))
    for _ in range(budget):
        s = int(rng.integers(0, 6))
        if sp[s] < 32:
            sp[s] += 1
    return sp


def packed(x):
    mv = ",".join(ids.MOVES[m] for m in x["moves"] if m > 0)
    return (f"{ids.SPECIES[x['species']]}||{ids.ITEMS[x['item']] if x['item'] else ''}|{ids.ABILITIES[x['ability']]}|"
            f"{mv}|{ids.NATURES[x['nature']]}|{','.join(str(int(v)) for v in x['sp'])}|M|||50|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--priors", default="data/priors/set_priors_bo1.npz")
    a = ap.parse_args()
    pri = np.load(a.priors)
    dex = load_dex("impl")
    cfg = EngineConfig(stage="impl", events=False)
    rng = np.random.default_rng(a.seed)
    species = [s for s in range(len(ids.SPECIES)) if pri["skel_n"][s] > 0]
    # the PS 'damage' RPC starts a real doubles battle (filler = the attacker set on both sides), so switch-in
    # abilities (Intimidate, weather / terrain setters, ...) act there but not in the hook-free duel: skip them
    START = {"intimidate", "drought", "drizzle", "sandstream", "snowwarning", "electricsurge", "psychicsurge",
             "grassysurge", "mistysurge", "hospitality", "curiousmedicine", "download", "intrepidsword",
             "dauntlessshield", "supersweetsyrup", "seedsower", "airlock", "cloudnine", "unnerve", "pressure",
             "frisk", "anticipation", "forewarn", "trace", "imposter", "illusion", "megasol", "orichalcumpulse",
             "hadronengine", "commander", "zerotohero", "protosynthesis", "quarkdrive", "slowstart", "costar"}
    start_ab = {ids.ABILITIES.index(x) for x in START if x in ids.ABILITIES}

    def sample_set(s):
        ks = [k for k in range(pri["skel_n"][s]) if int(pri["skel_ability"][s, k]) not in start_ab]
        if not ks:
            return None
        k = int(rng.choice(ks))
        return dict(species=s, item=int(pri["skel_item"][s, k]), ability=int(pri["skel_ability"][s, k]),
                    nature=int(pri["skel_nature"][s, k]), gender=0, moves=pri["skel_moves"][s, k].astype(np.int64),
                    sp=rand_sp(rng))

    @jax.jit
    def eng(team, move, crit):
        st = CA.duel_state(dex, team)
        return CA.damage_rolls(st, dex, cfg, move, crit)

    cases, metas = [], []
    while len(cases) < a.n * 16:
        A_, D_ = sample_set(int(rng.choice(species))), sample_set(int(rng.choice(species)))
        if A_ is None or D_ is None:
            continue
        dmg_moves = [m for m in A_["moves"] if m > 0 and dex.move_bp[m] > 0]
        if not dmg_moves:
            continue
        move = int(rng.choice(dmg_moves))
        crit = bool(rng.random() < 0.25)
        weather = str(rng.choice(["", "", "raindance", "sunnyday"]))
        for r in range(16):
            c = dict(attacker=packed(A_), defender=packed(D_), move=ids.MOVES[move], crit=crit, roll=r, modify=True,
                     filler="Magikarp|||SwiftSwim|Splash|Hardy||M|||50|")
            if weather:
                c["weather"] = weather
            cases.append(c)
        metas.append((A_, D_, move, crit, weather))
    t0 = time.time()
    ps = OracleClient().call("damage", {"cases": cases})
    print(f"oracle: {len(cases)} cases in {time.time() - t0:.1f}s", flush=True)
    bad = 0
    t0 = time.time()
    for i, (A_, D_, move, crit, weather) in enumerate(metas):
        if weather:                                  # weather cases need the field set in the duel: skip for now
            continue
        team = CA.team_arrays(A_, D_)
        vals, kind = eng(team, jnp.asarray(move), jnp.asarray(crit))
        mine = [int(v) for v in np.asarray(vals)]
        theirs = [ps[i * 16 + r] for r in range(16)]
        theirs_n = [t if t is not None else 0 for t in theirs]
        if mine != theirs_n:
            bad += 1
            if bad <= 15:
                print(f"MISMATCH {ids.SPECIES[A_['species']]}({ids.ITEMS[A_['item']]},{ids.ABILITIES[A_['ability']]}) "
                      f"{ids.MOVES[move]} -> {ids.SPECIES[D_['species']]}({ids.ITEMS[D_['item']]},"
                      f"{ids.ABILITIES[D_['ability']]}) crit={crit}: engine {mine[:4]}.. ps {theirs[:4]}..")
        if i == 0:
            print(f"first engine call (compile) {time.time() - t0:.1f}s", flush=True)
    n_checked = sum(1 for m in metas if not m[4])
    print(json.dumps(dict(checked=n_checked, mismatched=bad, ok=n_checked - bad)))


if __name__ == "__main__":
    main()
