"""Two-sample distribution checks (07 §11): engine natural-mode RNG vs PS on the same states.

A scenario is a PS snapshot at a decision plus that decision's recorded choices. PS continues it n times with fresh
PRNG seeds (oracle `sample_outcomes`); the engine steps the same state n times under distinct natural keys (vmap).
Each brought mon's post-step HP is compared with a chi-square contingency test (categories with < `min_count`
pooled samples merged); callers apply Bonferroni over all (scenario, mon) tests.
"""
from __future__ import annotations

import functools
import pathlib
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import orjson
from scipy import stats as sstats

from harness.oracle_client import default_client
from harness.replay_one import split_battles
from vgcjax.battle import state as S
from vgcjax.battle import step as STEP
from vgcjax.battle.snapshot import from_ps_snapshot
from vgcjax.config import EngineConfig
from vgcjax.core.dex import load_dex
from vgcjax.interop import ps_choice

ROOT = pathlib.Path(__file__).resolve().parent.parent


def scenarios(n: int, seed: str, stage: str = "core", teams: str = "stage:core", per_battle: int = 1) -> list[dict]:
    """n move-phase decisions (turn >= 1) from fresh oracle battles: {snapshot, team_sets, decision}."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["node", str(ROOT / "oracle/bin/oracle.js"), "gen", "--n", str(4 * n), "--seed", seed,
                        "--workers", "4", "--teams", teams, "--out", d], check=True, capture_output=True)
        recs = []
        for f in sorted(pathlib.Path(d).glob("*.ndjson.zst")):
            recs += [orjson.loads(l) for l in subprocess.run(["zstd", "-dc", str(f)], capture_output=True,
                                                              check=True).stdout.splitlines() if l]
    out = []
    client = default_client()
    for b in split_battles(recs):
        h = b[0]
        decs = [r for r in b if r["type"] == "decision"]
        picks = [k for k, dd in enumerate(decs) if dd["phase"] == "move" and dd["turn"] >= 1][:per_battle]
        for k in picks:
            snap = client.call("snapshot", {"teams": h["teams"], "seed": h["seed"],
                                            "choices": [x["choices"] for x in decs], "at": k})["snapshot"]
            out.append(dict(snapshot=snap, team_sets=h["team_sets"], decision=decs[k], stage=stage))
            if len(out) >= n:
                return out
    return out


def ps_samples(sc: dict, n: int, seed_base: str = "dist") -> np.ndarray:
    r = default_client().call("sample_outcomes", {"snapshot": sc["snapshot"], "choices": sc["decision"]["choices"],
                                                  "n": n, "seed_base": seed_base})
    return np.asarray(r["hp"], np.int64)                       # [n, 2, 4]


@functools.lru_cache(maxsize=None)
def _vstep(cfg: EngineConfig):
    dex = load_dex(cfg.stage)
    return jax.jit(jax.vmap(lambda st, a, k: STEP.step(st, a, k, cfg, dex)[0], in_axes=(0, None, 0)))


def engine_samples(sc: dict, n: int, cfg: EngineConfig | None = None, seed: int = 0) -> np.ndarray:
    cfg = cfg or EngineConfig(stage=sc.get("stage", "core"))
    st = jax.tree_util.tree_map(jnp.asarray, from_ps_snapshot(sc["snapshot"], sc["team_sets"]))
    dec = sc["decision"]
    acts = jnp.asarray([ps_choice.parse_choice(dec["choices"][s], dec["requests"][s], np.asarray(st.sides.ps_order[s]),
                                               dec["phase"]) for s in range(2)], jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    f = _vstep(cfg)
    sts = f(jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), st), acts, keys)
    for i in range(64):                       # continuation calls (phase CONT) until the decision is reached
        if not bool(jnp.any(sts.meta.phase == S.CONT)):
            break
        sts = f(sts, acts, jax.vmap(lambda k: jax.random.fold_in(k, i + 1))(keys))
    return np.asarray(sts.mons.hp, np.int64)   # [n, 2, 4]


def compare(ps: np.ndarray, eng: np.ndarray, min_count: int = 10) -> list[tuple[int, int, float]]:
    """[(side, b, p-value)] for every mon whose HP varies in either sample."""
    out = []
    for s in range(2):
        for b in range(4):
            x, y = ps[:, s, b], eng[:, s, b]
            vals, inv = np.unique(np.concatenate([x, y]), return_inverse=True)
            if len(vals) <= 1:
                continue
            cx = np.bincount(inv[:len(x)], minlength=len(vals))
            cy = np.bincount(inv[len(x):], minlength=len(vals))
            big = (cx + cy) >= min_count
            rows = [np.append(cx[big], cx[~big].sum()), np.append(cy[big], cy[~big].sum())]
            tab = np.array(rows)
            tab = tab[:, tab.sum(axis=0) > 0]
            if tab.shape[1] <= 1:
                continue
            out.append((s, b, float(sstats.chi2_contingency(tab).pvalue)))
    return out
