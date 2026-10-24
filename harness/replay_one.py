"""Single-battle forced replay against a trace (unbatched; the batched pipeline is harness/replay.py, M2.8).

For each decision k: compare the engine projection with PS's (k >= 1), step with the decision's choices and its
forced draws/ties, then compare the draw multiset (contract §9).
"""
from __future__ import annotations

import collections
import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

from harness import forced as FO
from harness import project as P
from vgcjax.battle import errors as E
from vgcjax.battle import init as INIT
from vgcjax.battle import requests as REQ
from vgcjax.battle import state as S
from vgcjax.battle import step as STEP
from vgcjax.config import EngineConfig
from vgcjax.core import rng as R
from vgcjax.core.dex import load_dex
from vgcjax.interop import ps_choice
from vgcjax.interop.team import team_from_sets


MAX_CONT = 256   # step calls per decision (a turn needs at most a few dozen micro-steps)

# persistent XLA cache shared by the replay workers and runs (keyed by the HLO, so engine edits miss it)
import os  # noqa: E402
import pathlib  # noqa: E402
jax.config.update("jax_compilation_cache_dir", os.environ.get(
    "VGC_JAX_CACHE", str(pathlib.Path(__file__).resolve().parents[1] / ".jax_cache")))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


@functools.lru_cache(maxsize=None)
def compiled(cfg: EngineConfig):
    dex = load_dex(cfg.stage)

    @jax.jit
    def f(st, actions, key, fkeys, fvals, fvalid):
        forced = R.ForcedOutcomes(keys=fkeys, value=fvals, valid=fvalid)
        st2, rng = STEP.step(st, actions, key, cfg, dex, forced)
        return st2, rng.log

    @jax.jit
    def masks(st):
        return REQ.joint_legal(st, dex)
    return f, masks


def split_battles(records):
    cur = []
    for r in records:
        if r["type"] == "header" and cur:
            yield cur
            cur = []
        cur.append(r)
    if cur:
        yield cur


def engine_draws(log) -> list:
    n = int(log.count)
    out = []
    keys = np.asarray(log.keys)
    args = np.asarray(log.args)
    flags = np.asarray(log.flags)
    for i in range(min(n, keys.shape[0])):
        out.append(((int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])), (int(args[i, 0]), int(args[i, 1])),
                    bool(flags[i] & R.LOG_IS_TIE)))
    return out


def replay_battle(battle_records, cfg: EngineConfig, check_masks=False, start=None):
    """Returns dict(clean, k (first failing decision or None), category, detail).
    `start` = (engine state, k0): resume at decision k0 (e.g. from from_ps_snapshot)."""
    header = battle_records[0]
    decisions = [r for r in battle_records if r["type"] == "decision"]
    final = battle_records[-1] if battle_records[-1]["type"] == "final" else None
    step_fn, mask_fn = compiled(cfg)
    st = INIT.init(team_from_sets(header["team_sets"]))
    k0 = 0
    if start is not None:
        st, k0 = start
        st = jax.tree_util.tree_map(jnp.asarray, st)
    key = jax.random.PRNGKey(0)
    ext = None
    events = None
    events_checked = any(d.get("events") is not None for d in decisions)
    for k, dec in enumerate(decisions):
        if k < k0:
            continue
        if k >= 1:
            got = P.engine_to_json(st)
            d = P.compare(got, dec["projection"])
            if d:
                return dict(clean=False, k=k, category="STATE", detail=d[:10], ext=ext)
            if ext is None and "ext" in dec["projection"]:
                de = P.compare_ext(got, dec["projection"])
                if de:
                    ext = dict(k=k, detail=de[:5])
        if check_masks and dec.get("legal") and k >= 1:
            jl = np.asarray(mask_fn(st))
            for s in range(2):
                lg = dec["legal"][s]
                if lg.get("kind") in ("team",):
                    continue
                ps_set = {p[0] * 29 + p[1] for p in lg["pairs"]}
                unav = {p[0] * 29 + p[1] for p in lg["unavailable"]}
                eng = set(np.flatnonzero(jl[s]).tolist())
                if not (eng <= ps_set and ps_set - eng <= unav):
                    return dict(clean=False, k=k, category="MASK",
                                detail=dict(side=s, engine_only=sorted(eng - ps_set)[:10],
                                            ps_only=sorted(ps_set - eng - unav)[:10]))
        try:
            acts = [ps_choice.parse_choice(dec["choices"][s], dec["requests"][s],
                                           np.asarray(st.sides.ps_order[s]), dec["phase"]) for s in range(2)]
            fk, fv, fvalid = FO.forced_arrays([dec], cfg.kf)
            exp = FO.compared_draws(dec)
        except FO.HarnessError as e:
            return dict(clean=False, k=k, category="HARNESS", detail=str(e))
        # one decision may take several step calls (phase CONT, ADR-104 option 4): same forced table, draws pooled
        got = []
        a = jnp.asarray(acts, jnp.int32)
        for _ in range(MAX_CONT):
            st, log = step_fn(st, a, key, fk[0], fv[0], fvalid[0])
            got += engine_draws(log)
            if int(st.meta.phase) != S.CONT:
                break
        err = int(st.meta.error_flags)
        gm = collections.Counter((g[0], g[1]) for g in got)
        em = collections.Counter((e[0], e[1]) for e in exp)
        if gm != em:
            gk = collections.Counter(g[0] for g in got)
            ek = collections.Counter(e[0] for e in exp)
            cat = "DRAW_ARGS" if gk == ek else "DRAW_MISMATCH"
            from vgcjax.core import drawkey as dk
            extra_e = [(dk.unpack(*kk), a) for (kk, a) in (em - gm)]
            extra_g = [(dk.unpack(*kk), a) for (kk, a) in (gm - em)]
            return dict(clean=False, k=k, category=cat, detail=dict(ps_only=extra_e[:5], engine_only=extra_g[:5],
                                                                   err=E.describe(err)))
        if err:
            return dict(clean=False, k=k, category="ERROR", detail=E.describe(err))
        if events is None and dec.get("events") is not None:
            from harness import events as HE
            got_ev, overflow = HE.engine_events(st)
            de = None if overflow else HE.compare(got_ev, dec["events"])
            if de:
                events = dict(k=k, detail=de)
    if final is not None:
        d = P.compare(P.engine_to_json(st), final["projection"])
        if d:
            return dict(clean=False, k=len(decisions), category="STATE", detail=d[:10], events=events,
                        events_checked=events_checked)
    return dict(clean=True, k=None, category=None, detail=None, ext=ext, events=events, events_checked=events_checked)


def main():
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import subprocess
    import orjson
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    raw = subprocess.run(["zstd", "-dc", path], capture_output=True, check=True).stdout.decode()
    recs = [orjson.loads(l) for l in raw.splitlines() if l]
    cfg = EngineConfig(stage="core", forced=True, draw_log=True)
    cats = collections.Counter()
    for i, b in enumerate(split_battles(recs)):
        if i >= n:
            break
        r = replay_battle(b, cfg, check_masks=True)
        cats[r["category"] or "CLEAN"] += 1
        if not r["clean"] and cats[r["category"]] <= 3:
            print(i, r["k"], r["category"], r["detail"])
    print(dict(cats))


if __name__ == "__main__":
    main()
