"""Side-by-side trace of one battle: PS vs engine per decision (projection keys, public events).

    uv run python -m harness.debug <trace dir> <battle index> [--keys field,sides.0.active] [--events]

Prints, for each decision k, the selected projection subtrees that differ (or all with --all) and, with --events,
the PS and engine PublicEvents of the step that led to k.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np


def _get(d, path):
    for p in path.split("."):
        if d is None:
            return None
        d = d[int(p)] if isinstance(d, list) else d.get(p)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("index", type=int)
    ap.add_argument("--keys", default="field")
    ap.add_argument("--events", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stage", default="impl")
    a = ap.parse_args()
    from harness import cli
    from harness import events as HE
    from harness import forced as FO
    from harness import project as P
    from harness.replay_one import MAX_CONT, compiled
    from vgcjax.battle import init as INIT
    from vgcjax.battle import state as S
    from vgcjax.interop import ps_choice
    from vgcjax.interop.team import team_from_sets
    files = sorted(pathlib.Path(a.dir).glob("*.ndjson.zst"))
    recs = cli._battle_at(files, a.index)
    header = recs[0]
    decisions = [r for r in recs if r["type"] == "decision"]
    cfg = cli._cfg(a.stage)
    step_fn, _ = compiled(cfg)
    st = INIT.init(team_from_sets(header["team_sets"]))
    key = jax.random.PRNGKey(0)
    keys = a.keys.split(",")
    for k, dec in enumerate(decisions):
        if k >= 1:
            got = P.engine_to_json(st)
            for kk in keys:
                g, p = _get(got, kk), _get(dec["projection"], kk)
                if a.all or g != p:
                    print(f"k={k} turn={dec['turn']} {kk}\n  PS : {json.dumps(p, sort_keys=True)}\n  ENG: "
                          f"{json.dumps(g, sort_keys=True)}")
            prev = decisions[k - 1]
            if a.events and prev.get("events") is not None:      # a decision's events: the step it starts
                ge, _ = HE.engine_events(st)
                print(f"k={k} events PS : {[(e['kind'], e.get('actor'), e.get('move')) for e in prev['events']]}")
                print(f"k={k} events ENG: {[(e['kind'], e.get('actor'), e.get('move')) for e in ge]}")
        acts = [ps_choice.parse_choice(dec["choices"][s], dec["requests"][s], np.asarray(st.sides.ps_order[s]),
                                       dec["phase"]) for s in range(2)]
        print(f"k={k} choices {dec['choices']}")
        fk, fv, fvalid = FO.forced_arrays([dec], cfg.kf)
        act = jnp.asarray(acts, jnp.int32)
        for _ in range(MAX_CONT):
            st, _log = step_fn(st, act, key, fk[0], fv[0], fvalid[0])
            if int(st.meta.phase) != S.CONT:
                break


if __name__ == "__main__":
    main()
