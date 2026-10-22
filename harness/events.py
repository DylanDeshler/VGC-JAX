"""PublicEvents conformance (03 §4.7): engine event records of one step vs PS's protocol reduced by
oracle/lib/protocol.js (trace records generated with `oracle.js gen --events`). Non-gating, reported like `ext`.

Normalisation: the target printed on a MOVE line of a move whose target PS resolved by a don't-care random pick
(spread / side / field targets: resolveAction's getRandomTarget, contract §6) is not compared.
"""
from __future__ import annotations

import numpy as np

from vgcjax.battle import events as EV
from vgcjax.battle import state as S
from vgcjax.data import ids

RANDOM_LOC_TARGETS = {"allAdjacentFoes", "allAdjacent", "foeSide", "allySide", "allyTeam", "all", "allies", "scripted"}
FIELDS = ("actor", "target", "move", "v1", "v2", "flags")


def engine_events(st) -> tuple[list[dict], bool]:
    """Records written during the last step (oldest first) and whether the ring overflowed."""
    ev = st.events
    count, start = int(ev.count), int(ev.step_start)
    buf = np.asarray(ev.buf)
    n = count - start
    out = []
    for a in range(max(start, count - S.NE), count):
        t, kind, actor, target, move, v1, v2, flags = (int(x) for x in buf[a % S.NE])
        out.append(dict(kind=EV.KINDS[kind], actor=actor, target=target,
                        move=ids.MOVES[move] if 0 < move < len(ids.MOVES) else "", v1=v1, v2=v2, flags=flags))
    return out, n > S.NE


def _norm(e: dict) -> tuple:
    e = dict(e)
    if e["kind"] == "MOVE" and e["move"] in ids.MOVES:
        from vgcjax.core.dex import load_tables_np
        tt = ids.TARGETS[int(load_tables_np()["move_target"][ids.MOVES.index(e["move"])])]
        if tt in RANDOM_LOC_TARGETS:
            e["target"] = "random"
    return (e["kind"],) + tuple(e.get(f, 0) for f in FIELDS)


def compare(engine: list[dict], ps: list[dict]):
    """None if equal, else dict(i = first difference, engine / ps records, context, bucket). Bucket
    "SWITCH_ORDER(ADR-103)": the sequences differ only by the order of SWITCH records (an unrecorded speed tie
    between replacement switches, sorted by PS's commitChoices queue.sort)."""
    a = [_norm(e) for e in engine]
    b = [_norm(e) for e in ps]
    d = _first_diff(a, b)
    if d is None:
        return None
    sw_a = [x for x in a if x[0] == "SWITCH"]
    sw_b = [x for x in b if x[0] == "SWITCH"]
    rest_equal = [x for x in a if x[0] != "SWITCH"] == [x for x in b if x[0] != "SWITCH"]
    if rest_equal and sorted(sw_a) == sorted(sw_b):
        d["bucket"] = "SWITCH_ORDER(ADR-103)"
    else:
        d["bucket"] = f"{d['engine'][0] if d['engine'] else None}/{d['ps'][0] if d['ps'] else None}"
    return d


def _first_diff(a, b):
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x != y:
            return dict(i=i, engine=x, ps=y, engine_ctx=a[max(0, i - 3):i + 2], ps_ctx=b[max(0, i - 3):i + 2])
    return None
