"""Fixture shrinking (07 §10.2 step 2): greedy reduction of a turn-isolated failing fixture.

Candidate edits come from the oracle (`shrink_candidates`: unrelated moves -> a neutral stage move, remove items,
core-inert abilities, faint bench mons, HP to max; moves chosen at the decision are protected). Each candidate is
applied in PS (`edit_snapshot`, through PS APIs), decision k is re-run in PS with the recorded choices and the
fixture's draws/ties forced (`replay` with force_dontcare; keys that no longer occur are simply not drawn, new
draws come from the snapshot's PRNG and are recorded), and the engine replays the new fixture. The candidate is kept
iff the engine still fails with the same signature. Rounds repeat until no candidate is kept.

Field and side conditions are not removed yet: no core-stage entity can create one (ADR-102 probe); the edit lands
with the first v0 condition.
"""
from __future__ import annotations

import copy

from harness.fixtures import replay_fixture, signature
from harness.oracle_client import OracleError, default_client


def _apply_team_sets(team_sets, ident, edit, neutral):
    ts = copy.deepcopy(team_sets)
    row = ts[edit["side"]][ident[edit["side"]][edit["b"]]["sheet"]]
    if edit["op"] == "move":
        row["moves"][edit["slot"]] = neutral
    elif edit["op"] == "no_item":
        row["item"] = ""
    elif edit["op"] == "ability":
        row["ability"] = edit["value"]
    return ts


def try_edit(fx: dict, edit: dict, neutral: str | None, client=None) -> dict | None:
    """The fixture after one edit, re-recorded in PS; None if PS rejects the edit or the recorded choices."""
    client = client or default_client()
    dec = fx["decision"]
    try:
        e = client.call("edit_snapshot", {"snapshot": fx["snapshot"], "edits": [edit]})
        if e["skipped"]:
            return None
        r = client.call("replay", {"snapshot": e["snapshot"], "choices": [dec["choices"]], "draws": dec["draws"],
                                   "ties": dec["ties"], "force_dontcare": True, "until": 1, "project_after": True})
    except OracleError:
        return None
    if not r["decisions"] or r.get("rejected"):
        return None
    nd = r["decisions"][0]
    out = dict(fx, snapshot=e["snapshot"], expected=r["projection_after"],
               team_sets=_apply_team_sets(fx["team_sets"], fx["snapshot"]["ident"], edit, neutral),
               decision={key: nd[key] for key in ("phase", "turn", "requests", "choices", "draws", "ties", "projection")})
    return out


def shrink(fx: dict, cfg=None, client=None, max_rounds: int = 4, log=None) -> dict:
    """Greedy reduction; returns the smallest fixture found (same signature), with `shrink_log` listing kept edits."""
    client = client or default_client()
    r0 = replay_fixture(fx, cfg)
    if r0["clean"]:
        raise ValueError("fixture does not reproduce: nothing to shrink")
    sig = signature(r0)
    kept = []
    cur = fx
    for _ in range(max_rounds):
        c = client.call("shrink_candidates", {"snapshot": cur["snapshot"], "choices": cur["decision"]["choices"],
                                              "stage": cur.get("stage", "core")})
        neutral = c["neutral_move"]
        changed = False
        for edit in c["candidates"]:
            cand = try_edit(cur, edit, neutral, client)
            if cand is None:
                continue
            r = replay_fixture(cand, cfg)
            if not r["clean"] and signature(r) == sig:
                cur = dict(cand, detail=r.get("detail"), neutral_move=neutral)
                kept.append(edit)
                changed = True
                if log:
                    log(f"kept {edit}")
        if not changed:
            break
    return dict(cur, signature=sig, shrink_log=kept)
