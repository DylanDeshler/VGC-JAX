"""Trace draw/tie records -> engine ForcedOutcomes and comparable keys (07 §4.2, contract §5, §9)."""
from __future__ import annotations

import math

import numpy as np

from vgcjax.core import drawkey as dk
from vgcjax.data import ids

DK = {name: getattr(ids, f"DK_{name}") for name in ids.DRAW_KINDS}
_IDX = {name: {x: i for i, x in enumerate(getattr(ids, name))} for name in ("MOVES", "ABILITIES", "ITEMS", "CONDITIONS")}
MOVE_KINDS = {"ACCURACY", "CRIT", "DAMAGE_ROLL", "SECONDARY", "SECONDARY_STATUS_PICK", "SELF_EFFECT_ROLL",
              "MULTIHIT_COUNT", "RANDOM_TARGET", "PROTECT_STALL", "MOVE_FICKLE_BEAM", "MOVE_SHELLSIDEARM_TIE",
              "MOVE_ACUPRESSURE_STAT", "MOVE_CALL_PICK", "FORCED_SWITCH_PICK"}
COND_KINDS = {"CONFUSION_SELFHIT_DAMAGE_ROLL", "PARALYSIS", "SLEEP_DURATION", "FREEZE_THAW", "CONFUSION_DURATION",
              "CONFUSION_SELFHIT", "PARTIAL_TRAP_DURATION", "LOCKEDMOVE_DURATION", "ALLY_SWITCH_STALL", "INFATUATION"}


import json as _json
import pathlib as _pl

_CONTRACT = _pl.Path(__file__).resolve().parent / "contract"
_DC = _json.loads((_CONTRACT / "dontcare.json").read_text())
DONTCARE_KINDS = set(_DC["kinds"])
DONTCARE_IF_NO_CHANCE = set(_DC["if_chance_undefined"])


def is_dontcare(d: dict) -> bool:
    """contract/drawkey.md §6 via the protected harness/contract/dontcare.json (oracle flags are not trusted)."""
    k = d["kind"]
    return k in DONTCARE_KINDS or (k in DONTCARE_IF_NO_CHANCE and d.get("chance", 0) is None)


def is_degenerate(d: dict) -> bool:
    """contract §7 (harness/contract/degenerate.json)."""
    a = d["args"]
    if d["method"] == "randomChance":
        return a[0] >= a[1] or a[0] <= 0
    if d["method"] == "sample":
        return a[0] <= 1
    if d["method"] == "random":
        return (len(a) == 1 and a[0] <= 1) or (len(a) == 2 and a[1] - a[0] <= 1)
    return False


class HarnessError(Exception):
    """Trace contents the harness cannot map (category HARNESS, 07 §10.1)."""


def eff(space: str, name: str, base: int) -> int:
    i = _IDX[space].get(name)
    if i is None or i == 0:
        raise HarnessError(f"unknown {space} id {name!r}")
    return base + i - 1


def subject_effect(kind: str, s) -> int:
    if s is None:
        return -1
    order = (["MOVES", "CONDITIONS", "ITEMS", "ABILITIES"] if kind in MOVE_KINDS or kind == "ITEM_FOCUS_BAND" else
             ["CONDITIONS", "MOVES"] if kind in COND_KINDS else
             ["ITEMS", "ABILITIES"] if kind == "FRAC_PRIO" else ["ABILITIES", "ITEMS", "MOVES", "CONDITIONS"])
    bases = {"MOVES": ids.MOVE_BASE, "ABILITIES": ids.ABILITY_BASE, "ITEMS": ids.ITEM_BASE, "CONDITIONS": ids.COND_BASE}
    for sp in order:
        if s in _IDX[sp] and _IDX[sp][s] > 0:
            return bases[sp] + _IDX[sp][s] - 1
    raise HarnessError(f"unmapped subject {s!r} for {kind}")


def step_subject(s) -> int:
    if s is None:
        return -1
    if s in ids.ACTION_KINDS:
        return ids.ACTION_BASE + ids.ACTION_KINDS.index(s)
    return eff("MOVES", s, ids.MOVE_BASE)


def draw_key(d: dict) -> dk.DrawKey:
    k = d["key"]
    kind = d["kind"]
    return dk.DrawKey(kind=DK[kind], turn=k["turn"], step_tag=k["step_tag"], step_actor=k["step_actor"],
                      step_subject=step_subject(k["step_subject"]), actor=k["actor"], target=k["target"],
                      subject=subject_effect(kind, k["subject"]), hit=k["hit"], ord=k["ord"], sub=k["sub"],
                      action_ord=k["action_ord"])


def tie_key(t: dict) -> dk.DrawKey:
    k = t["key"]
    return dk.DrawKey(kind=DK[t["kind"]], turn=k["turn"], step_tag=k["step_tag"], step_actor=0, step_subject=0,
                      actor=0, target=0, subject=0, hit=0, ord=k["ord"], sub=k["sub"], action_ord=0)


def draw_value(d: dict) -> int:
    v = d["value"]
    if d["method"] == "randomChance":
        return int(bool(v))
    if d["method"] == "sample" and d["kind"] in ("RANDOM_TARGET", "FORCED_SWITCH_PICK") and "pick" in d:
        return int(d["pick"])
    return int(v)


def forced_rows(decision: dict) -> list[tuple[tuple[int, int, int], int]]:
    """[(packed key, value)] for every compared (non-don't-care) draw and every tie record of a decision."""
    rows = []
    for d in decision.get("draws", []):
        if is_dontcare(d):
            continue
        rows.append((dk.pack(draw_key(d)), draw_value(d)))
    for t in decision.get("ties", []):
        rows.append((dk.pack(tie_key(t)), int(t["value"])))
    return rows


def forced_arrays(decisions: list[dict], kf: int):
    """Per-decision forced tables stacked: keys u32[K, kf, 3], value i32[K, kf], valid bool[K, kf]."""
    n = len(decisions)
    keys = np.zeros((n, kf, 3), np.uint32)
    vals = np.zeros((n, kf), np.int32)
    valid = np.zeros((n, kf), bool)
    for i, dec in enumerate(decisions):
        rows = forced_rows(dec)
        if len(rows) > kf:
            raise HarnessError(f"decision {i}: {len(rows)} forced rows > KF={kf}")
        for j, (c, v) in enumerate(rows):
            keys[i, j] = c
            vals[i, j] = v
            valid[i, j] = True
    return keys, vals, valid


def compared_draws(decision: dict):
    """Multiset of (packed key, args) the engine must reproduce: non-don't-care, non-degenerate draws + ties."""
    out = []
    for d in decision.get("draws", []):
        if is_dontcare(d) or is_degenerate(d):
            continue
        args = d["args"]
        a = (int(args[0]), int(args[1]) if len(args) > 1 else 0)
        if d["method"] == "randomChance":
            # ADR-105: random(d) < n with integer rolls, so a fractional n (multi-accuracy stage table: 67.5)
            # is exactly "< ceil(n)", the bound the engine draws with (sim/prng.ts:91-117)
            a = (math.ceil(args[0]), a[1])
        if d["method"] == "random" and len(args) == 1:
            a = (0, int(args[0]))  # random(n) == random(0, n)
        out.append((dk.pack(draw_key(d)), a, d["kind"]))
    for t in decision.get("ties", []):
        out.append((dk.pack(tie_key(t)), (len(t["tied"]), 0), t["kind"]))
    return out
