"""PS choice strings <-> engine actions (ADR-018, 04 §4.3; canonicalisation per 07 §6)."""
from __future__ import annotations

import itertools

import numpy as np

from vgcjax.data import ids

PREVIEW_INDEX = {p: i for i, p in enumerate(itertools.permutations(range(6), 4))}
CHOOSABLE = {ids.TARGET_NORMAL, ids.TARGET_ANY, ids.TARGET_ADJACENTALLY, ids.TARGET_ADJACENTALLYORSELF,
             ids.TARGET_ADJACENTFOE}


def preview_action(choice: str) -> int:
    """'team 3412' -> index into PREVIEW_ORDERS (ordered 4-tuples of sheet indices)."""
    digits = [int(c) - 1 for c in choice.split()[1]]
    used = list(digits[:4])
    i = 0
    while len(used) < 4:  # PS pads with the lowest unused indices (side.ts chooseTeam)
        if i not in used:
            used.append(i)
        i += 1
    return PREVIEW_INDEX[tuple(used)]


def slot_action(part: str, slot: int, req_active: dict | None, ps_order_side) -> int:
    """One slot's PS choice -> engine slot action (0..28)."""
    part = part.strip()
    if part in ("pass", "skip"):
        return 28
    toks = part.split()
    if toks[0] == "switch":
        return 24 + int(ps_order_side[int(toks[1]) - 1])
    assert toks[0] == "move", part
    mi = int(toks[1]) - 1
    mega = "mega" in toks[2:]
    loc = None
    for t in toks[2:]:
        if t.lstrip("+-").isdigit():
            loc = int(t)
    moves = req_active["moves"] if req_active else []
    mv = moves[mi] if mi < len(moves) else {}
    locked = len(moves) == 1 and ("target" not in mv or mv.get("id") == "struggle")
    if locked:
        return 0
    if loc is None:
        t = 0
    else:
        target = mv.get("target")
        if target == "adjacentAllyOrSelf":
            t = 0 if loc == -(slot + 1) else 2
        else:
            t = 0 if loc == 1 else 1 if loc == 2 else 2
    return 6 * mi + 2 * t + (1 if mega else 0)


def parse_choice(choice: str | None, request: dict | None, ps_order_side, phase: str) -> int:
    """Whole-side choice -> joint action (slot0*29 + slot1) or preview index."""
    if phase == "preview":
        return preview_action(choice)
    if choice is None or request is None or request.get("wait"):
        return 28 * 29 + 28
    parts = [p for p in choice.split(",")]
    slots = [28, 28]
    if phase == "switch":
        flags = request.get("forceSwitch", [False, False])
        if len(parts) == 1:  # positional after auto-passes of unflagged slots
            idx = 0 if flags[0] else 1
            slots[idx] = slot_action(parts[0], idx, None, ps_order_side)
        else:
            for k in range(2):
                slots[k] = slot_action(parts[k], k, None, ps_order_side)
        return slots[0] * 29 + slots[1]
    act = request.get("active", [None, None])
    for k in range(min(2, len(parts))):
        slots[k] = slot_action(parts[k], k, act[k] if k < len(act) else None, ps_order_side)
    return slots[0] * 29 + slots[1]


def ps_order_np(state, side):
    return np.asarray(state.sides.ps_order[side])


def slot_to_ps(a: int, slot: int, move_target: str | None, locked: bool, ps_order_side) -> str:
    """Engine slot action -> PS choice part (inverse of slot_action on canonical strings)."""
    if a == 28:
        return "pass"
    if a >= 24:
        b = a - 24
        return f"switch {list(ps_order_side).index(b) + 1}"
    m, t, mega = a // 6, (a % 6) // 2, a % 2
    if locked:
        return "move 1"
    loc = None
    if move_target in CHOOSABLE_NAMES:
        if move_target == "adjacentAllyOrSelf":
            loc = -(slot + 1) if t == 0 else -(2 - slot)
        else:
            loc = 1 if t == 0 else 2 if t == 1 else -(2 - slot)
    s = f"move {m + 1}" + (f" {'+' if loc > 0 else ''}{loc}" if loc is not None else "")
    return s + (" mega" if mega else "")


CHOOSABLE_NAMES = {"normal", "any", "adjacentAlly", "adjacentAllyOrSelf", "adjacentFoe"}


def to_ps_choice(joint: int, request: dict, ps_order_side, phase: str) -> str:
    """Joint engine action (or preview index) -> PS choice string (08 §3 interop)."""
    if phase == "preview":
        return "team " + "".join(str(int(x) + 1) for x in PREVIEW_INDEX_INV[int(joint)])
    a = [int(joint) // 29, int(joint) % 29]
    parts = []
    act = request.get("active", [None, None]) if phase == "move" else [None, None]
    for k in range(2):
        mv_t, locked = None, False
        if phase == "move" and a[k] < 24 and act and act[k]:
            moves = act[k]["moves"]
            mv = moves[a[k] // 6] if a[k] // 6 < len(moves) else {}
            locked = len(moves) == 1 and ("target" not in mv or mv.get("id") == "struggle")
            mv_t = mv.get("target")
        parts.append(slot_to_ps(a[k], k, mv_t, locked, ps_order_side))
    return ", ".join(parts)


PREVIEW_INDEX_INV = {i: p for p, i in PREVIEW_INDEX.items()}
