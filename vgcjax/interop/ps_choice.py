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
