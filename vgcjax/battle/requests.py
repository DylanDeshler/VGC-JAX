"""Legal-action masks (04 §4.3, ADR-018) and action sanitizing (ADR-014).

Slot action encoding: a = 6*m + 2*t + mega (a < 24); 24 + b = switch to brought index b; 28 = pass.
Joint action per side = slot0 * 29 + slot1. Masks model PS *acceptance* with full information (07 §6).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import errors as E
from vgcjax.battle import faint as F
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, or_flag
from vgcjax.data import ids

I32 = jnp.int32
NA = 29


def choosable(tt):
    """targetTypeChoices (PS: sim/battle-actions.ts:3): normal, any, adjacentAlly, adjacentAllyOrSelf, adjacentFoe."""
    out = jnp.zeros(jnp.shape(tt), bool)
    for v in (ids.TARGET_NORMAL, ids.TARGET_ANY, ids.TARGET_ADJACENTALLY, ids.TARGET_ADJACENTALLYORSELF,
              ids.TARGET_ADJACENTFOE):
        out = out | (tt == v)
    return out


def target_mask(tt):
    """bool[3] legal t for a target type (04 §4.3 table)."""
    t0 = (tt != ids.TARGET_ADJACENTALLY)
    t1 = (tt == ids.TARGET_NORMAL) | (tt == ids.TARGET_ANY) | (tt == ids.TARGET_ADJACENTFOE)
    t2 = (tt == ids.TARGET_NORMAL) | (tt == ids.TARGET_ANY) | (tt == ids.TARGET_ADJACENTALLY) | (
        tt == ids.TARGET_ADJACENTALLYORSELF)
    return jnp.stack([t0, t1, t2])


def move_slot_mask(st, dex, side, slot):
    """bool[29] for one slot in a MOVE request."""
    b = st.sides.active[side, slot].astype(I32)
    u = side * 4 + jnp.maximum(b, 0)
    m = st.mons
    present = (b >= 0) & ~mget(m.fainted, u)
    moves = mget(m.move_ids, u)
    usable = (mget(m.pp, u) > 0) & (mget(m.move_disabled, u) == 0) & (moves > 0)
    struggle = ~jnp.any(usable)
    can_mega = mget(m.can_mega, u) & ~st.sides.mega_used[side] & ~struggle
    acts = []
    for mi in range(4):
        tm = target_mask(dex.move_target[moves[mi]])
        for t in range(3):
            for mega in range(2):
                ok = usable[mi] & tm[t] & ((mega == 0) | can_mega)
                acts.append(ok)
    acts = jnp.stack(acts)
    acts = jnp.where(struggle, jnp.zeros(24, bool).at[0].set(True), acts)
    poss = F.possible_switches(st, side)
    trapped = mget(m.trapped, u) != 0
    sw = poss & ~trapped
    out = jnp.concatenate([acts, sw, jnp.zeros(1, bool)])
    return jnp.where(present, out, jnp.zeros(NA, bool).at[28].set(True))


def switch_side_masks(st, side):
    """(slot masks [2,29], pair_ok [29,29]) for a SWITCH request (04 §4.3, RF #20)."""
    m = st.mons
    poss = F.possible_switches(st, side)
    nb = jnp.sum(poss.astype(I32))
    flags = []
    for slot in range(2):
        b = st.sides.active[side, slot].astype(I32)
        u = side * 4 + jnp.maximum(b, 0)
        flags.append((b >= 0) & (mget(m.switch_flag, u) != 0))
    nf = flags[0].astype(I32) + flags[1].astype(I32)
    ns = jnp.minimum(nf, nb)
    masks = []
    for slot in range(2):
        sw = jnp.where(flags[slot], poss, False)
        pas = jnp.where(flags[slot], nb < nf, True)
        masks.append(jnp.concatenate([jnp.zeros(24, bool), sw, pas[None]]))
    a = jnp.arange(NA)
    is_sw = (a >= 24) & (a < 28)
    nsw = is_sw[:, None].astype(I32) + is_sw[None, :].astype(I32)
    distinct = ~(is_sw[:, None] & is_sw[None, :] & (a[:, None] == a[None, :]))
    return jnp.stack(masks), distinct & (nsw == ns)


def side_masks(st, dex, side):
    """(slot masks [2,29], pair_ok [29,29]) for the current phase."""
    phase = st.meta.phase
    mm = jnp.stack([move_slot_mask(st, dex, side, 0), move_slot_mask(st, dex, side, 1)])
    a = jnp.arange(NA)
    mega = (a < 24) & (a % 2 == 1)
    is_sw = (a >= 24) & (a < 28)
    pair_move = ~(mega[:, None] & mega[None, :]) & ~(is_sw[:, None] & is_sw[None, :] & (a[:, None] == a[None, :]))
    sm, pair_sw = switch_side_masks(st, side)
    is_switch = phase == S.SWITCH
    masks = jnp.where(is_switch, sm, mm)
    pair = jnp.where(is_switch, pair_sw, pair_move)
    passonly = jnp.zeros((2, NA), bool).at[:, 28].set(True)
    masks = jnp.where((phase == S.MOVE) | is_switch, masks, passonly)
    return masks, pair


def legal_masks(st, dex):
    """Per side: {'slot': bool[2,29], 'pair_ok': bool[29,29], 'preview': bool[360]} stacked over sides."""
    out = [side_masks(st, dex, s) for s in range(2)]
    slot = jnp.stack([o[0] for o in out])
    pair = jnp.stack([o[1] for o in out])
    preview = jnp.broadcast_to(st.meta.phase == S.PREVIEW, (2, 360))
    return {"slot": slot, "pair_ok": pair, "preview": preview}


def joint_legal(st, dex):
    """bool[2, 841] legal joint actions per side."""
    mk = legal_masks(st, dex)
    j = mk["slot"][:, 0, :, None] & mk["slot"][:, 1, None, :] & mk["pair_ok"]
    return j.reshape(2, NA * NA)


def sanitize(st, dex, actions):
    """Replace illegal actions by the first legal one and flag ERR_ILLEGAL_ACTION. Returns (st, actions)."""
    phase = st.meta.phase
    jl = joint_legal(st, dex)
    out = []
    bad_any = jnp.zeros((), bool)
    for s in range(2):
        a = actions[s]
        in_preview = phase == S.PREVIEW
        ok_prev = (a >= 0) & (a < 360)
        ok_joint = (a >= 0) & (a < NA * NA) & jl[s, jnp.clip(a, 0, NA * NA - 1)]
        ok = jnp.where(in_preview, ok_prev, jnp.where(phase == S.DONE, True, ok_joint))
        first = jnp.argmax(jl[s])
        out.append(jnp.where(ok, a, jnp.where(in_preview, 0, first)))
        bad_any = bad_any | ~ok
    st = or_flag(st, E.ERR_ILLEGAL_ACTION, bad_any)
    return st, jnp.stack(out)
