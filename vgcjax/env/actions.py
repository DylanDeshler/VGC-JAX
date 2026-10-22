"""Action encodings for the env (ADR-018, 08 §3): preview orders, canonical preview subset, joint masks."""
from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np

from vgcjax.data import ids

NA = 29
NJ = NA * NA
PREVIEW_ORDERS = np.array(list(itertools.permutations(range(6), 4)), np.int8)   # == dex.preview_orders
# canonical: leads ascending, bench ascending (90 = C(6,2) lead pairs x C(4,2) bench pairs)
CANONICAL = np.array([(o[0] < o[1]) and (o[2] < o[3]) for o in PREVIEW_ORDERS], bool)
# bench-order-free variant: leads ascending only (bench order kept, for Illusion: up to 180)
LEADS_ASC = np.array([o[0] < o[1] for o in PREVIEW_ORDERS], bool)


def preview_mask(team_abilities_side, canonical: bool = True):
    """bool[360] preview options for one side. With `canonical`, 90 options, or 180 if a sheet mon has Illusion
    (bench order matters: Illusion disguises as the last party member, 08 §3)."""
    if not canonical:
        return jnp.ones(360, bool)
    illusion = getattr(ids, "ABILITY_ILLUSION", -1)
    has_ill = jnp.any(jnp.asarray(team_abilities_side) == illusion)
    return jnp.where(has_ill, jnp.asarray(LEADS_ASC), jnp.asarray(CANONICAL))


def joint_mask(masks, side):
    """bool[841] legal joint actions (slot0*29 + slot1) from Masks {slot: [2,2,29], pair_ok: [2,29,29]}."""
    s = masks["slot"][side]
    return (s[0][:, None] & s[1][None, :] & masks["pair_ok"][side]).reshape(NJ)


def split_joint(j):
    j = jnp.asarray(j)
    return j // NA, j % NA
