"""Champions stat formula, stat stages and the (hook-free) effective-stat pipeline (04 §4.1-4.2).

Stat order: 6-vectors are (hp, atk, def, spa, spd, spe); 5-vectors drop hp. Boost order (atk, def, spa, spd, spe,
accuracy, evasion). Nature plus/minus are 6-space stat indices (-1 = neutral).
"""
from __future__ import annotations

import jax.numpy as jnp

I32 = jnp.int32


def calc_stats(base6, sp6, nature_plus, nature_minus):
    """(maxhp, stored[..., 5]) from base stats, Stat Points and nature.

    PS: data/mods/champions/scripts.ts:10-40 statModify (no Level Clause Mod): HP = base + SP + 75;
    X = base + SP + 20; nature +/-: trunc(trunc(X * 110|90, 16) / 100).
    """
    base6 = jnp.asarray(base6, I32)
    sp6 = jnp.asarray(sp6, I32)
    raw = base6 + sp6 + 20
    idx = jnp.arange(6)
    plus = idx == jnp.asarray(nature_plus, I32)[..., None]
    minus = idx == jnp.asarray(nature_minus, I32)[..., None]
    up = ((raw * 110) & 0xFFFF) // 100
    down = ((raw * 90) & 0xFFFF) // 100
    stats = jnp.where(plus, up, jnp.where(minus, down, raw))
    maxhp = base6[..., 0] + sp6[..., 0] + 75
    return maxhp, stats[..., 1:]


def boosted(stat, boost):
    """Stage multiplier. PS: sim/pokemon.ts:635-650 floor(stat * [1,1.5,..,4][b]) / floor(stat / table[-b])."""
    stat = jnp.asarray(stat, I32)
    b = jnp.clip(jnp.asarray(boost, I32), -6, 6)
    return jnp.where(b >= 0, (stat * (2 + b)) // 2, (stat * 2) // (2 - b))


def paralysis_speed(spe, paralyzed, quick_feet=False):
    """PS: data/conditions.ts:30-38 par.onModifySpe: finalModify(spe) then floor(spe * 50 / 100) unless Quick Feet.

    `spe` must already have the ModifySpe chain applied (finalModify).
    """
    spe = jnp.asarray(spe, I32)
    return jnp.where(jnp.asarray(paralyzed) & ~jnp.asarray(quick_feet), (spe * 50) // 100, spe)


def cap_speed(spe):
    """PS: sim/pokemon.ts:658 speed cap 10000."""
    return jnp.minimum(jnp.asarray(spe, I32), 10000)


def action_speed(spe_stat, trick_room):
    """PS: data/mods/champions/scripts.ts:46-54 getActionSpeed [C]: negated under Trick Room (no underflow wrap)."""
    spe_stat = jnp.asarray(spe_stat, I32)
    return jnp.where(jnp.asarray(trick_room), -spe_stat, spe_stat)


def get_stat_hookless(stored5, boosts7, stat_idx, paralyzed=False):
    """getStat with an identity Modify* chain (core stage): stages, paralysis (spe), 10000 cap (spe).

    PS: sim/pokemon.ts:617-660. stat_idx is the 5-space index (0=atk .. 4=spe).
    """
    s = boosted(jnp.take(stored5, stat_idx, axis=-1), jnp.take(boosts7, stat_idx, axis=-1))
    is_spe = jnp.asarray(stat_idx) == 4
    s = jnp.where(is_spe, cap_speed(paralysis_speed(s, paralyzed)), s)
    return s
