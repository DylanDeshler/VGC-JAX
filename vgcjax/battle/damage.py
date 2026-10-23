"""Damage formula (04 §7.4), integer-exact (ADR-011).

M1.5 scope: the hook-free core. Hook points that later milestones fill in are marked `HOOK(Hn)`; until then each
is the identity (BasePower, ModifyAtk/Def, WeatherModifyDamage, ModifySTAB, Effectiveness, ModifyDamage), except
the CriticalHit veto of Battle Armor / Shell Armor which the core stage needs.

Result kinds mirror PS getDamage's return values: DMG_NUMBER (a number), DMG_FALSE (immune: `false`),
DMG_UNDEFINED (basePower 0: `undefined`), DMG_ZERO (BasePower hook returned 0: `0`).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.core import fixedpoint as fp
from vgcjax.core import stats as S
from vgcjax.core import typechart as tc
from vgcjax.data import ids

I32 = jnp.int32
DMG_NUMBER, DMG_FALSE, DMG_UNDEFINED, DMG_ZERO = 0, 1, 2, 3
CRIT_DEN = jnp.array([0, 24, 8, 2, 1], I32)  # PS: sim/battle-actions.ts:1687 gen >= 7 critMult
LEVEL = 50


def crit_denominator(crit_ratio):
    """PS: sim/battle-actions.ts:1678-1689: critRatio clamped to [0, 4]; randomChance(1, critMult[critRatio])."""
    return CRIT_DEN[jnp.clip(jnp.asarray(crit_ratio, I32), 0, 4)]


def base_damage(level, bp, attack, defense):
    """PS: sim/battle-actions.ts:1772 tr(tr(tr(tr(2 * L / 5 + 2) * P * A) / D) / 50)."""
    a = (2 * jnp.asarray(level, I32)) // 5 + 2
    return ((a * jnp.asarray(bp, I32) * jnp.asarray(attack, I32)) // jnp.asarray(defense, I32)) // 50


def modify_damage(base, *, spread, pbond_hit2, crit, roll, stab_m4096, type_mod, burn, final_m4096,
                  weather_m4096=4096, bypass_protect=False, crit_num=3, crit_den=2):
    """Champions modifyDamage. PS: data/mods/champions/scripts.ts:196-312.

    `roll` is the DAMAGE_ROLL draw (0..15); `stab_m4096` the STAB modifier (4096 = none, 6144 = 1.5x) after
    HOOK(H18 ModifySTAB); `type_mod` the clamped effectiveness; `final_m4096` the ModifyDamage chain (HOOK H20).
    """
    d = jnp.asarray(base, I32) + 2
    d = jnp.where(spread, fp.modify_raw(d, 3072), jnp.where(pbond_hit2, fp.modify_raw(d, 1024), d))
    d = fp.modify_raw(d, weather_m4096)                      # HOOK(H17) WeatherModifyDamage (priorityEvent)
    d = jnp.where(crit, (d * crit_num) // crit_den, d)       # tr(baseDamage * 1.5)
    d = (d * (100 - jnp.asarray(roll, I32))) // 100          # PS: sim/battle.ts:2407-2410 randomizer
    d = fp.modify_raw(d, stab_m4096)
    tm = jnp.clip(jnp.asarray(type_mod, I32), -6, 6)
    d = jnp.where(tm > 0, d << jnp.maximum(tm, 0), d >> jnp.maximum(-tm, 0))  # x2 per step / tr(d/2) per step
    d = jnp.where(burn, fp.modify_raw(d, 2048), d)
    d = fp.modify_raw(d, final_m4096)                        # HOOK(H20) ModifyDamage chain
    d = jnp.where(bypass_protect, fp.modify_raw(d, 1024), d)  # [C] Unseen Fist / Piercing Drill
    return jnp.where(d == 0, 1, fp.trunc16(d))


def get_damage_core(dex, move, *, atk_stats5, atk_boosts7, atk_types, atk_burned, def_stats5, def_boosts7, def_types,
                    def_maxhp, def_grounded, def_crit_immune, crit_draw, roll, spread, pbond_hit2=False,
                    canary_crit=False, final_m4096=4096):
    """Hook-free getDamage for move `move` (MoveId). PS: sim/battle-actions.ts:1639-1776.

    `crit_draw` is the CRIT draw outcome (ignored when crit_ratio == 0 or the move has willCrit);
    `def_crit_immune` the CriticalHit veto (Battle Armor / Shell Armor, H13); `def_grounded` from isGrounded
    (Levitate / Flying). Returns (damage i32, kind i32, crit bool).
    """
    I = lambda x: jnp.asarray(x, I32)  # noqa: E731
    mtype = dex.move_type[move]
    cat = dex.move_category[move]
    is_phys = cat == ids.CAT_PHYSICAL
    # runImmunity (PS: sim/pokemon.ts:2257-2285): typed immunity, Ground via isGrounded; ignoreImmunity === true
    type_immune = jnp.where(mtype == ids.TYPE_GROUND, ~jnp.asarray(def_grounded), tc.immune(dex, mtype, def_types))
    type_immune = type_immune & (dex.move_ignore_immunity[move] != 1) & (mtype != ids.TYPE_TYPELESS)
    ohko = dex.move_ohko[move] > 0
    fixed = dex.move_fixed_damage[move]
    bp = I(dex.move_bp[move])
    # crit (PS :1678-1700): ModifyCritRatio HOOK(H13) = identity; willCrit; CriticalHit veto
    crit_ratio = jnp.clip(I(dex.move_crit_ratio[move]), 0, 4)
    crit = jnp.where(dex.move_will_crit[move] == 1, True, (crit_ratio > 0) & jnp.asarray(crit_draw))
    crit = crit & ~jnp.asarray(def_crit_immune)
    # BasePower HOOK(H14) = identity
    # stats (PS :1725-1760): crit ignores negative offensive / positive defensive boosts
    off_idx = jnp.where(dex.move_override_off_stat[move] > 0, dex.move_override_off_stat[move] - 1,
                        jnp.where(is_phys, 0, 2))
    def_idx = jnp.where(dex.move_override_def_stat[move] > 0, dex.move_override_def_stat[move] - 1,
                        jnp.where(is_phys, 1, 3))
    use_target_off = dex.move_override_off_pokemon[move] == 1  # Foul Play
    off_stats = jnp.where(use_target_off, I(def_stats5), I(atk_stats5))
    off_boosts = jnp.where(use_target_off, I(def_boosts7), I(atk_boosts7))
    ab = jnp.take(off_boosts, off_idx)
    db = jnp.take(I(def_boosts7), def_idx)
    ign_neg_off = crit | (dex.move_ignore_negative_offensive[move] == 1)
    ign_pos_def = crit | (dex.move_ignore_positive_defensive[move] == 1)
    ab = jnp.where((dex.move_ignore_offensive[move] == 1) | (ign_neg_off & (ab < 0)), 0, ab)
    db = jnp.where((dex.move_ignore_defensive[move] == 1) | (ign_pos_def & (db > 0)), 0, db)
    attack = S.boosted(jnp.take(off_stats, off_idx), ab)      # HOOK(H15) ModifyAtk/SpA = identity
    defense = S.boosted(jnp.take(I(def_stats5), def_idx), db)  # HOOK(H16) ModifyDef/SpD = identity
    base = base_damage(LEVEL, jnp.maximum(bp, 1), attack, defense)
    stab = (mtype != ids.TYPE_TYPELESS) & (jnp.any(I(atk_types) == mtype) | (dex.move_force_stab[move] == 1))
    stab_m = jnp.where(stab, 6144, 4096)
    type_mod = tc.effectiveness(dex, mtype, def_types)       # HOOK(H19) Effectiveness = identity
    burn = jnp.asarray(atk_burned) & is_phys                 # Guts / Facade handled with hooks later
    crit_num, crit_den = (5, 4) if canary_crit else (3, 2)
    dmg = modify_damage(base, spread=spread, pbond_hit2=pbond_hit2, crit=crit, roll=roll, stab_m4096=stab_m,
                        type_mod=type_mod, burn=burn, final_m4096=final_m4096, crit_num=crit_num, crit_den=crit_den)
    kind = jnp.where(type_immune, DMG_FALSE,
                     jnp.where(ohko | (fixed != 0), DMG_NUMBER, jnp.where(bp == 0, DMG_UNDEFINED, DMG_NUMBER)))
    dmg = jnp.where(ohko, I(def_maxhp), jnp.where(fixed == -1, LEVEL, jnp.where(fixed > 0, fixed, dmg)))
    dmg = jnp.where(kind == DMG_NUMBER, dmg, 0)
    crit = crit & (kind == DMG_NUMBER) & ~ohko & (fixed == 0)
    return dmg, kind, crit
