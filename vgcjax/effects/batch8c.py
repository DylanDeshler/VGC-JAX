"""Batch 8c: Zero to Hero (Palafin), Ice Body, Surge Surfer (Raichu-Alola), Marvel Scale.

PS: data/abilities.ts (no Champions override: data/mods/champions/abilities.ts does not mention any of them),
[C] data/mods/champions/scripts.ts:55-115 formeChange. MUL-hook callbacks are effects/modifiers.py mods; the others are
undispatched helpers called from their PS event sites (perf rule):
- Zero to Hero onSwitchOut: battle/switching.switch_in right after runEvent('SwitchOut') (`zero_to_hero_out`);
  onSwitchIn: abilities.switch_in_vec (`zero_to_hero_in`; the ability is listed in abilities.SWITCH_IN);
- Ice Body onWeather: weather.each_weather, per mon next to Rain Dish / Dry Skin (`ice_body_weather`); its
  onImmunity('hail') cannot matter (no hail in Reg M-C: Snowscape deals no residual damage);
- Surge Surfer onModifySpe: battle/common.spe_chain (`surge_surfer_spe`);
- Marvel Scale onModifyDef: a mod (breakable: Mold Breaker masks the ability, effects/batch_reactive.py).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle.util import mget, mset
from vgcjax.core import fixedpoint as fp
from vgcjax.data import ids
from vgcjax.effects.abilities import aeff, aid
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32


def _sid(name):
    return getattr(ids, f"SPECIES_{name.upper()}", -1)


def _is(ab, name):
    return ab == aid(name) if aid(name) > 0 else jnp.zeros(jnp.shape(ab), bool)


# ------------------------------------------------------------------ Zero to Hero
def _palafin_line(st, dex, r):
    """pokemon.baseSpecies.baseSpecies === 'Palafin' with Zero to Hero not ignored: a notransform ability of a
    Transformed mon is ignored (sim/pokemon.ts:883 ignoringAbility; cantsuppress covers Gastro Acid / Neutralizing
    Gas)."""
    m = st.mons
    base = dex.species_base_species[mget(m.base_species, r).astype(I32)].astype(I32)
    return _is(mget(m.ability, r), "zerotohero") & (base == _sid("palafin")) & ~mget(m.transformed, r)


def zero_to_hero_out(st, dex, r, on):
    """Zero to Hero onSwitchOut (data/abilities.ts:5629-5635): Palafin not in its Hero forme ->
    formeChange('Palafin-Hero', Zero to Hero, true) ([C] data/mods/champions/scripts.ts:55-115: setSpecies (types,
    weight, storedStats, speed); baseSpecies = Palafin-Hero; 'detailschange' X details; updateMaxHp (no change:
    Palafin and Palafin-Hero share base HP 100); no '-formechange' for an Ability source; setAbility('Zero to Hero',
    null, null, true) (sim/pokemon.ts:1929-1967: End (no handler), a new abilityState (effectOrder++), Start (no
    handler)); baseAbility unchanged), then heroMessageDisplayed = false. Runs while the mon is still active, before
    the ability End / clearVolatile of switchIn (sim/battle-actions.ts:100-121)."""
    if aid("zerotohero") <= 0 or _sid("palafinhero") <= 0:
        return st
    from vgcjax.battle import switching as SW
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    hero = _sid("palafinhero")
    go = jnp.asarray(on) & _palafin_line(st, dex, r) & (mget(st.mons.species, r) != hero)
    st = SW.set_species(st, dex, r, hero, go)
    m = st.mons
    meta = st.meta
    ab = mget(m.ability, r)
    st = st.replace(mons=m.replace(base_species=mset(m.base_species, r, hero, go),
                                   ability_state=mset(m.ability_state, r, jnp.zeros(4, I16), go),
                                   ability_state_of=mset(m.ability_state_of, r, ab, go),
                                   ability_seq=mset(m.ability_seq, r, meta.effect_seq, go),
                                   hero_msg=mset(m.hero_msg, r, False, go)),
                    meta=meta.replace(effect_seq=meta.effect_seq + go.astype(meta.effect_seq.dtype)))
    return EV.emit(st, go, EV.EV_DETAILSCHANGE, actor=r, v1=hero)


def zero_to_hero_in(st, dex, r, on):
    """Zero to Hero onSwitchIn (data/abilities.ts:5636-5642): Palafin in its Hero forme with heroMessageDisplayed
    false (sim/pokemon.ts:500: false at construction) -> '-activate' X 'ability: Zero to Hero', heroMessageDisplayed
    = true. `on`: the SwitchIn handler of r's ability runs (not singleEvent('Start'): no onStart)."""
    if aid("zerotohero") <= 0 or _sid("palafinhero") <= 0:
        return st
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    m = st.mons
    go = jnp.asarray(on) & _palafin_line(st, dex, r) & (mget(m.species, r) == _sid("palafinhero")) & \
        ~mget(m.hero_msg, r)
    st = EV.emit(st, go, EV.EV_ACTIVATE, actor=r, move=aeff("zerotohero"))
    m = st.mons
    return st.replace(mons=m.replace(hero_msg=mset(m.hero_msg, r, True, go)))


# ------------------------------------------------------------------ Ice Body
def ice_body_weather(st, dex, rng, r, ok, w):
    """Ice Body onWeather (data/abilities.ts:1957-1961): effect (the field weather condition, eachEvent's
    this.effect: not effectiveWeather) is Hail / Snowscape -> heal(baseMaxhp / 16) ('-heal' X hp '[from] ability: Ice
    Body'). `ok`: eachEvent('Weather') reached r with weather unsuppressed (sim/battle.ts:470-481); w = the weather.
    Returns (st, rng)."""
    if aid("icebody") <= 0 or "snowscape" not in ids.CONDITIONS:
        return st, rng
    from vgcjax.effects import statuses as STS
    from vgcjax.effects import weather as WE
    go = jnp.asarray(ok) & (w == WE.wid("snowscape")) & _is(mget(st.mons.ability, r), "icebody")
    return STS.effect_heal(st, dex, rng, r, mget(st.mons.maxhp, r) // 16, aeff("icebody"), go)


# ------------------------------------------------------------------ Surge Surfer
def surge_surfer_spe(st, ab, active, m4096):
    """Surge Surfer onModifySpe (data/abilities.ts:4758-4762): field.isTerrain('electricterrain') (sim/field.ts:
    169-180; no Reg M-C TryTerrain handler) -> chainModify(2). Exact, so order-free in spe_chain."""
    if aid("surgesurfer") <= 0 or "electricterrain" not in ids.CONDITIONS:
        return m4096
    et = st.field.terrain.astype(I32) == ids.CONDITIONS.index("electricterrain")
    return jnp.where(active & _is(ab, "surgesurfer") & et, fp.chain_modify(m4096, 2, 1), m4096)


# ------------------------------------------------------------------ Marvel Scale
if aid("marvelscale") > 0:
    @mod("ModifyDef", aeff("marvelscale"), ps="data/abilities.ts:2536-2541 marvelscale.onModifyDef")
    def marvelscale_def(ctx, h):
        """onModifyDef (priority 6), the defender's: pokemon.status -> chainModify(1.5)."""
        return mget(ctx.st.mons.status, jnp.clip(h, 0, 7)) > 0, jnp.asarray(6144, I32), None
