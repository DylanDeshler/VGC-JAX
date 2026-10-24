"""Abilities, first batch (05 §4 step 4 setters; M4 framework): switch-in abilities (onStart run as the SwitchIn
handler, PS sim/battle.ts:1025-1036 getCallback) and Pressure's DeductPP.

PS: data/abilities.ts (Champions overrides in data/mods/champions/abilities.ts: none for this batch). Ability state
fields (abilityState keys shown by the projection) are declared with `fields` and live in mons.ability_state
(slots 0..2, presence bits in slot 3).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle.util import mget, mset
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler

I16, I32 = jnp.int16, jnp.int32
FIELDS: dict[str, list[tuple[str, int, str]]] = {}      # ability id -> [(key, slot, kind)]


def fields(name, *spec):
    """Declare abilityState keys of ability `name`: (key, kind) pairs -> slots 0.. (presence bits in slot 3)."""
    FIELDS[name] = [(k, i, kind) for i, (k, kind) in enumerate(spec)]


def aid(name):
    return getattr(ids, f"ABILITY_{name.upper()}", -1)


def aeff(name):
    """EffectId of an ability."""
    return ids.ABILITY_BASE + aid(name) - 1


def is_ability_effect(eff):
    eff = jnp.asarray(eff, I32)
    return (eff >= ids.ABILITY_BASE) & (eff < ids.ABILITY_BASE + len(ids.ABILITIES) - 1)


def state_field(st, ref, name, key):
    slot = [s for k, s, _ in FIELDS[name] if k == key][0]
    return mget(st.mons.ability_state, ref)[slot]


def set_state_field(st, ref, name, key, value, on):
    slot = [s for k, s, _ in FIELDS[name] if k == key][0]
    m = st.mons
    s = mget(m.ability_state, ref)
    s = s.at[slot].set(jnp.asarray(value, I16)).at[3].set(s[3] | (1 << slot))
    return st.replace(mons=m.replace(ability_state=mset(m.ability_state, ref, s, on)))


def switch_in(name, ps):
    """Register fn as ability `name`'s onStart, run as its SwitchIn handler (fieldEvent('SwitchIn'))."""
    def deco(fn):
        if aid(name) > 0:
            handler(ids.HOOK_SWITCHIN, aeff(name), ps=ps)(fn)
        return fn
    return deco


def _ability_msg(st, on, ref, name, boost=False):
    return EV.emit(st, on, EV.EV_ABILITY, actor=ref, move=aeff(name), flags=EV.F_BOOST if boost else 0)


# ------------------------------------------------------------------ weather / terrain setters
def _weather_setter(name, weather):
    def fn(hc, st, dex, rng):
        """onStart(source): this.field.setWeather(weather) (source = the holder, sourceEffect = the ability)."""
        from vgcjax.effects import weather as WE
        st, rng, _ = WE.set_weather(st, dex, rng, CO.cid(weather), True, source=hc.target, source_effect=aeff(name))
        return hc, st, rng
    return fn


def _terrain_setter(name, terrain):
    def fn(hc, st, dex, rng):
        """onStart(source): this.field.setTerrain(terrain)."""
        from vgcjax.effects import terrain as TR
        st, rng, _ = TR.set_terrain(st, dex, rng, CO.cid(terrain), True, source=hc.target, source_effect=aeff(name))
        return hc, st, rng
    return fn


for _a, _w in (("drought", "sunnyday"), ("drizzle", "raindance"), ("sandstream", "sandstorm"),
               ("snowwarning", "snowscape")):
    # Drought / Drizzle: the Groudon / Kyogre primal-orb exception cannot occur in Reg M-C
    switch_in(_a, f"data/abilities.ts {_a}.onStart")(_weather_setter(_a, _w))
for _a, _t in (("electricsurge", "electricterrain"), ("grassysurge", "grassyterrain"),
               ("psychicsurge", "psychicterrain"), ("mistysurge", "mistyterrain")):
    switch_in(_a, f"data/abilities.ts {_a}.onStart")(_terrain_setter(_a, _t))


# ------------------------------------------------------------------ Cloud Nine (suppressWeather: effects/weather.py)
fields("cloudnine", ("ending", "bool"))


@switch_in("cloudnine", "data/abilities.ts cloudnine.onSwitchIn")
def cloudnine_switch_in(hc, st, dex, rng):
    """'-ability' pokemon 'Cloud Nine'; onStart: abilityState.ending = false, eachEvent('WeatherChange') (no Reg M-C
    handler yet)."""
    st = _ability_msg(st, True, hc.target, "cloudnine")
    return hc, set_state_field(st, hc.target, "cloudnine", "ending", 0, True), rng


if aid("cloudnine") > 0:
    @handler(ids.HOOK_END, aeff("cloudnine"), ps="data/abilities.ts cloudnine.onEnd")
    def cloudnine_end(hc, st, dex, rng):
        """abilityState.ending = true; eachEvent('WeatherChange')."""
        return hc, set_state_field(st, hc.target, "cloudnine", "ending", 1, True), rng


# ------------------------------------------------------------------ Intimidate
@switch_in("intimidate", "data/abilities.ts intimidate.onStart")
def intimidate_start(hc, st, dex, rng):
    """For each unfainted adjacent foe (slot order): '-ability' pokemon 'Intimidate' 'boost' once, then
    boost({atk: -1}, target, pokemon, null, isSecondary) (Substitute: '-immune', M3)."""
    from vgcjax.battle import move_pipeline as MP
    u = hc.target
    fs = 1 - jnp.clip(u, 0, 7) // 4
    act = st.sides.active[fs].astype(I32)
    refs = fs * 4 + jnp.maximum(act, 0)
    alive = (act >= 0) & ~mget(st.mons.fainted, refs) & (mget(st.mons.hp, refs) > 0)
    st = _ability_msg(st, jnp.any(alive), u, "intimidate", boost=True)
    d = jnp.zeros(7, I32).at[0].set(-1)
    for j in range(2):
        st, _ = MP.boost(st, refs[j], d, alive[j], primary=True)
    return hc, st, rng


# ------------------------------------------------------------------ Pressure
@switch_in("pressure", "data/abilities.ts pressure.onStart")
def pressure_start(hc, st, dex, rng):
    """'-ability' pokemon 'Pressure'."""
    return hc, _ability_msg(st, True, hc.target, "pressure"), rng


def pressure_extra_pp(st, dex, u, move, targets, on):
    """runEvent('DeductPP', source, pokemon, move) over pressureTargets (sim/battle-actions.ts:533-548,
    sim/pokemon.ts getMoveTargets): Pressure onDeductPP returns 1 unless the target is an ally. pressureTargets =
    targets (side / field moves: alliesAndSelf + foes(true) for 'all', none for foeSide; the foes for mustpressure
    moves). Returns the extra PP to deduct."""
    p = aid("pressure")
    if p <= 0:
        return jnp.zeros((), I32)
    m = st.mons
    us = jnp.clip(u, 0, 7) // 4
    t = jnp.clip(targets, 0, 7)
    from_targets = (targets >= 0) & (mget(m.ability, t) == p) & ((t // 4) != us)
    n = jnp.sum(from_targets.astype(I32))
    tt = dex.move_target[move]
    fs = 1 - us
    act = st.sides.active[fs].astype(I32)
    frefs = fs * 4 + jnp.maximum(act, 0)
    # foes(true) for 'all' includes fainted foes, whose handlers runEvent skips (inactive: ignoringAbility)
    foes = (act >= 0) & mget(m.is_active, frefs) & ~mget(m.fainted, frefs) & (mget(m.ability, frefs) == p)
    nf = jnp.sum(foes.astype(I32))
    n = jnp.where(tt == ids.TARGET_ALL, nf, jnp.where(
        (tt == ids.TARGET_FOESIDE) | (tt == ids.TARGET_ALLYSIDE) | (tt == ids.TARGET_ALLYTEAM), 0, n))
    w, b = ids.MF_MUSTPRESSURE
    must = ((dex.move_flags[move, w] >> b) & 1) == 1
    n = jnp.where(must, nf, n)
    return jnp.where(on, n, 0)


# ------------------------------------------------------------------ Unnerve (berries: items, M4)
fields("unnerve", ("unnerved", "bool"))


@switch_in("unnerve", "data/abilities.ts unnerve.onStart")
def unnerve_start(hc, st, dex, rng):
    """if (effectState.unnerved) return; '-ability' pokemon 'Unnerve'; effectState.unnerved = true."""
    r = hc.target
    done = (state_field(st, r, "unnerve", "unnerved") != 0) & ((mget(st.mons.ability_state, r)[3] & 1) == 1)
    st = _ability_msg(st, ~done, r, "unnerve")
    st = set_state_field(st, r, "unnerve", "unnerved", 1, ~done)
    return hc, st, rng


if aid("unnerve") > 0:
    @handler(ids.HOOK_END, aeff("unnerve"), ps="data/abilities.ts unnerve.onEnd")
    def unnerve_end(hc, st, dex, rng):
        """effectState.unnerved = false."""
        return hc, set_state_field(st, hc.target, "unnerve", "unnerved", 0, True), rng


# ------------------------------------------------------------------ Frisk
@switch_in("frisk", "data/abilities.ts frisk.onStart")
def frisk_start(hc, st, dex, rng):
    """For each unfainted foe holding an item: '-item' target item '[from] ability: Frisk' '[of] pokemon'."""
    u = hc.target
    fs = 1 - jnp.clip(u, 0, 7) // 4
    for slot in range(2):
        b = st.sides.active[fs, slot].astype(I32)
        r = fs * 4 + jnp.maximum(b, 0)
        it = mget(st.mons.item, r).astype(I32)
        on = (b >= 0) & ~mget(st.mons.fainted, r) & (it > 0)
        st = EV.emit(st, on, EV.EV_ITEM, actor=r, target=u, move=ids.ITEM_BASE + it - 1, v1=aeff("frisk"),
                     flags=EV.F_FROM_ABILITY)
    return hc, st, rng
