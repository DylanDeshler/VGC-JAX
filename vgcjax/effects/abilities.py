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


SWITCH_IN = ("drought", "drizzle", "sandstream", "snowwarning", "electricsurge", "grassysurge", "psychicsurge",
             "mistysurge", "cloudnine", "intimidate", "pressure", "unnerve", "frisk")
WEATHER_SETTERS = {"drought": "sunnyday", "drizzle": "raindance", "sandstream": "sandstorm", "snowwarning": "snowscape"}
TERRAIN_SETTERS = {"electricsurge": "electricterrain", "grassysurge": "grassyterrain", "psychicsurge": "psychicterrain",
                   "mistysurge": "mistyterrain"}
fields("cloudnine", ("ending", "bool"))
fields("unnerve", ("unnerved", "bool"))


def _noop(hc, st, dex, rng):
    return hc, st, rng


# the SwitchIn / End handlers are registered for the static candidate lists (effects/static.py) but run undispatched
# through switch_in_vec / end_vec: one masked pass instead of a lax.switch whose every branch carries the whole state
for _a in SWITCH_IN:
    if aid(_a) > 0:
        handler(ids.HOOK_SWITCHIN, aeff(_a), ps=f"data/abilities.ts {_a}.onStart (switch_in_vec)")(_noop)


def _ability_msg(st, on, ref, name, boost=False):
    return EV.emit(st, on, EV.EV_ABILITY, actor=ref, move=aeff(name), flags=EV.F_BOOST if boost else 0)


def _table(mapping, value_fn):
    t = [0] * (len(ids.ABILITIES) + 1)
    for a, v in mapping.items():
        if aid(a) > 0 and v in ids.CONDITIONS:
            t[aid(a)] = value_fn(v)
    return jnp.array(t, I32)


def switch_in_vec(st, dex, rng, u, on):
    """The SwitchIn handler (onStart, or onSwitchIn when defined) of holder u's ability, for the abilities of
    SWITCH_IN (data/abilities.ts):
      Drought / Drizzle / Sand Stream / Snow Warning: this.field.setWeather(w) (source = holder, sourceEffect = the
        ability; the Groudon / Kyogre primal-orb exception cannot occur in Reg M-C);
      Electric / Grassy / Psychic / Misty Surge: this.field.setTerrain(t);
      Cloud Nine onSwitchIn: '-ability' X 'Cloud Nine', onStart: abilityState.ending = false, eachEvent('WeatherChange')
        (no Reg M-C handler);
      Intimidate: '-ability' X 'Intimidate' 'boost' once, then per unfainted adjacent foe (slot order)
        boost({atk: -1}, target, pokemon, null, isSecondary) (Substitute: '-immune', M3);
      Pressure: '-ability' X 'Pressure';
      Unnerve: unless effectState.unnerved, '-ability' X 'Unnerve' and unnerved = true;
      Frisk: per unfainted foe holding an item, '-item' target item '[from] ability: Frisk' '[of] pokemon'.
    Returns (st, rng)."""
    from vgcjax.battle import move_pipeline as MP
    from vgcjax.effects import terrain as TR
    from vgcjax.effects import weather as WE
    on = jnp.asarray(on)
    u = jnp.asarray(u, I32)
    r = jnp.clip(u, 0, 7)
    ab = mget(st.mons.ability, r).astype(I32)
    is_ = lambda name: on & (ab == aid(name)) if aid(name) > 0 else jnp.zeros((), bool)  # noqa: E731
    src_eff = ids.ABILITY_BASE + ab - 1
    wc = _table(WEATHER_SETTERS, CO.cid)[ab]
    st, rng, _ = WE.set_weather(st, dex, rng, wc, on & (wc > 0), source=u, source_effect=src_eff)
    tc = _table(TERRAIN_SETTERS, CO.cid)[ab]
    st, rng, _ = TR.set_terrain(st, dex, rng, tc, on & (tc > 0), source=u, source_effect=src_eff)
    c9 = is_("cloudnine")
    st = _ability_msg(st, c9, u, "cloudnine")
    if aid("cloudnine") > 0:
        st = set_state_field(st, r, "cloudnine", "ending", 0, c9)
    st = _ability_msg(st, is_("pressure"), u, "pressure")
    if aid("unnerve") > 0:
        un = is_("unnerve")
        done = (state_field(st, r, "unnerve", "unnerved") != 0) & ((mget(st.mons.ability_state, r)[3] & 1) == 1)
        st = _ability_msg(st, un & ~done, u, "unnerve")
        st = set_state_field(st, r, "unnerve", "unnerved", 1, un & ~done)
    fs = 1 - r // 4
    act = st.sides.active[fs].astype(I32)
    frefs = fs * 4 + jnp.maximum(act, 0)
    foe_up = (act >= 0) & ~mget(st.mons.fainted, frefs)
    intim = is_("intimidate")
    alive = foe_up & (mget(st.mons.hp, frefs) > 0) & intim
    st = _ability_msg(st, jnp.any(alive), u, "intimidate", boost=True)
    d = jnp.zeros(7, I32).at[0].set(-1)
    for j in range(2):
        st, _ = MP.boost(st, frefs[j], d, alive[j], primary=True, source=u, effect=aeff("intimidate"))
    frisk = is_("frisk")
    for j in range(2):
        it = mget(st.mons.item, frefs[j]).astype(I32)
        st = EV.emit(st, frisk & foe_up[j] & (it > 0), EV.EV_ITEM, actor=frefs[j], target=u,
                     move=ids.ITEM_BASE + it - 1, v1=aeff("frisk"), flags=EV.F_FROM_ABILITY)
    return st, rng


def end_vec(st, refs, on):
    """singleEvent('End', ability) for refs [N] at once: Cloud Nine onEnd (abilityState.ending = true;
    eachEvent('WeatherChange'): no Reg M-C handler), Unnerve onEnd (unnerved = false). State only."""
    refs = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    on = jnp.asarray(on)
    m = st.mons
    ab = mget(m.ability, refs)
    s = mget(m.ability_state, refs)                                   # [N, 4]
    for name, key, val in (("cloudnine", "ending", 1), ("unnerve", "unnerved", 0)):
        if aid(name) <= 0:
            continue
        slot = [sl for k, sl, _ in FIELDS[name] if k == key][0]
        hit = (on & (ab == aid(name)))[:, None]
        s = jnp.where(hit, s.at[:, slot].set(val).at[:, 3].set(s[:, 3] | (1 << slot)), s)
    from vgcjax.battle.util import mset_many
    return st.replace(mons=m.replace(ability_state=mset_many(m.ability_state, refs, s, on)))


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


