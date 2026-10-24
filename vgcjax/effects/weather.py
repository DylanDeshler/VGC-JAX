"""Weather (05 §4 step 4): Sun, Rain, Sandstorm, Snow, set by moves (abilities: M4).

PS: data/conditions.ts sunnyday / raindance / sandstorm / snowscape (no Champions overrides), sim/field.ts:39-125
setWeather / clearWeather / effectiveWeather / suppressingWeather, sim/pokemon.ts:2209-2227 effectiveWeather,
sim/battle.ts:470-480 eachEvent. The weathers' damage / stat / immunity / move callbacks that sit inside formulas
are explicit here and called from the formula sites: WeatherModifyDamage (weather_damage_mod), sunnyday onImmunity
(frz_immune), and the weather-dependent onModifyType / onModifyMove of Weather Ball, Thunder, Hurricane, Blizzard
(modify_move); sandstorm onModifySpD / snowscape onModifyDef are effects/modifiers.py mods.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget
from vgcjax.core import fixedpoint as fp
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects import modifiers as MOD
from vgcjax.effects.hooks import HookCtx, handler

I16, I32 = jnp.int16, jnp.int32
WEATHERS = ("sunnyday", "raindance", "sandstorm", "snowscape")
ROCKS = {"sunnyday": "HEATROCK", "raindance": "DAMPROCK", "sandstorm": "SMOOTHROCK", "snowscape": "ICYROCK"}
SUPPRESSORS = tuple(getattr(ids, f"ABILITY_{a}") for a in ("CLOUDNINE", "AIRLOCK") if hasattr(ids, f"ABILITY_{a}"))


def wid(name):
    return CO.cid(name) if name in ids.CONDITIONS else -1


def suppressing(st):
    """PS: sim/field.ts:106-116 suppressingWeather: an active, unfainted Cloud Nine / Air Lock holder whose
    abilityState.ending is not set (ignoringAbility: M4)."""
    act = st.sides.active.reshape(4).astype(I32)
    refs = jnp.clip((jnp.arange(4) // 2) * 4 + act, 0, 7)
    ab = mget(st.mons.ability, refs)
    sup = jnp.zeros(4, bool)
    for a in SUPPRESSORS:
        sup = sup | (ab == a)
    s = mget(st.mons.ability_state, refs)
    ending = ((s[:, 3] & 1) == 1) & (s[:, 0] != 0)              # abilityState.ending (effects/abilities.py slot 0)
    return jnp.any((act >= 0) & ~mget(st.mons.fainted, refs) & sup & ~ending)


def field_weather(st):
    """PS: sim/field.ts:101-104 effectiveWeather (ConditionId, 0 = none)."""
    w = st.field.weather.astype(I32)
    return jnp.where(suppressing(st), 0, w)


def mon_weather(st, ref):
    """PS: sim/pokemon.ts:2209-2227 pokemon.effectiveWeather: Utility Umbrella (not in Reg M-C) and Mega Sol (M3.6)
    aside, the field's effective weather."""
    return field_weather(st)


def is_weather(st, name):
    return field_weather(st) == wid(name)


# ------------------------------------------------------------------ formula-site callbacks
def weather_damage_mod(st, dex, u, move, mtype, targets):
    """priorityEvent('WeatherModifyDamage', pokemon, target, move) ([C] scripts.ts:216): sunnyday / raindance
    onWeatherModifyDamage (data/conditions.ts) on the defender's effectiveWeather; Hydro Steam keeps x1.5 in sun on
    the attacker's. Returns the 4096-scale modifier per target slot."""
    wt = jax.vmap(lambda t: mon_weather(st, t))(jnp.clip(targets, 0, 7))
    sun, rain = wt == wid("sunnyday"), wt == wid("raindance")
    fire, water = mtype == ids.TYPE_FIRE, mtype == ids.TYPE_WATER
    up = (sun & fire) | (rain & water)
    down = (sun & water) | (rain & fire)
    hs = getattr(ids, "MOVE_HYDROSTEAM", -1)
    steam = (move == hs) & (mon_weather(st, u) == wid("sunnyday"))
    return jnp.where(steam | up, 6144, jnp.where(down, 2048, 4096)).astype(I32)


def frz_immune(st, refs):
    """sunnyday onImmunity(type, pokemon): 'frz' under the pokemon's effective Sun -> false (data/conditions.ts)."""
    return jax.vmap(lambda r: mon_weather(st, r))(jnp.clip(jnp.atleast_1d(refs), 0, 7)).reshape(jnp.shape(refs)) \
        == wid("sunnyday")


def modify_move(st, dex, u, target, move):
    """The ActiveMove's weather-dependent fields after singleEvent('ModifyType') / singleEvent('ModifyMove')
    (sim/battle-actions.ts:494-495): Weather Ball onModifyType / onModifyMove (type by the user's weather, basePower
    x2 in any weather), Thunder / Hurricane onModifyMove (target's Rain: accuracy true, Sun: 50), Blizzard onModifyMove
    (Snow: accuracy true) (data/moves.ts). Returns (type, basePower, accuracy (0 = true))."""
    mtype = dex.move_type[move].astype(I32)
    bp = dex.move_bp[move].astype(I32)
    acc = dex.move_accuracy[move].astype(I32)
    wu = mon_weather(st, u)
    wb = move == getattr(ids, "MOVE_WEATHERBALL", -1)
    for name, ty in (("sunnyday", ids.TYPE_FIRE), ("raindance", ids.TYPE_WATER), ("sandstorm", ids.TYPE_ROCK),
                     ("snowscape", ids.TYPE_ICE)):
        hit = wb & (wu == wid(name))
        mtype = jnp.where(hit, ty, mtype)
        bp = jnp.where(hit, bp * 2, bp)
    wt = jnp.where(target >= 0, mon_weather(st, jnp.clip(target, 0, 7)), 0)
    th = (move == getattr(ids, "MOVE_THUNDER", -1)) | (move == getattr(ids, "MOVE_HURRICANE", -1))
    acc = jnp.where(th & (wt == wid("raindance")), 0, jnp.where(th & (wt == wid("sunnyday")), 50, acc))
    acc = jnp.where((move == getattr(ids, "MOVE_BLIZZARD", -1)) & is_weather(st, "snowscape"), 0, acc)
    return mtype, bp, acc


# ------------------------------------------------------------------ setWeather
def set_weather(st, dex, rng, c, on, source=-1, source_effect=-1):
    """PS: sim/field.ts:39-89 setWeather(status, source, sourceEffect) for a move source: false when already active;
    SetWeather event (no Reg M-C handler); new state (duration 5, durationCallback; no target so effectOrder 0);
    FieldStart (falsy restores the previous weather); eachEvent('WeatherChange') (ability handlers: M4).
    Returns (st, rng, result)."""
    on = jnp.asarray(on)
    c = jnp.asarray(c, I32)
    e = ids.COND_BASE + c - 1
    f = st.field
    go = on & (f.weather.astype(I32) != c)
    prev_w, prev_s = f.weather, f.weather_state
    dur = dex.cond_duration[c].astype(I32)
    hc = HookCtx.make(-1, source, source_effect).replace(holder=10)
    hc, st, rng = CO.single_event(CO.DURATION_CALLBACK, e, st, dex, rng, hc, go)
    dur = jnp.where(hc.ret_t == RS.NUM, hc.ret_v, dur)
    f = st.field
    st = st.replace(field=f.replace(weather=jnp.where(go, c, f.weather).astype(f.weather.dtype),
                                    weather_state=jnp.where(go, jnp.zeros(3, I16).at[0].set(dur.astype(I16)),
                                                            f.weather_state)))
    hc = HookCtx.make(-1, source, source_effect).replace(holder=10)
    hc, st, rng = CO.single_event(ids.HOOK_START, e, st, dex, rng, hc, go, prefix=ids.PFX_FIELD)
    started = (hc.ret_t == RS.UNDEF) | RS.truthy((hc.ret_t, hc.ret_v))
    f = st.field
    st = st.replace(field=f.replace(weather=jnp.where(go & ~started, prev_w, f.weather),
                                    weather_state=jnp.where(go & ~started, prev_s, f.weather_state)))
    return st, rng, RS.where(go & started, RS.R(RS.BOOL, 1), RS.R(RS.BOOL, 0))


# ------------------------------------------------------------------ eachEvent('Weather')
def each_weather(st, dex, rng, on):
    """PS: sim/battle.ts:470-480 eachEvent('Weather'): the unfainted actives speed-sorted (SPEED_TIE_WEATHER_EACH,
    sub = eachevent_ord), runEvent('Weather', pokemon, null, weather) on each; eachEvent('Update') follows (no Reg
    M-C handler yet). The Reg M-C Weather handlers are fixed data, applied without dispatch (at most one applies to a
    mon, so their order cannot matter; every Weather handler is skipped while weather is suppressed, :895-900):
    sandstorm onWeather (baseMaxhp / 16 unless weather-immune: runStatusImmunity('sandstorm'), Sand Force) and Solar
    Power onWeather (Sun: baseMaxhp / 8, '[from] ability: Solar Power' '[of] target'). Returns (st, rng)."""
    from vgcjax.battle import switching as SW
    from vgcjax.effects import statuses as STS
    on = jnp.asarray(on)
    act = st.sides.active.reshape(4).astype(I32)
    refs = jnp.clip((jnp.arange(4) // 2) * 4 + act, 0, 7)
    alive = (act >= 0) & ~mget(st.mons.fainted, refs)
    sc = st.scratch
    rng, order, refs = SW.slot_speed_sort(st, rng, alive, ids.DK_SPEED_TIE_WEATHER_EACH, sc.eachevent_ord, on)
    st = st.replace(scratch=sc.replace(eachevent_ord=sc.eachevent_ord + on.astype(sc.eachevent_ord.dtype)))
    w = st.field.weather.astype(I32)
    run = on & ~suppressing(st)
    solar = getattr(ids, "ABILITY_SOLARPOWER", -1)
    sun_eff = ids.ABILITY_BASE + solar - 1
    for i in range(4):
        k = order[i]
        r = refs[k]
        ok = run & alive[k]
        sand = ok & (w == wid("sandstorm")) & STS.run_status_immunity(st, dex, r, wid("sandstorm"))
        st, rng = STS.effect_damage(st, dex, rng, r, mget(st.mons.maxhp, r) // 16, CO.eff("sandstorm"), sand)
        if solar > 0:
            sp = ok & (w == wid("sunnyday")) & (mon_weather(st, r) == w) & (mget(st.mons.ability, r) == solar)
            st, rng = STS.effect_damage(st, dex, rng, r, mget(st.mons.maxhp, r) // 8, sun_eff, sp, source=r)
    return st, rng


def residual_each(st, dex, rng, c, on):
    """The eachEvent('Weather') part of each weather's onFieldResidual, run by the fieldEvent walk right after the
    handler (outside its dispatch): Sun / Rain always, Sandstorm / Snowscape only if isWeather (data/conditions.ts)."""
    gated = (c == wid("sandstorm")) | (c == wid("snowscape"))
    return each_weather(st, dex, rng, jnp.asarray(on) & (~gated | (field_weather(st) == c)))


# ------------------------------------------------------------------ the weather conditions
def _rock_duration(name):
    item = getattr(ids, f"ITEM_{ROCKS[name]}", -1)

    def fn(hc, st, dex, rng):
        """durationCallback(source): source holding the weather's rock -> 8, else 5."""
        src = jnp.asarray(hc.source, I32)
        n = jnp.where((src >= 0) & (mget(st.mons.item, jnp.maximum(src, 0)) == item), 8, 5)
        return hc.ret(RS.R(RS.NUM, n)), st, rng
    return fn


def _field_start(name):
    def fn(hc, st, dex, rng):
        """'-weather' Name, with '[from] ability: X' '[of] source' for an ability source."""
        ab = (hc.effect >= ids.ABILITY_BASE) & (hc.effect < ids.ABILITY_BASE + len(ids.ABILITIES) - 1)
        return hc, EV.emit(st, True, EV.EV_WEATHER, actor=jnp.where(ab, hc.source, -1), move=CO.eff(name),
                           v1=jnp.where(ab, hc.effect, -1), flags=jnp.where(ab, EV.F_FROM_ABILITY, 0)), rng
    return fn


def _field_residual(name, gated):
    def fn(hc, st, dex, rng):
        """'-weather' Name '[upkeep]'; eachEvent('Weather') follows in the walk (residual_each)."""
        return hc, EV.emit(st, True, EV.EV_WEATHER, move=CO.eff(name), v1=-1, flags=EV.F_UPKEEP), rng
    return fn


def _field_end(hc, st, dex, rng):
    """'-weather' 'none'."""
    return hc, EV.emit(st, True, EV.EV_WEATHER, move=-1, v1=-1), rng


for _w in WEATHERS:
    if _w not in ids.CONDITIONS:
        continue
    _e = CO.eff(_w)
    handler(CO.DURATION_CALLBACK, _e, ps=f"data/conditions.ts {_w}.durationCallback")(_rock_duration(_w))
    handler(ids.HOOK_START, _e, ps=f"data/conditions.ts {_w}.onFieldStart", prefix=ids.PFX_FIELD)(_field_start(_w))
    handler(ids.HOOK_RESIDUAL, _e, ps=f"data/conditions.ts {_w}.onFieldResidual", prefix=ids.PFX_FIELD)(
        _field_residual(_w, _w in ("sandstorm", "snowscape")))
    handler(ids.HOOK_END, _e, ps=f"data/conditions.ts {_w}.onFieldEnd", prefix=ids.PFX_FIELD)(_field_end)


def _def_boost(name, ty):
    def fn(ctx, h):
        """onModifySpD (sandstorm, Rock) / onModifyDef (snowscape, Ice), priority 10: this.modify(stat, 1.5) (a new
        relay, applied before the chained modifiers); skipped while weather is suppressed (sim/battle.ts:895-900)."""
        d = ctx.tgt
        ok = jnp.any(mget(ctx.st.mons.types, jnp.clip(d, 0, 7)) == ty, axis=-1) & (mon_weather(ctx.st, d) == wid(name))
        return ok & ~suppressing(ctx.st), None, (lambda v: fp.modify(v, 3, 2))
    return fn


if "sandstorm" in ids.CONDITIONS:
    MOD.mod("ModifySpD", CO.eff("sandstorm"), ps="data/conditions.ts sandstorm.onModifySpD")(
        _def_boost("sandstorm", ids.TYPE_ROCK))
if "snowscape" in ids.CONDITIONS:
    MOD.mod("ModifyDef", CO.eff("snowscape"), ps="data/conditions.ts snowscape.onModifyDef")(
        _def_boost("snowscape", ids.TYPE_ICE))
