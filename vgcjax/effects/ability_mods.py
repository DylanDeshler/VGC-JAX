"""Abilities whose only callbacks are MUL-hook modifiers (effects/modifiers.py). PS: data/abilities.ts (no Champions
overrides for these). Ctx per hook (event target / source): BasePower, ModifyAtk / SpA, ModifySTAB, ModifyDamage:
attacker / defender; ModifyDef / SpD, ModifyAccuracy: defender / attacker. ModifyDamage ctx.extra: crit, type_mod.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle.util import mget
from vgcjax.core import fixedpoint as fp
from vgcjax.data import ids
from vgcjax.effects.abilities import aeff, aid
from vgcjax.effects.modifiers import mod

I32 = jnp.int32
X15, X2, X05, X075 = 6144, 8192, 2048, 3072


def _flag(ctx, flag):
    w, b = flag
    return ((ctx.dex.move_flags[ctx.move, w] >> b) & 1) == 1


def _reg(hook, name, prefix=ids.PFX_ON):
    def deco(fn):
        if aid(name) > 0:
            mod(hook, aeff(name), ps=f"data/abilities.ts {name}.on{['', 'Source', 'Ally', 'Any', 'Foe'][_PFX[prefix]]}"
                f"{hook}", prefix=prefix)(fn)
        return fn
    return deco


_PFX = {ids.PFX_ON: 0, ids.PFX_SOURCE: 1, ids.PFX_ALLY: 2, ids.PFX_ANY: 3, ids.PFX_FOE: 4}


def _const(m):
    return jnp.asarray(m, I32)


def _cat(ctx):
    return ctx.dex.move_category[ctx.move]


# ------------------------------------------------------------------ BasePower (attacker = event target)
@_reg("BasePower", "technician")
def technician(ctx, h):
    """priority 30: modify(basePower, event.modifier) <= 60 -> chainModify(1.5)."""
    return jnp.ones(h.shape, bool), (lambda t, val, modifier: jnp.where(fp.modify_raw(val, modifier) <= 60, X15,
                                                                           4096)), None


for _n, _flagname, _m in (("toughclaws", "MF_CONTACT", 5325), ("ironfist", "MF_PUNCH", 4915),
                          ("sharpness", "MF_SLICING", X15), ("megalauncher", "MF_PULSE", X15),
                          ("strongjaw", "MF_BITE", X15), ("punkrock", "MF_SOUND", 5325)):
    def _fn(ctx, h, _flagname=_flagname, _m=_m):
        """move.flags[...] -> chainModify(m)."""
        return jnp.broadcast_to(_flag(ctx, getattr(ids, _flagname)), h.shape), _const(_m), None
    _reg("BasePower", _n)(_fn)


@_reg("BasePower", "reckless")
def reckless(ctx, h):
    """move.recoil || move.hasCrashDamage -> chainModify([4915, 4096])."""
    d = ctx.dex
    rec = (d.move_recoil[ctx.move, 0] > 0) | (d.move_has_crash_damage[ctx.move] == 1)
    return jnp.broadcast_to(rec, h.shape), _const(4915), None


@_reg("BasePower", "sandforce")
def sandforce(ctx, h):
    """Sandstorm and a Rock / Ground / Steel move -> chainModify([5325, 4096])."""
    from vgcjax.effects import weather as WE
    t = ctx.mtype
    ok = WE.is_weather(ctx.st, "sandstorm") & ((t == ids.TYPE_ROCK) | (t == ids.TYPE_GROUND) | (t == ids.TYPE_STEEL))
    return jnp.broadcast_to(ok, h.shape), _const(5325), None


# ------------------------------------------------------------------ ModifyAtk / ModifySpA (attacker)
for _n in ("hugepower", "purepower"):
    _reg("ModifyAtk", _n)(lambda ctx, h: (jnp.ones(h.shape, bool), _const(X2), None))


@_reg("ModifyAtk", "hustle")
def hustle_atk(ctx, h):
    """this.modify(atk, 1.5): applied to the stat directly."""
    return jnp.ones(h.shape, bool), None, (lambda v: fp.modify(v, 3, 2))


@_reg("ModifyAtk", "guts")
def guts(ctx, h):
    return mget(ctx.st.mons.status, jnp.clip(h, 0, 7)) > 0, _const(X15), None


for _n, _ty in (("blaze", "TYPE_FIRE"), ("torrent", "TYPE_WATER"), ("overgrow", "TYPE_GRASS"), ("swarm", "TYPE_BUG")):
    def _pinch(ctx, h, _ty=_ty):
        """move.type and attacker.hp <= attacker.maxhp / 3 -> chainModify(1.5)."""
        m = ctx.st.mons
        r = jnp.clip(h, 0, 7)
        low = mget(m.hp, r).astype(I32) * 3 <= mget(m.maxhp, r).astype(I32)
        return low & (ctx.mtype == getattr(ids, _ty)), _const(X15), None
    _reg("ModifyAtk", _n)(_pinch)
    _reg("ModifySpA", _n)(_pinch)


for _hook in ("ModifyAtk", "ModifySpA"):
    @_reg(_hook, "thickfat", prefix=ids.PFX_SOURCE)
    def thickfat(ctx, h):
        """onSourceModifyAtk / SpA (the defender's): Ice / Fire -> chainModify(0.5)."""
        ok = (ctx.mtype == ids.TYPE_ICE) | (ctx.mtype == ids.TYPE_FIRE)
        return jnp.broadcast_to(ok, h.shape), _const(X05), None


@_reg("ModifySpA", "solarpower")
def solarpower_spa(ctx, h):
    from vgcjax.effects import weather as WE
    return jnp.broadcast_to(WE.mon_weather(ctx.st, h) == WE.wid("sunnyday"), h.shape), _const(X15), None


# ------------------------------------------------------------------ ModifyDamage (attacker = event target)
@_reg("ModifyDamage", "multiscale", prefix=ids.PFX_SOURCE)
def multiscale(ctx, h):
    """onSourceModifyDamage (the defender's): target.hp >= target.maxhp -> chainModify(0.5)."""
    m = ctx.st.mons
    r = jnp.clip(h, 0, 7)
    return mget(m.hp, r) >= mget(m.maxhp, r), _const(X05), None


for _n in ("filter", "solidrock"):
    _reg("ModifyDamage", _n, prefix=ids.PFX_SOURCE)(
        lambda ctx, h: (ctx.extra["type_mod"] > 0, _const(X075), None))


@_reg("ModifyDamage", "punkrock", prefix=ids.PFX_SOURCE)
def punkrock_dmg(ctx, h):
    return jnp.broadcast_to(_flag(ctx, ids.MF_SOUND), h.shape), _const(X05), None


@_reg("ModifyDamage", "friendguard", prefix=ids.PFX_ANY)
def friendguard(ctx, h):
    """onAnyModifyDamage(damage, source, target): target !== holder and an ally of it -> chainModify(0.75)."""
    d = ctx.src
    ok = (d >= 0) & (d != h) & ((jnp.clip(d, 0, 7) // 4) == (jnp.clip(h, 0, 7) // 4))
    return ok, _const(X075), None


@_reg("ModifyDamage", "sniper")
def sniper(ctx, h):
    return ctx.extra["crit"], _const(X15), None


# ------------------------------------------------------------------ ModifyAccuracy (defender = event target)
@_reg("ModifyAccuracy", "compoundeyes", prefix=ids.PFX_SOURCE)
def compoundeyes(ctx, h):
    """onSourceModifyAccuracy (the attacker's), numeric accuracy -> chainModify([5325, 4096])."""
    return ctx.extra["numeric"], _const(5325), None


@_reg("ModifyAccuracy", "hustle", prefix=ids.PFX_SOURCE)
def hustle_acc(ctx, h):
    """Physical move with numeric accuracy -> chainModify([3277, 4096])."""
    return ctx.extra["numeric"] & (_cat(ctx) == ids.CAT_PHYSICAL), _const(3277), None


# ------------------------------------------------------------------ ModifySTAB (attacker); relay = STAB x4096
@_reg("ModifySTAB", "adaptability")
def adaptability(ctx, h):
    """move.forceSTAB || source.hasType(move.type): 2 (2.25 when the STAB is already 2)."""
    d = ctx.dex
    stab = jnp.any(mget(ctx.st.mons.types, jnp.clip(h, 0, 7)) == ctx.mtype, axis=-1) | (d.move_force_stab[ctx.move] == 1)
    return stab, None, (lambda v: jnp.where(v == X2, 9216, X2))


# Solar Power onWeather: effects/weather.py each_weather


# ------------------------------------------------------------------ Water Bubble / Purifying Salt (status parts:
# effects/ability_blocks.py)
for _hook in ("ModifyAtk", "ModifySpA"):
    _reg(_hook, "waterbubble", prefix=ids.PFX_SOURCE)(
        lambda ctx, h: (jnp.broadcast_to(ctx.mtype == ids.TYPE_FIRE, h.shape), _const(X05), None))
    _reg(_hook, "waterbubble")(
        lambda ctx, h: (jnp.broadcast_to(ctx.mtype == ids.TYPE_WATER, h.shape), _const(X2), None))
    _reg(_hook, "purifyingsalt", prefix=ids.PFX_SOURCE)(
        lambda ctx, h: (jnp.broadcast_to(ctx.mtype == ids.TYPE_GHOST, h.shape), _const(X05), None))
