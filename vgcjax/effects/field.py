"""Speed control and screens (05 §4 steps 4-5): Tailwind, Trick Room, Reflect, Light Screen.

PS: data/moves.ts tailwind / trickroom / reflect / lightscreen conditions (no Champions overrides). Trick Room's
speed effect is getActionSpeed's negation ([C] scripts.ts:46-54, battle/common.py action_speed); Tailwind's
onModifySpe is the explicit chain in battle/common.py spe_chain; the screens' onAnyModifyDamage are
effects/modifiers.py mods.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects import modifiers as MOD
from vgcjax.effects.hooks import handler

I32 = jnp.int32


def _side_msg(kind, name):
    def fn(hc, st, dex, rng):
        return hc, EV.emit(st, True, kind, actor=hc.holder - 8, move=CO.eff(name)), rng
    return fn


def _const_duration(n):
    def fn(hc, st, dex, rng):
        return hc.ret(RS.R(RS.NUM, n)), st, rng
    return fn


def _light_clay(hc, st, dex, rng):
    """durationCallback(target, source): source holding Light Clay -> 8, else 5."""
    clay = getattr(ids, "ITEM_LIGHTCLAY", -1)
    src = jnp.asarray(hc.source, I32)
    n = jnp.where((src >= 0) & (mget(st.mons.item, jnp.maximum(src, 0)) == clay), 8, 5)
    return hc.ret(RS.R(RS.NUM, n)), st, rng


# Tailwind (Persistent is not in Reg M-C: durationCallback returns 4)
handler(CO.DURATION_CALLBACK, CO.eff("tailwind"), ps="data/moves.ts tailwind.condition.durationCallback")(_const_duration(4))
handler(ids.HOOK_START, CO.eff("tailwind"), ps="data/moves.ts tailwind.condition.onSideStart",
        prefix=ids.PFX_SIDE)(_side_msg(EV.EV_SIDESTART, "tailwind"))
handler(ids.HOOK_END, CO.eff("tailwind"), ps="data/moves.ts tailwind.condition.onSideEnd",
        prefix=ids.PFX_SIDE)(_side_msg(EV.EV_SIDEEND, "tailwind"))

# Reflect / Light Screen
for _n in ("reflect", "lightscreen"):
    handler(CO.DURATION_CALLBACK, CO.eff(_n), ps=f"data/moves.ts {_n}.condition.durationCallback")(_light_clay)
    handler(ids.HOOK_START, CO.eff(_n), ps=f"data/moves.ts {_n}.condition.onSideStart",
            prefix=ids.PFX_SIDE)(_side_msg(EV.EV_SIDESTART, _n))
    handler(ids.HOOK_END, CO.eff(_n), ps=f"data/moves.ts {_n}.condition.onSideEnd",
            prefix=ids.PFX_SIDE)(_side_msg(EV.EV_SIDEEND, _n))

def _screen(cat):
    def fn(ctx, h):
        """onAnyModifyDamage(damage, source, target, move): target !== source, target on the screen's side, the
        category, not a crit (Infiltrator: M4) -> chainModify([2732, 4096]) (activePerHalf 2)."""
        d, a = ctx.src, ctx.tgt
        ok = (d >= 0) & (d != a) & ((jnp.clip(d, 0, 7) // 4) == (h - 8)) & (ctx.dex.move_category[ctx.move] == cat) & \
            ~ctx.extra["crit"]
        return ok, jnp.asarray(2732, I32), None
    return fn


for _n, _cat in (("reflect", ids.CAT_PHYSICAL), ("lightscreen", ids.CAT_SPECIAL)):
    MOD.mod("ModifyDamage", CO.eff(_n), ps=f"data/moves.ts {_n}.condition.onAnyModifyDamage", prefix=ids.PFX_ANY)(
        _screen(_cat))

# Trick Room
handler(CO.DURATION_CALLBACK, CO.eff("trickroom"), ps="data/moves.ts trickroom.condition.durationCallback")(
    _const_duration(5))


@handler(ids.HOOK_START, CO.eff("trickroom"), ps="data/moves.ts trickroom.condition.onFieldStart", prefix=ids.PFX_FIELD)
def trickroom_start(hc, st, dex, rng):
    return hc, EV.emit(st, True, EV.EV_FIELDSTART, move=CO.eff("trickroom"), target=hc.source), rng


@handler(ids.HOOK_RESTART, CO.eff("trickroom"), ps="data/moves.ts trickroom.condition.onFieldRestart",
         prefix=ids.PFX_FIELD)
def trickroom_restart(hc, st, dex, rng):
    from vgcjax.effects import sideconds as SC
    st, rng = SC.remove_pseudo_weather(st, dex, rng, CO.cid("trickroom"), True)
    return hc, st, rng


@handler(ids.HOOK_END, CO.eff("trickroom"), ps="data/moves.ts trickroom.condition.onFieldEnd", prefix=ids.PFX_FIELD)
def trickroom_end(hc, st, dex, rng):
    return hc, EV.emit(st, True, EV.EV_FIELDEND, move=CO.eff("trickroom")), rng
