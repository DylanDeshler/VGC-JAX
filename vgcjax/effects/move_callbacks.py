"""Moves with bespoke formula callbacks (data/moves.ts; Champions overrides in data/mods/champions/moves.ts where
noted): basePowerCallback, damageCallback, onModifyPriority, onTry / onTryHit, onModifyType / onModifyMove.
Fixed-data per move id, so they are evaluated at their formula sites without dispatch (perf rule).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle.util import mget
from vgcjax.data import ids
from vgcjax.effects import conditions as CO

I32 = jnp.int32


def mid(name):
    return getattr(ids, f"MOVE_{name.upper()}", -999)


def _weight_bp(w):
    """Low Kick / Grass Knot: target weight (hg) thresholds 2000 / 1000 / 500 / 250 / 100."""
    return jnp.where(w >= 2000, 120, jnp.where(w >= 1000, 100, jnp.where(w >= 500, 80, jnp.where(
        w >= 250, 60, jnp.where(w >= 100, 40, 20)))))


def _ratio_bp(uw, tw):
    """Heavy Slam / Heat Crash: user weight vs target weight x5 / x4 / x3 / x2."""
    return jnp.where(uw >= tw * 5, 120, jnp.where(uw >= tw * 4, 100, jnp.where(uw >= tw * 3, 80, jnp.where(
        uw >= tw * 2, 60, 40))))


def base_power(st, dex, u, move, targets, bp):
    """move.basePowerCallback (getDamage, sim/battle-actions.ts:1662-1668), per target slot; `bp` = the ActiveMove
    basePower. The result is clamped to >= 1 by the caller's existing path (falsy -> undefined)."""
    m = st.mons
    t = jnp.clip(targets, 0, 7)
    uw = mget(m.weight_hg, u).astype(I32)
    tw = mget(m.weight_hg, t).astype(I32)
    out = jnp.broadcast_to(jnp.asarray(bp, I32), targets.shape)
    is_ = lambda n: move == mid(n)  # noqa: E731
    out = jnp.where(is_("lowkick") | is_("grassknot"), _weight_bp(tw), out)
    out = jnp.where(is_("heavyslam") | is_("heatcrash"), _ratio_bp(uw, tw), out)
    out = jnp.where(is_("lastrespects"), 50 + 50 * st.sides.total_fainted[jnp.clip(u, 0, 7) // 4].astype(I32), out)
    failed_last = mget(m.move_last_turn_result, u) == 2     # moveLastTurnResult === false (state.MR_FALSE)
    out = jnp.where((is_("stompingtantrum") | is_("temperflare")) & failed_last, bp * 2, out)
    hp, mx = mget(m.hp, u).astype(I32), mget(m.maxhp, u).astype(I32)
    out = jnp.where(is_("eruption") | is_("waterspout"), jnp.maximum((bp * hp) // jnp.maximum(mx, 1), 1), out)
    out = jnp.where(is_("acrobatics") & (mget(m.item, u) == 0), bp * 2, out)
    pos = jnp.sum(jnp.maximum(mget(m.boosts, u).astype(I32), 0))
    out = jnp.where(is_("powertrip") | is_("storedpower"), bp + 20 * pos, out)
    out = jnp.where(is_("ragefist"), jnp.minimum(350, 50 + 50 * mget(m.times_attacked, u).astype(I32)), out)
    if "electricterrain" in ids.CONDITIONS:
        rv = is_("risingvoltage") & (st.field.terrain == CO.cid("electricterrain")) & CO.is_grounded(st, t)
        out = jnp.where(rv, bp * 2, out)
    return out


def fixed_damage(st, dex, u, move, targets, fixed):
    """move.damageCallback (getDamage returns it directly: no crit / roll): Super Fang (max(1, floor(hp / 2)))."""
    t = jnp.clip(targets, 0, 7)
    hp = mget(st.mons.hp, t).astype(I32)
    out = jnp.broadcast_to(jnp.asarray(fixed, I32), targets.shape)
    return jnp.where(move == mid("superfang"), jnp.maximum(hp // 2, 1), out)


def priority_bonus(st, dex, ref, move):
    """onModifyPriority: Grassy Glide (+1 in Grassy Terrain for a grounded user)."""
    if "grassyterrain" not in ids.CONDITIONS:
        return jnp.zeros(jnp.shape(move), I32)
    gg = (move == mid("grassyglide")) & (st.field.terrain == CO.cid("grassyterrain")) & CO.is_grounded(st, ref)
    return gg.astype(I32)


def always_hits(st, u, move):
    """hitStepAccuracy's built-in alwaysHit cases: Toxic from a Poison-type user (gen >= 8)."""
    return (move == mid("toxic")) & jnp.any(mget(st.mons.types, u) == ids.TYPE_POISON)


def terrain_pulse(st, dex, u, move, mtype, bp):
    """Terrain Pulse onModifyType / onModifyMove: a grounded user in a terrain -> its type, basePower x2."""
    tp = (move == mid("terrainpulse")) & CO.is_grounded(st, u) & (st.field.terrain > 0)
    t = st.field.terrain.astype(I32)
    ty = mtype
    for name, typ in (("electricterrain", ids.TYPE_ELECTRIC), ("grassyterrain", ids.TYPE_GRASS),
                      ("mistyterrain", ids.TYPE_FAIRY), ("psychicterrain", ids.TYPE_PSYCHIC)):
        if name in ids.CONDITIONS:
            ty = jnp.where(tp & (t == CO.cid(name)), typ, ty)
    return ty, jnp.where(tp, bp * 2, bp)


# ------------------------------------------------------------------ dispatched single-effect handlers
from vgcjax.battle import common as C  # noqa: E402
from vgcjax.battle import results as RS  # noqa: E402
from vgcjax.effects.hooks import handler  # noqa: E402

if mid("suckerpunch") > 0:
    @handler(ids.HOOK_TRY, int(C.eff_move(mid("suckerpunch"))), ps="data/moves.ts suckerpunch.onTry")
    def suckerpunch_try(hc, st, dex, rng):
        """The target's queued action must be a move that is not Status (Me First excepted) and the target must not
        be recharging; else false."""
        from vgcjax.battle import queue as Q
        from vgcjax.battle import state as S
        tgt = hc.source                    # singleEvent('Try', move, null, pokemon, target): onTry(source, target)
        q = st.queue
        mine = q.valid & (q.kind == S.Q_MOVE) & (Q.entry_ref(q) == tgt)
        mv = jnp.max(jnp.where(mine, q.move_id.astype(I32), 0))
        status = (dex.move_category[mv] == ids.CAT_STATUS) & (mv != mid("mefirst"))
        rech = CO.has_volatile(st, tgt, CO.cid("mustrecharge")) if "mustrecharge" in ids.CONDITIONS else False
        fail = ~jnp.any(mine) | status | rech
        return hc.ret(RS.R(RS.BOOL, 0), fail), st, rng


for _n in ("psychicfangs", "brickbreak"):
    if mid(_n) > 0:
        def _break_screens(hc, st, dex, rng):
            """onTryHit(pokemon): the target's side loses Reflect, Light Screen, Aurora Veil (before the hit)."""
            from vgcjax.effects import sideconds as SC
            side = jnp.clip(hc.target, 0, 7) // 4
            for c in ("reflect", "lightscreen", "auroraveil"):
                if c in ids.CONDITIONS:
                    st, rng = SC.remove_side_condition(st, dex, rng, side, CO.cid(c), True)
            return hc, st, rng
        handler(ids.HOOK_TRYHIT, int(C.eff_move(mid(_n))), ps=f"data/moves.ts {_n}.onTryHit")(_break_screens)
