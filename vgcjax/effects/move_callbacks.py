"""Moves with bespoke formula callbacks (data/moves.ts; Champions overrides in data/mods/champions/moves.ts where
noted): basePowerCallback, damageCallback, onModifyPriority, onTry / onTryHit, onModifyType / onModifyMove.
Fixed-data per move id, so they are evaluated at their formula sites without dispatch (perf rule).
"""
from __future__ import annotations

import jax
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


def base_power(st, dex, u, move, targets, bp, hit=1):
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
    # Assurance (target.hurtThisTurn), Hex (target status), Payback (the target already moved and did not just switch)
    out = jnp.where(is_("assurance") & (mget(m.hurt_this_turn, t) > 0), bp * 2, out)
    out = jnp.where(is_("hex") & (mget(m.status, t) > 0), bp * 2, out)
    if mid("payback") > 0:
        from vgcjax.effects import disables as DS
        moved = ~mget(m.newly_switched, t) & ~jax.vmap(lambda r: DS.will_move(st, r))(t)
        out = jnp.where(is_("payback") & moved, bp * 2, out)
    # Hard Press: floor(floor((100 * (100 * floor(hp * 4096 / maxhp)) + 2047) / 4096) / 100) || 1
    thp, tmx = mget(m.hp, t).astype(I32), jnp.maximum(mget(m.maxhp, t).astype(I32), 1)
    hpb = ((100 * (100 * ((thp * 4096) // tmx)) + 2047) // 4096) // 100
    out = jnp.where(is_("hardpress"), jnp.maximum(hpb, 1), out)
    # Gyro Ball: floor(25 * target.getStat('spe') / pokemon.getStat('spe')) + 1, capped at 150 (division by 0 -> 1)
    if mid("gyroball") > 0:
        from vgcjax.battle import common as C_
        us = C_.get_stat(st, dex, u, 4).astype(I32)
        ts = jax.vmap(lambda r: C_.get_stat(st, dex, r, 4))(t).astype(I32)
        gb = jnp.where(us > 0, jnp.minimum((25 * ts) // jnp.maximum(us, 1) + 1, 150), 1)
        out = jnp.where(is_("gyroball"), gb, out)
    out = jnp.where(is_("tripleaxel"), 20 * jnp.asarray(hit, I32), out)            # 20 * move.hit
    return out


def fixed_damage(st, dex, u, move, targets, fixed):
    """move.damageCallback (getDamage returns it directly: no crit / roll): Super Fang (max(1, floor(hp / 2)))."""
    t = jnp.clip(targets, 0, 7)
    hp = mget(st.mons.hp, t).astype(I32)
    uhp = mget(st.mons.hp, jnp.clip(u, 0, 7)).astype(I32)
    out = jnp.broadcast_to(jnp.asarray(fixed, I32), targets.shape)
    out = jnp.where(move == mid("finalgambit"), uhp, out)              # pokemon.hp (faint: selfdestruct 'ifHit')
    out = jnp.where(move == mid("endeavor"), hp - uhp, out)           # target hp - pokemon hp (TryImmunity: >0)
    return jnp.where(move == mid("superfang"), jnp.maximum(hp // 2, 1), out)


def priority_bonus(st, dex, ref, move):
    """onModifyPriority: Grassy Glide (+1 in Grassy Terrain for a grounded user); Prankster (+1 for a Status move,
    data/abilities.ts prankster: move.pranksterBoosted); Gale Wings (+1 for a Flying move at full HP)."""
    out = jnp.zeros(jnp.shape(move), I32)
    if "grassyterrain" in ids.CONDITIONS:
        gg = (move == mid("grassyglide")) & (st.field.terrain == CO.cid("grassyterrain")) & CO.is_grounded(st, ref)
        out = out + gg.astype(I32)
    gw = getattr(ids, "ABILITY_GALEWINGS", -1)
    if gw > 0:                                           # Gale Wings: a Flying move at full HP
        r = jnp.clip(ref, 0, 7)
        m = st.mons
        out = out + ((mget(m.ability, r) == gw) & (dex.move_type[move] == ids.TYPE_FLYING) &
                     (mget(m.hp, r) == mget(m.maxhp, r))).astype(I32)
    return out + prankster_boosted(st, dex, ref, move).astype(I32)


def prankster_boosted(st, dex, ref, move):
    """move.pranksterBoosted: the user's Prankster raised a Status move's priority (getActionSpeed; the flag stays on
    the queued ActiveMove, sim/battle-actions.ts:281-290)."""
    a = getattr(ids, "ABILITY_PRANKSTER", -1)
    if a <= 0:
        return jnp.zeros(jnp.shape(move), bool)
    return (mget(st.mons.ability, jnp.clip(ref, 0, 7)) == a) & (dex.move_category[move] == ids.CAT_STATUS)


def sheer_force(st, dex, u, move):
    """Sheer Force onModifyMove: a move with secondaries (not hasSheerForceBoost) loses them and its `self`
    (move.hasSheerForce). Returns (suppressed (= battle.suppressingSecondaries for this move), BasePower boosted)."""
    a = getattr(ids, "ABILITY_SHEERFORCE", -1)
    if a <= 0:
        z = jnp.zeros((), bool)
        return z, z
    holder = mget(st.mons.ability, jnp.clip(u, 0, 7)) == a
    boost = dex.move_has_sheer_force[move] == 1
    sup = holder & (dex.move_nsec[move] > 0) & ~boost
    return sup, holder & (sup | boost)


ATE = {"aerilate": "FLYING", "pixilate": "FAIRY"}
ATE_EXCLUDED = ("judgment", "multiattack", "naturalgift", "revelationdance", "technoblast", "terrainpulse", "weatherball")


def ate_boosted(st, dex, u, move):
    """-ate abilities' onModifyType condition (move.typeChangerBoosted): a Normal move (base type; not in the excluded
    list) used by an Aerilate / Pixilate holder. Returns (boosted, new type)."""
    ab = mget(st.mons.ability, jnp.clip(u, 0, 7))
    # Struggle's onModifyMove (singleEvent, PS battle-actions.ts:495) sets type '???' before runEvent('ModifyType')
    normal = (dex.move_type[move] == ids.TYPE_NORMAL) & (move != ids.MOVE_STRUGGLE)
    for n in ATE_EXCLUDED:
        if mid(n) > 0:
            normal = normal & (move != mid(n))
    boosted = jnp.zeros((), bool)
    ty = jnp.int32(0)
    for name, t in ATE.items():
        a = getattr(ids, f"ABILITY_{name.upper()}", -1)
        if a > 0:
            hit = normal & (ab == a)
            boosted = boosted | hit
            ty = jnp.where(hit, getattr(ids, f"TYPE_{t}"), ty)
    return boosted, ty


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


def expanding_force(st, u, move):
    """Expanding Force onModifyMove / onBasePower condition: Psychic Terrain and a grounded user."""
    if mid("expandingforce") < 0 or "psychicterrain" not in ids.CONDITIONS:
        return jnp.zeros((), bool)
    return (move == mid("expandingforce")) & (st.field.terrain.astype(I32) == CO.cid("psychicterrain")) & \
        CO.is_grounded(st, u)


def move_target(st, dex, u, move):
    """The ActiveMove's target after onModifyMove: Expanding Force -> 'allAdjacentFoes'."""
    tt = dex.move_target[move].astype(I32)
    return jnp.where(expanding_force(st, u, move), ids.TARGET_ALLADJACENTFOES, tt)


# ------------------------------------------------------------------ dispatched single-effect handlers
from vgcjax.battle import common as C  # noqa: E402
from vgcjax.battle import results as RS  # noqa: E402
from vgcjax.effects.hooks import handler  # noqa: E402
from vgcjax.effects.modifiers import mod  # noqa: E402

if mid("expandingforce") > 0:
    @mod("BasePower", int(C.eff_move(mid("expandingforce"))), ps="data/moves.ts expandingforce.onBasePower")
    def expandingforce_bp(ctx, h):
        """Psychic Terrain and a grounded user -> chainModify(1.5)."""
        import jax
        ok = jax.vmap(lambda r: expanding_force(ctx.st, r, ctx.move))(jnp.clip(ctx.tgt, 0, 7))
        return ok & (ctx.tgt >= 0), jnp.asarray(6144, I32), None

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
