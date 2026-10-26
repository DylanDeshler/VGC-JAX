"""Locked moves (getLockedMove, sim/pokemon.ts:970-973): recharge (mustrecharge: Hyper Beam family) and charge
(twoturnmove: Electro Shot, Solar Beam, Solar Blade, Sky Attack, Meteor Beam) moves. data/conditions.ts mustrecharge /
twoturnmove, data/moves.ts <move>.onTryMove (no Champions overrides).

A locked Pokemon's request offers one option and keeps it in (getMoveRequestData: trapped, no Mega); the choice is
forced (sim/side.ts:675-690): the locked move at lastMoveTargetLoc, or the move volatile's targetLoc. A locked move
deducts no PP and pays no Pressure (sim/battle-actions.ts:336-347, :530-547).

Storage: twoturnmove's effectState is {move} (slot 1); the move volatile PS adds alongside it (attacker.addVolatile(
move.id): a Condition with exists false for moves without a `condition`) lives in the same state: slot 2 = its
targetLoc, slot 3 = present. harness/project.py projects it as its own volatile.

Recharge: the forced action's move is 'recharge' (no move data: priority 0, target undefined). The engine queues the
recharging move itself (every recharge move has priority 0, is not Status and not Flying, so ModifyPriority agrees)
and draws no target (getRandomTarget -> side.randomFoe, a RANDOM_TARGET_NOMINAL don't-care draw, contract §6).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32
CO.fields("twoturnmove", ("move", "move"))
TT_LOC, TT_PRESENT = 2, 3                    # raw twoturnmove slots of the move volatile (no presence bits)


def mid(name):
    return getattr(ids, f"MOVE_{name.upper()}", -999)


# onTryMove: (name, boost spa +1, weather that skips the charge turn)
CHARGE_MOVES = tuple((n, b, w) for n, b, w in (
    ("electroshot", True, ("raindance", "primordialsea")),
    ("solarbeam", False, ("sunnyday", "desolateland")),
    ("solarblade", False, ("sunnyday", "desolateland")),
    ("meteorbeam", True, ()),
    ("skyattack", False, ())) if mid(n) > 0)
HAS_RECHARGE = "mustrecharge" in ids.CONDITIONS
HAS_TWOTURN = "twoturnmove" in ids.CONDITIONS


def locked(st, u):
    """getLockedMove() and the forced choice (sim/side.ts:675-690). Returns (locked, recharge, move_id, target_loc)."""
    m = st.mons
    rc = CO.has_volatile(st, u, CO.cid("mustrecharge")) if HAS_RECHARGE else jnp.zeros((), bool)
    if HAS_TWOTURN:
        c = CO.cid("twoturnmove")
        tw = CO.has_volatile(st, u, c)
        s = CO.vol_state(st, u, c).astype(I32)
        tw_move, tw_loc = s[CO.slot_of("twoturnmove", "move")], jnp.where(s[TT_PRESENT] != 0, s[TT_LOC], 0)
    else:
        tw = jnp.zeros((), bool)
        tw_move = tw_loc = jnp.int32(0)
    last_loc = mget(m.last_move_target_loc, u).astype(I32)
    move = jnp.where(rc, mget(m.last_move, u).astype(I32), tw_move)
    loc = jnp.where(rc, last_loc, jnp.where(tw_loc != 0, tw_loc, last_loc))
    return rc | tw, rc, move, loc


# ------------------------------------------------------------------ mustrecharge
if HAS_RECHARGE:
    @handler(ids.HOOK_START, CO.eff("mustrecharge"), ps="data/conditions.ts mustrecharge.onStart")
    def mustrecharge_start(hc, st, dex, rng):
        """'-mustrecharge' pokemon."""
        return hc, EV.emit(st, True, EV.EV_MUSTRECHARGE, actor=hc.target), rng

    @handler(ids.HOOK_BEFOREMOVE, CO.eff("mustrecharge"), ps="data/conditions.ts mustrecharge.onBeforeMove")
    def mustrecharge_before_move(hc, st, dex, rng):
        """priority 11: 'cant' pokemon 'recharge'; removeVolatile('mustrecharge') (and Truant's: not in Reg M-C);
        null."""
        r = hc.target
        st = EV.emit(st, True, EV.EV_CANT, actor=r, move=CO.eff("mustrecharge"))
        st, rng, _ = CO.remove_volatile(st, dex, rng, r, CO.cid("mustrecharge"), True)
        return hc.ret(RS.R(RS.NULL)), st, rng


# ------------------------------------------------------------------ twoturnmove
def move_aborted(st, dex, rng, u, on):
    """runEvent('MoveAborted') after a falsy BeforeMove (sim/battle-actions.ts:314): twoturnmove.onMoveAborted ->
    removeVolatile('twoturnmove') (its onEnd removes the move volatile)."""
    if not HAS_TWOTURN:
        return st, rng
    st, rng, _ = CO.remove_volatile(st, dex, rng, u, CO.cid("twoturnmove"), on)
    return st, rng


def try_move(st, dex, rng, u, target, move, on):
    """The charge moves' onTryMove (singleEvent('TryMove', move), sim/battle-actions.ts:550): the move volatile set
    (second turn) -> removeVolatile, run; else '-prepare' user move, Electro Shot / Meteor Beam boost({spa: 1}); the
    user's effective weather skipping the charge (Rain: Electro Shot; Sun: Solar Beam / Blade) -> attrLastMove(
    '[still]') + '-anim', run; else runEvent('ChargeMove') (Power Herb: not in Reg M-C) -> addVolatile('twoturnmove',
    target) (onStart: effectState.move, attacker.addVolatile(move.id) with targetLoc = lastMoveTargetLoc,
    attrLastMove('[still]'), runEvent('PrepareHit'): no Reg M-C handler); null. Returns (st, rng, null result)."""
    if not (HAS_TWOTURN and CHARGE_MOVES):
        return st, rng, jnp.zeros((), bool)
    from vgcjax.battle import move_pipeline as MP
    from vgcjax.effects import weather as WE
    on = jnp.asarray(on)
    c = CO.cid("twoturnmove")
    is_charge = jnp.zeros((), bool)
    boosts = skip = jnp.zeros((), bool)
    wu = WE.mon_weather(st, u)
    for n, b, ws in CHARGE_MOVES:
        hit = move == mid(n)
        is_charge = is_charge | hit
        if b:
            boosts = boosts | hit
        for w in ws:
            if w in ids.CONDITIONS:
                skip = skip | (hit & (wu == CO.cid(w)))
    on = on & is_charge
    s = CO.vol_state(st, u, c)
    second = on & CO.has_volatile(st, u, c) & (s[TT_PRESENT] != 0)
    st = CO.set_vol_state(st, u, c, s.at[TT_PRESENT].set(0).at[TT_LOC].set(0), second)
    first = on & ~second
    st = EV.emit(st, first, EV.EV_PREPARE, actor=u, move=move)
    st, _ = MP.boost(st, u, jnp.zeros(7, I32).at[2].set(1), first & boosts, source=u, effect=C.eff_move(move))
    st = EV.attr_last_move(st, first & skip, EV.F_STILL)
    st = EV.anim_last_move(st, first & skip)                # addMove('-anim', attacker, move.name, defender)
    charge = first & ~skip
    st, rng, _ = CO.add_volatile(st, dex, rng, u, c, charge, source=target, source_effect=C.eff_move(move))
    added = charge & CO.has_volatile(st, u, c)
    s = CO.vol_state(st, u, c)
    s = s.at[1].set(move.astype(I16)).at[5].set(s[5] | 2)
    s = s.at[TT_LOC].set(mget(st.mons.last_move_target_loc, u).astype(I16)).at[TT_PRESENT].set(1)
    st = CO.set_vol_state(st, u, c, s, added)
    st = CO.next_effect_order(st, added)                    # the move volatile's effect state
    st = EV.attr_last_move(st, added, EV.F_STILL)
    return st, rng, added


def _solar_bp(ctx, h):
    """onBasePower: the user's effectiveWeather Rain / Primordial Sea / Sandstorm / Hail / Snow -> chainModify(0.5)."""
    import jax
    from vgcjax.effects import weather as WE
    wt = jax.vmap(lambda r: WE.mon_weather(ctx.st, r))(jnp.clip(ctx.tgt, 0, 7))
    weak = jnp.zeros(wt.shape, bool)
    for w in ("raindance", "primordialsea", "sandstorm", "hail", "snowscape"):
        if w in ids.CONDITIONS:
            weak = weak | (wt == CO.cid(w))
    return weak & (ctx.tgt >= 0), jnp.asarray(2048, I32), None


for _n in ("solarbeam", "solarblade"):
    if mid(_n) > 0:
        mod("BasePower", int(C.eff_move(mid(_n))), ps=f"data/moves.ts {_n}.onBasePower")(_solar_bp)
