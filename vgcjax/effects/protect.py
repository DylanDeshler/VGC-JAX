"""Protect family + Fake Out (05 §4 step 3): Protect, Detect, stall, Wide Guard, Quick Guard, Fake Out ([C] disable).

PS: data/moves.ts protect / detect / wideguard / quickguard / fakeout (+ data/mods/champions/moves.ts fakeout
onDisableMove), data/conditions.ts stall, sim/battle.ts:1314-1323 checkMoveBypassesProtect. Feint's breaksProtect is
the generic hitStepBreakProtect step (battle/move_pipeline.py).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import queue as Q
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget, mset
from vgcjax.core import rng as R
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects import hooks as H
from vgcjax.effects.hooks import handler

I32 = jnp.int32
CO.fields("stall", ("counter", "int"))
STALLING = ("protect", "detect")


def move_of(eff):
    return jnp.maximum(jnp.asarray(eff, I32) - ids.MOVE_BASE + 1, 0)


def mflag(dex, move, flag):
    w, b = flag
    return ((dex.move_flags[move, w] >> b) & 1) == 1


def will_act(st):
    """PS: sim/battle-queue.ts:315-322 willAct: a queued move / switch / instaswitch."""
    from vgcjax.battle import state as S
    q = st.queue
    return jnp.any(q.valid & ((q.kind == S.Q_MOVE) | (q.kind == S.Q_SWITCH) | (q.kind == S.Q_INSTASWITCH)))


def bypasses_protect(dex, move):
    """checkMoveBypassesProtect (sim/battle.ts:1314-1323): protection applies to moves with the `protect` flag
    (HitProtect handlers: Unseen Fist, v1)."""
    return ~mflag(dex, move, ids.MF_PROTECT)


# ------------------------------------------------------------------ Protect / Detect (moves)
def _stall_prepare_hit(hc, st, dex, rng):
    """onPrepareHit(pokemon): !!queue.willAct() && runEvent('StallMove', pokemon)."""
    wa = will_act(st)
    h2 = H.HookCtx.make(hc.target, -1, hc.effect, relay=RS.R(RS.BOOL, 1))
    h2, st, rng = H.run_event("StallMove", st, dex, rng, h2, branches=H.dispatch(ids.HOOK_STALLMOVE), on=wa)
    ok = wa & RS.truthy((h2.relay_t, h2.relay_v))
    return hc.ret(RS.R(RS.BOOL, ok.astype(I32))), st, rng


def _stall_on_hit(hc, st, dex, rng):
    """onHit(pokemon): pokemon.addVolatile('stall')."""
    st, rng, _ = CO.add_volatile(st, dex, rng, hc.target, CO.cid("stall"), True, source=hc.source,
                                 source_effect=hc.effect)
    return hc, st, rng


for _m in STALLING:
    _e = C.eff_move(getattr(ids, f"MOVE_{_m.upper()}"))
    handler(ids.HOOK_PREPAREHIT, int(_e), ps=f"data/moves.ts {_m}.onPrepareHit")(_stall_prepare_hit)
    handler(ids.HOOK_HIT, int(_e), ps=f"data/moves.ts {_m}.onHit")(_stall_on_hit)


# ------------------------------------------------------------------ protect volatile
@handler(ids.HOOK_START, CO.eff("protect"), ps="data/moves.ts protect.condition.onStart")
def protect_start(hc, st, dex, rng):
    return hc, EV.emit(st, True, EV.EV_SINGLETURN, actor=hc.target, move=CO.eff("protect")), rng


def _block_activate(hc, st, dex, eff_name, cond):
    """'-activate' target 'move: X' and NOT_FAIL when `cond` (smartTarget moves: silent; lockedmove reset: M3)."""
    move = move_of(hc.effect)
    # move.smartTarget is still set only if getSmartTargets found a second (adjacent, live, not-the-user) target
    from vgcjax.battle.move_pipeline import living_ally
    tgt = jnp.clip(hc.target, 0, 7)
    ally = living_ally(st, tgt)
    smart = (dex.move_smart_target[move] == 1) & (ally >= 0) & (ally != hc.source)
    st = EV.emit(st, cond & ~smart, EV.EV_ACTIVATE, actor=hc.target, move=C.eff_move(getattr(ids, f"MOVE_{eff_name}")))
    return hc.ret(RS.R(RS.NOTFAIL), cond), st


@handler(ids.HOOK_TRYHIT, CO.eff("protect"), ps="data/moves.ts protect.condition.onTryHit")
def protect_try_hit(hc, st, dex, rng):
    block = ~bypasses_protect(dex, move_of(hc.effect))
    hc, st = _block_activate(hc, st, dex, "PROTECT", block)
    return hc, st, rng


# ------------------------------------------------------------------ stall volatile
@handler(ids.HOOK_START, CO.eff("stall"), ps="data/conditions.ts stall.onStart")
def stall_start(hc, st, dex, rng):
    return hc, CO.set_vol_field(st, hc.target, CO.cid("stall"), CO.slot_of("stall", "counter"), 3, True), rng


@handler(ids.HOOK_RESTART, CO.eff("stall"), ps="data/conditions.ts stall.onRestart")
def stall_restart(hc, st, dex, rng):
    c, k = CO.cid("stall"), CO.slot_of("stall", "counter")
    cnt = CO.vol_state(st, hc.target, c)[k].astype(I32)
    st = CO.set_vol_field(st, hc.target, c, k, jnp.where(cnt < 729, cnt * 3, cnt), True)
    s = CO.vol_state(st, hc.target, c)
    st = CO.set_vol_state(st, hc.target, c, s.at[0].set(2), True)
    return hc, st, rng


@handler(ids.HOOK_STALLMOVE, CO.eff("stall"), ps="data/conditions.ts stall.onStallMove")
def stall_stall_move(hc, st, dex, rng):
    ref = hc.target
    c = CO.cid("stall")
    cnt = jnp.maximum(CO.vol_state(st, ref, c)[CO.slot_of("stall", "counter")].astype(I32), 1)
    key = C.dkey(st, ids.DK_PROTECT_STALL, actor=ref, subject=hc.effect)
    ok, rng = R.chance(rng, key, 1, cnt)
    # delete pokemon.volatiles['stall'] (no End event)
    st = CO._set_vol_bit(st, ref, c, False, ~ok)
    st = CO.set_vol_state(st, ref, c, jnp.zeros(6, jnp.int16), ~ok)
    return hc.ret(RS.R(RS.BOOL, ok.astype(I32))), st, rng


# ------------------------------------------------------------------ Wide Guard / Quick Guard
def _guard_try(hc, st, dex, rng):
    """onTry(): !!this.queue.willAct()."""
    return hc.ret(RS.R(RS.BOOL, will_act(st).astype(I32))), st, rng


def _guard_hit_side(hc, st, dex, rng):
    """onHitSide(side, source): source.addVolatile('stall')."""
    st, rng, _ = CO.add_volatile(st, dex, rng, hc.source, CO.cid("stall"), True, source=hc.source,
                                 source_effect=hc.effect)
    return hc, st, rng


for _m in ("wideguard", "quickguard"):
    _e = int(C.eff_move(getattr(ids, f"MOVE_{_m.upper()}")))
    handler(ids.HOOK_TRY, _e, ps=f"data/moves.ts {_m}.onTry")(_guard_try)
    handler(ids.HOOK_HITSIDE, _e, ps=f"data/moves.ts {_m}.onHitSide")(_guard_hit_side)


def _guard_side_start(name):
    def fn(hc, st, dex, rng):
        return hc, EV.emit(st, True, EV.EV_SINGLETURN, actor=hc.source, move=CO.eff(name)), rng
    return fn


handler(ids.HOOK_START, CO.eff("wideguard"), ps="data/moves.ts wideguard.condition.onSideStart",
        prefix=ids.PFX_SIDE)(_guard_side_start("wideguard"))
handler(ids.HOOK_START, CO.eff("quickguard"), ps="data/moves.ts quickguard.condition.onSideStart",
        prefix=ids.PFX_SIDE)(_guard_side_start("quickguard"))


@handler(ids.HOOK_TRYHIT, CO.eff("wideguard"), ps="data/moves.ts wideguard.condition.onTryHit")
def wideguard_try_hit(hc, st, dex, rng):
    move = move_of(hc.effect)
    from vgcjax.effects import move_callbacks as MC
    tt = MC.move_target(st, dex, hc.source, move)
    spread = (tt == ids.TARGET_ALLADJACENT) | (tt == ids.TARGET_ALLADJACENTFOES)
    block = spread & ~bypasses_protect(dex, move)
    st = EV.emit(st, block, EV.EV_ACTIVATE, actor=hc.target, move=C.eff_move(ids.MOVE_WIDEGUARD))
    return hc.ret(RS.R(RS.NOTFAIL), block), st, rng


@handler(ids.HOOK_TRYHIT, CO.eff("quickguard"), ps="data/moves.ts quickguard.condition.onTryHit")
def quickguard_try_hit(hc, st, dex, rng):
    move = move_of(hc.effect)
    prio = Q.move_priority(st, dex, hc.source, move)          # move.priority (ModifyPriority: M4 Prankster)
    block = (prio > 0) & ~bypasses_protect(dex, move)
    st = EV.emit(st, block, EV.EV_ACTIVATE, actor=hc.target, move=C.eff_move(ids.MOVE_QUICKGUARD))
    return hc.ret(RS.R(RS.NOTFAIL), block), st, rng


# ------------------------------------------------------------------ Fake Out
@handler(ids.HOOK_TRY, int(C.eff_move(ids.MOVE_FAKEOUT)), ps="data/moves.ts fakeout.onTry")
def fakeout_try(hc, st, dex, rng):
    later = mget(st.mons.active_move_actions, hc.target) > 1       # singleEvent('Try', move, null, pokemon, ...)
    return hc.ret(RS.R(RS.BOOL, 0), later), st, rng


@handler(ids.HOOK_DISABLEMOVE, int(C.eff_move(ids.MOVE_FAKEOUT)), ps="data/mods/champions/moves.ts fakeout.onDisableMove")
def fakeout_disable_move(hc, st, dex, rng):
    ref = hc.target
    m = st.mons
    go = mget(m.active_move_actions, ref) > 0
    slots = mget(m.move_ids, ref) == ids.MOVE_FAKEOUT
    dis = mget(m.move_disabled, ref)
    new = jnp.where(slots & (dis != 1) & go, 1, dis)
    return hc, st.replace(mons=m.replace(move_disabled=mset(m.move_disabled, ref, new.astype(dis.dtype), True))), rng
