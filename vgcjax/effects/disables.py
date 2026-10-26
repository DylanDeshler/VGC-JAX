"""Move-restricting volatiles: Taunt, choicelock (+ Choice Scarf); doubles support moves Follow Me / Rage Powder
(redirection), Helping Hand; Parting Shot (data/moves.ts, data/conditions.ts choicelock, data/items.ts choicescarf;
no Champions overrides). DisableMove effects run undispatched at end of turn (step.end_turn -> disable_moves).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32
CO.fields("choicelock", ("move", "move"))
CO.fields("helpinghand", ("multiplier", "x1.5"))


def will_move(st, ref):
    """PS: sim/battle-queue.ts willMove(pokemon): a queued 'move' action of pokemon."""
    from vgcjax.battle import queue as Q
    q = st.queue
    return jnp.any(q.valid & (q.kind == S.Q_MOVE) & (Q.entry_ref(q) == ref))


def _move_cat(dex, mids):
    return dex.move_category[jnp.clip(mids, 0, len(ids.MOVES) - 1)]


# ------------------------------------------------------------------ Taunt
if "taunt" in ids.CONDITIONS:
    @handler(ids.HOOK_START, CO.eff("taunt"), ps="data/moves.ts taunt.condition.onStart")
    def taunt_start(hc, st, dex, rng):
        """activeTurns and not willMove -> duration++; '-start' X 'move: Taunt'."""
        r = hc.target
        c = CO.cid("taunt")
        more = (mget(st.mons.active_turns, r) > 0) & ~will_move(st, r)
        s = CO.vol_state(st, r, c)
        st = CO.set_vol_state(st, r, c, s.at[0].set(s[0] + more.astype(I16)), True)
        return hc, EV.emit(st, True, EV.EV_START, actor=r, move=CO.eff("taunt"), v1=-1), rng

    @handler(ids.HOOK_END, CO.eff("taunt"), ps="data/moves.ts taunt.condition.onEnd")
    def taunt_end(hc, st, dex, rng):
        return hc, EV.emit(st, True, EV.EV_END, actor=hc.target, move=CO.eff("taunt"), v1=-1), rng

    @handler(ids.HOOK_BEFOREMOVE, CO.eff("taunt"), ps="data/moves.ts taunt.condition.onBeforeMove")
    def taunt_before_move(hc, st, dex, rng):
        """priority 5: a Status move (not Me First) -> 'cant' X 'move: Taunt' move; false."""
        move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
        block = dex.move_category[move] == ids.CAT_STATUS
        st = EV.emit(st, block, EV.EV_CANT, actor=hc.target, move=CO.eff("taunt"))
        return hc.ret(RS.R(RS.BOOL, 0), block), st, rng


# ------------------------------------------------------------------ choicelock / Choice Scarf
CHOICE_ITEMS = tuple(getattr(ids, f"ITEM_{n.upper()}") for n in ("choicescarf", "choiceband", "choicespecs")
                     if hasattr(ids, f"ITEM_{n.upper()}"))


def is_choice(item):
    out = jnp.zeros(jnp.shape(item), bool)
    for i in CHOICE_ITEMS:
        out = out | (item == i)
    return out


def choice_lock(st, dex, rng, u, move, on):
    """The choice item's onModifyMove: pokemon.addVolatile('choicelock') -> onStart: effectState.move = the active
    move (an existing lock has no onRestart: unchanged). Returns (st, rng)."""
    if "choicelock" not in ids.CONDITIONS:
        return st, rng
    go = jnp.asarray(on) & is_choice(mget(st.mons.item, u))
    c = CO.cid("choicelock")
    st, rng, _ = CO.add_volatile(st, dex, rng, u, c, go & ~CO.has_volatile(st, u, c), source=u)
    fresh = go & CO.has_volatile(st, u, c) & (CO.vol_state(st, u, c)[CO.slot_of("choicelock", "move")] == 0)
    return CO.set_vol_field(st, u, c, CO.slot_of("choicelock", "move"), move, fresh), rng


if "choicelock" in ids.CONDITIONS:
    @handler(ids.HOOK_BEFOREMOVE, CO.eff("choicelock"), ps="data/conditions.ts choicelock.onBeforeMove")
    def choicelock_before_move(hc, st, dex, rng):
        """No choice item -> removeVolatile; a different move (not Struggle) -> 'move' X move '[still]', '-fail' X;
        false (no PP lost)."""
        r = hc.target
        c = CO.cid("choicelock")
        move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
        held = is_choice(mget(st.mons.item, r))
        st, rng, _ = CO.remove_volatile(st, dex, rng, r, c, ~held)
        locked = CO.vol_state(st, r, c)[CO.slot_of("choicelock", "move")].astype(I32)
        block = held & (move != locked) & (move != ids.MOVE_STRUGGLE)
        st = EV.emit_move(st, block, r, -1, move)
        st = EV.attr_last_move(st, block, EV.F_STILL)
        st = EV.emit(st, block, EV.EV_FAIL, actor=r)
        return hc.ret(RS.R(RS.BOOL, 0), block), st, rng

# Choice Scarf onModifySpe x1.5: battle/common.py spe_chain


# ------------------------------------------------------------------ DisableMove (end of turn)
def disable_moves(st, dex, rng, r, on):
    """runEvent('DisableMove', pokemon) for the volatiles (sim/battle.ts:1708): Taunt (Status moves but Me First),
    choicelock (no choice item or no locked move -> removeVolatile, else every other slot). Returns (st, rng)."""
    m = st.mons
    mids = mget(m.move_ids, r).astype(I32)
    dis = mget(m.move_disabled, r)
    on = jnp.asarray(on)
    if "taunt" in ids.CONDITIONS:
        t = on & CO.has_volatile(st, r, CO.cid("taunt"))
        hit = t & (_move_cat(dex, mids) == ids.CAT_STATUS) & (mids > 0) & (mids != getattr(ids, "MOVE_MEFIRST", -1))
        dis = jnp.where(hit, 1, dis)
    if "choicelock" in ids.CONDITIONS:
        c = CO.cid("choicelock")
        cl = on & CO.has_volatile(st, r, c)
        locked = CO.vol_state(st, r, c)[CO.slot_of("choicelock", "move")].astype(I32)
        keep = is_choice(mget(m.item, r)) & jnp.any(mids == locked)
        st, rng, _ = CO.remove_volatile(st, dex, rng, r, c, cl & ~keep)
        dis = jnp.where(cl & keep & (mids != locked) & (mids > 0), 1, dis)
    m = st.mons
    return st.replace(mons=m.replace(move_disabled=mset(m.move_disabled, r, dis.astype(m.move_disabled.dtype), on))), rng


# ------------------------------------------------------------------ Follow Me / Rage Powder (RedirectTarget)
for _n, _mv in (("followme", "Follow Me"), ("ragepowder", "Rage Powder")):
    if _n in ids.CONDITIONS:
        def _start(hc, st, dex, rng, _n=_n):
            """'-singleturn' X 'move: <name>'."""
            return hc, EV.emit(st, True, EV.EV_SINGLETURN, actor=hc.target, move=CO.eff(_n)), rng
        handler(ids.HOOK_START, CO.eff(_n), ps=f"data/moves.ts {_n}.condition.onStart")(_start)
        # onTry: activePerHalf > 1 (always true in doubles)


def redirect_target(st, dex, u, move, target, on):
    """priorityEvent('RedirectTarget', user, user, move, target) (sim/pokemon.ts getMoveTargets, not tracksTarget):
    a foe's Follow Me / Rage Powder (onFoeRedirectTargetPriority 1; Rage Powder needs the user not powder-immune)
    whose holder is a valid target for the move redirects it (and clears smartTarget). Two holders: the faster
    (effectOrder ties: the earlier). Returns (new target, redirected)."""
    from vgcjax.core import typechart as TC
    m = st.mons
    fs = 1 - jnp.clip(u, 0, 7) // 4
    act = st.sides.active.astype(I32)[fs]
    refs = fs * 4 + jnp.maximum(act, 0)
    alive = (act >= 0) & ~mget(m.fainted, refs) & (mget(m.hp, refs) > 0)
    from vgcjax.effects import move_callbacks as MC
    tt = MC.move_target(st, dex, u, move)
    valid_t = (tt == ids.TARGET_NORMAL) | (tt == ids.TARGET_ANY) | (tt == ids.TARGET_ADJACENTFOE) | \
        (tt == ids.TARGET_RANDOMNORMAL)
    powder_ok = ~TC.status_immune(dex, ids.IMMUNITY_KEYS.index("powder"), mget(m.types, jnp.clip(u, 0, 7)))
    best, best_spd = jnp.asarray(-1, I32), jnp.asarray(-(1 << 20), I32)
    for j in range(2):
        r = refs[j]
        fm = CO.has_volatile(st, r, CO.cid("followme")) if "followme" in ids.CONDITIONS else False
        rp = (CO.has_volatile(st, r, CO.cid("ragepowder")) & powder_ok) if "ragepowder" in ids.CONDITIONS else False
        ok = on & alive[j] & valid_t & (fm | rp)
        spd = mget(m.speed, r).astype(I32)
        take = ok & (spd > best_spd)
        best = jnp.where(take, r, best)
        best_spd = jnp.where(take, spd, best_spd)
    red = best >= 0
    return jnp.where(red, best, target), red


# ------------------------------------------------------------------ Helping Hand
if "helpinghand" in ids.CONDITIONS:
    _hh = int(C.eff_move(ids.MOVE_HELPINGHAND))

    @handler(ids.HOOK_TRYHIT, _hh, ps="data/moves.ts helpinghand.onTryHit")
    def helpinghand_try_hit(hc, st, dex, rng):
        """!target.newlySwitched && !queue.willMove(target) -> false."""
        r = hc.target
        fail = ~mget(st.mons.newly_switched, r) & ~will_move(st, r)
        return hc.ret(RS.R(RS.BOOL, 0), fail), st, rng

    @handler(ids.HOOK_START, CO.eff("helpinghand"), ps="data/moves.ts helpinghand.condition.onStart")
    def helpinghand_start(hc, st, dex, rng):
        """multiplier = 1.5; '-singleturn' target 'Helping Hand' '[of] source'."""
        r = hc.target
        st = CO.set_vol_field(st, r, CO.cid("helpinghand"), CO.slot_of("helpinghand", "multiplier"), 1, True)
        return hc, EV.emit(st, True, EV.EV_SINGLETURN, actor=r, move=CO.eff("helpinghand")), rng

    @handler(ids.HOOK_RESTART, CO.eff("helpinghand"), ps="data/moves.ts helpinghand.condition.onRestart")
    def helpinghand_restart(hc, st, dex, rng):
        """multiplier *= 1.5; '-singleturn' target 'Helping Hand' '[of] source'."""
        r = hc.target
        c, k = CO.cid("helpinghand"), CO.slot_of("helpinghand", "multiplier")
        n = CO.vol_state(st, r, c)[k]
        st = CO.set_vol_field(st, r, c, k, n + 1, True)
        st = EV.emit(st, True, EV.EV_SINGLETURN, actor=r, move=CO.eff("helpinghand"))
        return hc.ret(RS.R(RS.BOOL, 1)), st, rng

    @mod("BasePower", CO.eff("helpinghand"), ps="data/moves.ts helpinghand.condition.onBasePower")
    def helpinghand_bp(ctx, h):
        """priority 10: chainModify(multiplier) (1.5^n as a 4096-scale factor: chainModify(1.5) = 6144, 2.25 = 9216)."""
        vs = mget(ctx.st.mons.vol_state, jnp.clip(h, 0, 7))              # [T, NV, 6]
        n = vs[..., CO.cid("helpinghand") - ids.VOL_BASE, CO.slot_of("helpinghand", "multiplier")]
        return jnp.ones(h.shape, bool), jnp.where(n >= 2, 9216, 6144).astype(I32), None
