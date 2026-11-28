"""Batch 9b (Reg M-C ladder blockers): Heal Bell, Thrash, Future Sight (futuremove slot condition + delayed hit), Salt
Cure, Safeguard, Acupressure, Grav Apple, Burn Up, Morning Sun.

PS: data/moves.ts (line numbers below), data/conditions.ts:379-427 futuremove; Champions overrides in
data/mods/champions/moves.ts: saltcure (:842-850, residual 1/8 Water / Steel else 1/16), gravapple (:459-462, basePower
90: generated data), burnup (:109-112, isNonstandard null). Reused machinery (no second implementation):
- Thrash: the lockedmove volatile of Outrage (effects/batch7c.py; self.volatileStatus is generic move data);
- Morning Sun: Synthesis / Moonlight's onHit (effects/batch7c.py _weather_heal);
- Burn Up: Double Shock's onTryMove / self.onHit (effects/moves_b2.py, a (move, type) family);
- Future Sight: slot conditions (effects/sideconds.py add_slot_condition, Wish's state layout), the move pipeline's
  trySpreadMoveHit steps 2+ (battle/move_pipeline.py move_begin_back / move_hit / move_end);
- Safeguard: generic side-condition add / duration / end (effects/sideconds.py, effects/fieldevent.py);
- Salt Cure: generic secondary volatileStatus; effect damage (effects/statuses.py effect_damage).
Call sites outside this module (minimal hooks, each commented at the site):
- Future Sight: battle/step.py micro (the scheduled hit) / block_residual / run_until_decision (suspension, resume),
  effects/fieldevent.py field_event (stored rows, suspend / resume), battle/move_pipeline.py move_begin_back (the
  hit's ignoreImmunity false) and move_end (no AfterMoveSecondarySelf for a futuremove move, PS :600);
- Safeguard: effects/statuses.py set_status / try_set_status_v / emit_status_msgs (onSetStatus) and
  effects/conditions.py add_volatile (onTryAddVolatile);
- Heal Bell: battle/move_pipeline.py spread_move_hit (an allyTeam move's onHit is singleEvent('Hit'), not HitSide).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget, mset
from vgcjax.core import rng as R
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler

I8, I16, I32, U32 = jnp.int8, jnp.int16, jnp.int32, jnp.uint32


def mid(name):
    return getattr(ids, f"MOVE_{name.upper()}", -999)


def has(name):
    return name in ids.CONDITIONS


def meff(name):
    """EffectId of a move (a Python int)."""
    m = mid(name)
    return ids.MOVE_BASE + m - 1 if m > 0 else -1


def _ab(name):
    return getattr(ids, f"ABILITY_{name.upper()}", -999)


def aeff(name):
    return ids.ABILITY_BASE + _ab(name) - 1


def _has_ab(st, refs, name):
    """pokemon.hasAbility(name): the (Mold Breaker-masked) ability and an active holder (ignoringAbility: inactive)."""
    if _ab(name) <= 0:
        return jnp.zeros(jnp.shape(refs), bool)
    r = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    return (mget(st.mons.ability, r) == _ab(name)) & mget(st.mons.is_active, r)


# ================================================================== Thrash (data/moves.ts:19369-19384)
# self.volatileStatus 'lockedmove' (generic move data) -> effects/batch7c.py lockedmove (onStart trueDuration
# random(2, 4), effectState.move, onLockMove, onResidual / onAfterMove / onEnd fatigue confusion); target randomNormal.

# ================================================================== Morning Sun (data/moves.ts:12280-12315)
if mid("morningsun") > 0:
    from vgcjax.effects import batch7c as _B7C
    handler(ids.HOOK_HIT, meff("morningsun"), ps="data/moves.ts:12289-12309 morningsun.onHit")(_B7C._weather_heal)


# ================================================================== Grav Apple (data/moves.ts:7734-7756)
if mid("gravapple") > 0:
    from vgcjax.effects.modifiers import mod

    @mod("BasePower", meff("gravapple"), ps="data/moves.ts:7743-7747 gravapple.onBasePower")
    def gravapple_bp(ctx, h):
        """field.getPseudoWeather('gravity') -> chainModify(1.5). (Gravity may be absent from the id space.)"""
        if not has("gravity"):
            return jnp.zeros(jnp.shape(ctx.tgt), bool), jnp.asarray(6144, I32), None
        g = CO.has_pseudo(ctx.st, CO.cid("gravity"))
        return g & (ctx.tgt >= 0), jnp.asarray(6144, I32), None


# ================================================================== Heal Bell (data/moves.ts:8243-8275)
if mid("healbell") > 0:
    @handler(ids.HOOK_HIT, meff("healbell"), ps="data/moves.ts:8252-8270 healbell.onHit")
    def healbell_hit(hc, st, dex, rng):
        """'-activate' source 'move: Heal Bell'; for each ally in target.side.pokemon order (sides.ps_order): an ally
        other than the source whose ability is not suppressed (battle.suppressingAbility: the Mold Breaker mask)
        with Soundproof / Good as Gold (hasAbility: active holders only) -> '-immune' ally '[from] ability: X',
        skipped; else cureStatus() ('-curestatus' X status '[msg]'; hp and a status) -> success. Returns success."""
        from vgcjax.effects import statuses as STS
        src = jnp.clip(hc.source, 0, 7)
        side = jnp.clip(hc.target, 0, 7) // 4
        st = EV.emit(st, True, EV.EV_ACTIVATE, actor=src, move=meff("healbell"))
        ok = jnp.zeros((), bool)
        for k in range(4):
            r = side * 4 + st.sides.ps_order[side, k].astype(I32)
            other = r != src
            sp = other & _has_ab(st, r, "soundproof")
            gg = other & ~sp & _has_ab(st, r, "goodasgold")
            st = EV.emit(st, sp, EV.EV_IMMUNE, actor=r, v1=aeff("soundproof"))
            st = EV.emit(st, gg, EV.EV_IMMUNE, actor=r, v1=aeff("goodasgold"))
            cure = ~sp & ~gg & (mget(st.mons.hp, r) > 0) & (mget(st.mons.status, r) > 0)
            st = STS.cure_status(st, r, cure)
            ok = ok | cure
        return hc.ret(RS.R(RS.BOOL, ok.astype(I32))), st, rng


# ================================================================== Acupressure (data/moves.ts:137-167)
if mid("acupressure") > 0:
    @handler(ids.HOOK_HIT, meff("acupressure"), ps="data/moves.ts:146-162 acupressure.onHit")
    def acupressure_hit(hc, st, dex, rng):
        """The target's boosts below 6 in boosts-table order (atk def spa spd spe accuracy evasion); none -> false;
        else sample one (MOVE_ACUPRESSURE_STAT: an index into that list) and boost({stat: 2}) (event target /
        source / effect: the move's; a primary boost)."""
        from vgcjax.battle import move_pipeline as MP
        t = jnp.clip(hc.target, 0, 7)
        u = jnp.clip(hc.source, 0, 7)
        b = mget(st.mons.boosts, t).astype(I32)
        ok = b < 6
        n = jnp.sum(ok.astype(I32))
        key = C.dkey(st, ids.DK_MOVE_ACUPRESSURE_STAT, actor=u, target=t, subject=meff("acupressure"))
        i, rng = R.sample(rng, key, n, active=n > 0)
        stat = jnp.argmax(ok & ((jnp.cumsum(ok.astype(I32)) - 1) == i))
        d = jnp.where(jnp.arange(7) == stat, 2, 0)
        st, _ = MP.boost(st, t, d, n > 0, primary=True, source=u, effect=meff("acupressure"), simple=True)
        return hc.ret(RS.R(RS.BOOL, 0), n == 0), st, rng


# ================================================================== Salt Cure (data/moves.ts:15628-15656)
if mid("saltcure") > 0 and has("saltcure"):
    @handler(ids.HOOK_START, CO.eff("saltcure"), ps="data/moves.ts:15638-15640 saltcure.condition.onStart")
    def saltcure_start(hc, st, dex, rng):
        """'-start' pokemon 'Salt Cure'."""
        return hc, EV.emit(st, True, EV.EV_START, actor=hc.target, move=CO.eff("saltcure"), v1=-1), rng

    @handler(ids.HOOK_RESIDUAL, CO.eff("saltcure"),
             ps="data/mods/champions/moves.ts:842-850 saltcure.condition.onResidual (data/moves.ts:15641-15644)")
    def saltcure_residual(hc, st, dex, rng):
        """onResidualOrder 13 [Champions]: damage(baseMaxhp / (hasType Water or Steel ? 8 : 16)) ('-damage' X hp
        '[from] Salt Cure'; Damage event: Magic Guard; clamped to >= 1)."""
        from vgcjax.effects import statuses as STS
        r = jnp.clip(hc.holder, 0, 7)
        ty = mget(st.mons.types, r)
        ws = jnp.any((ty == ids.TYPE_WATER) | (ty == ids.TYPE_STEEL))
        mx = mget(st.mons.maxhp, r).astype(I32)
        st, rng = STS.effect_damage(st, dex, rng, r, jnp.where(ws, mx // 8, mx // 16), CO.eff("saltcure"), True)
        return hc, st, rng

    @handler(ids.HOOK_END, CO.eff("saltcure"), ps="data/moves.ts:15645-15647 saltcure.condition.onEnd")
    def saltcure_end(hc, st, dex, rng):
        """'-end' pokemon 'Salt Cure'."""
        return hc, EV.emit(st, True, EV.EV_END, actor=hc.target, move=CO.eff("saltcure"), v1=-1), rng


# ================================================================== Safeguard (data/moves.ts:15571-15627)
SG = mid("safeguard") > 0 and has("safeguard")
if SG:
    @handler(CO.DURATION_CALLBACK, CO.eff("safeguard"), ps="data/moves.ts:15583-15589 safeguard.condition.durationCallback")
    def safeguard_duration(hc, st, dex, rng):
        """Persistent (not in Reg M-C) -> 7; else 5."""
        return hc.ret(RS.R(RS.NUM, 5)), st, rng

    @handler(ids.HOOK_START, CO.eff("safeguard"), ps="data/moves.ts:15610-15616 safeguard.condition.onSideStart",
             prefix=ids.PFX_SIDE)
    def safeguard_side_start(hc, st, dex, rng):
        """'-sidestart' side 'Safeguard' (no Persistent in Reg M-C)."""
        return hc, EV.emit(st, True, EV.EV_SIDESTART, actor=hc.holder - 8, move=CO.eff("safeguard")), rng

    @handler(ids.HOOK_END, CO.eff("safeguard"), ps="data/moves.ts:15619-15621 safeguard.condition.onSideEnd",
             prefix=ids.PFX_SIDE)
    def safeguard_side_end(hc, st, dex, rng):
        """onSideResidualOrder 26 / SubOrder 3 (duration: fieldevent.py): '-sideend' side 'Safeguard'."""
        return hc, EV.emit(st, True, EV.EV_SIDEEND, actor=hc.holder - 8, move=CO.eff("safeguard")), rng


def _sg_common(st, refs, src, source_effect):
    """(held, a Move effect, infiltrated) for Safeguard's handlers on refs [...]: the target's side holds it; the
    effect is a Move with move.infiltrates (Infiltrator onModifyMove) from a non-ally source -> no effect."""
    from vgcjax.effects import ability_mods as AM
    refs = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    src = jnp.asarray(src, I32)
    held = CO.has_side(st, refs // 4, CO.cid("safeguard"))
    eff = jnp.asarray(source_effect, I32)
    is_move = (eff >= ids.MOVE_BASE) & (eff < ids.MOVE_BASE + len(ids.MOVES) - 1)
    inf = is_move & AM.infiltrates(st, jnp.clip(src, 0, 7)) & (src >= 0) & ((jnp.clip(src, 0, 7) // 4) != refs // 4)
    return held & ~inf, is_move, jnp.clip(eff - ids.MOVE_BASE + 1, 0, len(ids.MOVES) - 1)


def safeguard_status(st, dex, refs, source, source_effect, move_status):
    """safeguard.condition.onSetStatus (data/moves.ts:15590-15601), a side handler of runEvent('SetStatus') (speed 0,
    subOrder 4: after the abilities of faster holders, before the terrains' subOrder 5): the effect is Yawn -> no
    effect; target !== source (setStatus defaults source to the target) -> null, with '-activate' target 'move:
    Safeguard' when the effect is Synchronize (its sourceEffect {status, id: 'synchronize'}: no effect id, a status;
    effects/batch_reactive.py) or a Move without secondaries. Returns (blocked, message) per ref."""
    refs = jnp.asarray(refs, I32)
    if not SG:
        z = jnp.zeros(jnp.shape(refs), bool)
        return z, z
    src = jnp.where(jnp.asarray(source, I32) >= 0, jnp.asarray(source, I32), refs)
    held, is_move, mv = _sg_common(st, refs, src, source_effect)
    eff = jnp.asarray(source_effect, I32)
    yawn = eff == CO.eff("yawn") if has("yawn") else jnp.zeros((), bool)
    sync = (eff < 0) & (jnp.asarray(move_status, I32) > 0)
    blocked = held & ~yawn & (src != refs)
    return blocked, blocked & (sync | (is_move & (dex.move_nsec[mv] == 0)))


def safeguard_volatile(st, dex, ref, c, src, source_effect, only=None):
    """safeguard.condition.onTryAddVolatile (data/moves.ts:15602-15609): confusion / yawn from another source ->
    null, with '-activate' target 'move: Safeguard' for a Move effect without secondaries. Returns (blocked, msg).
    `only`: the call site's static candidate EffectIds (None: any); without confusion / yawn nothing is traced."""
    rel = {CO.eff(n) for n in ("confusion", "yawn") if has(n)}
    if not SG or (only is not None and not (set(only) & rel)):
        z = jnp.zeros((), bool)
        return z, z
    held, is_move, mv = _sg_common(st, ref, src, source_effect)
    c = jnp.asarray(c, I32)
    which = (c == CO.cid("confusion")) if has("confusion") else jnp.zeros((), bool)
    if has("yawn"):
        which = which | (c == CO.cid("yawn"))
    blocked = held & which & (jnp.asarray(src, I32) != jnp.asarray(ref, I32))
    return blocked, blocked & is_move & (dex.move_nsec[mv] == 0)


# ================================================================== Future Sight (data/moves.ts:6394-6428,
# data/conditions.ts:379-427 futuremove)
FUT = mid("futuresight") > 0 and has("futuremove")
if FUT:
    CO.fields("futuremove", ("targetSlot", "slot"), ("endingTurn", "int"), ("move", "move"), ("moveData", "move"))
    _FK = CO.cid("futuremove") - ids.SLOT_BASE
    _F_SLOT, _F_END, _F_MOVE, _F_DATA = (CO.slot_of("futuremove", k) for k in ("targetSlot", "endingTurn", "move",
                                                                              "moveData"))
    _F_PRES = (1 << _F_SLOT) | (1 << _F_END) | (1 << _F_MOVE) | (1 << _F_DATA)
    # effectState.source (dropped by the projection): (ref + 1) in the presence slot's high byte

    @handler(ids.HOOK_TRY, meff("futuresight"),
             ps="data/moves.ts:6404-6424 futuresight.onTry; data/conditions.ts:382-394 futuremove.onStart / onRestart")
    def futuresight_try(hc, st, dex, rng):
        """onTry(source, target): target.side.addSlotCondition(target, 'futuremove') (present: onRestart false ->
        return false); onStart: effectState.targetSlot = target.getSlot(), endingTurn = (turn - 1) + 2; then
        effectState.move / source / moveData (projected as 'futuresight'); '-start' source 'move: Future Sight';
        return NOT_FAIL."""
        from vgcjax.battle import move_pipeline as MP
        from vgcjax.effects import sideconds as SC
        u = jnp.clip(hc.target, 0, 7)
        t = jnp.clip(hc.source, 0, 7)
        s, p = t // 4, jnp.maximum(MP.position(st, t), 0)
        st, rng, res = SC.add_slot_condition(st, dex, rng, s, p, CO.cid("futuremove"), hc.source >= 0, source=u,
                                             source_effect=meff("futuresight"))
        ok = (hc.source >= 0) & RS.truthy(res)
        sd = st.sides
        v = sd.slot_cond_state[s, p, _FK]
        v = v.at[_F_SLOT].set((s * 2 + p).astype(I16)).at[_F_END].set((st.meta.turn.astype(I32) + 1).astype(I16))
        v = v.at[_F_MOVE].set(mid("futuresight")).at[_F_DATA].set(mid("futuresight"))
        v = v.at[5].set((_F_PRES | ((u + 1) << 8)).astype(I16))
        sd = sd.replace(slot_cond_state=sd.slot_cond_state.at[s, p, _FK].set(jnp.where(ok, v,
                                                                                        sd.slot_cond_state[s, p, _FK])))
        st = EV.emit(st.replace(sides=sd), ok, EV.EV_START, actor=u, move=meff("futuresight"), v1=-1)
        return hc.ret(RS.where(ok, RS.R(RS.NOTFAIL), RS.R(RS.BOOL, 0))), st, rng

    @handler(ids.HOOK_RESIDUAL, CO.eff("futuremove"), ps="data/conditions.ts:395-426 futuremove.onResidual / onEnd")
    def futuremove_residual(hc, st, dex, rng):
        """onResidualOrder 3: getOverflowedTurnCount() = (turn - 1) mod 256 < endingTurn -> nothing; else
        removeSlotCondition(getAtSlot(targetSlot)) (this slot): onEnd(target) -> target fainted or the source ->
        no hit; else '-end' target 'move: Future Sight' and trySpreadMoveHit([target], source, hitMove, true): the hit
        is scheduled here (Scratch.fut), the walk suspends after this row (effects/fieldevent.py) and the next
        micro-step runs it (battle/step.py micro, future_start). The delete happens now (PS: after the hit; no
        decision point in between)."""
        from vgcjax.battle import move_pipeline as MP
        r = jnp.clip(hc.holder, 0, 7)
        s, p = r // 4, jnp.maximum(MP.position(st, r), 0)
        v = st.sides.slot_cond_state[s, p, _FK].astype(I32)
        go = ((st.meta.turn.astype(I32) - 1) & 255) >= v[_F_END]
        src = ((v[5] >> 8) & 0xFF) - 1
        hit = go & ~mget(st.mons.fainted, r) & (r != src) & (src >= 0)
        st = EV.emit(st, hit, EV.EV_END, actor=r, move=meff("futuresight"), v1=-1)
        st = fset(st, {F_PENDING: hit | (fget(st, F_PENDING) != 0), F_U: jnp.where(hit, src, fget(st, F_U)),
                       F_T: jnp.where(hit, r, fget(st, F_T)),
                       F_MOVE: jnp.where(hit, jnp.maximum(v[_F_DATA], 1), fget(st, F_MOVE))})
        sd = st.sides
        bits = sd.slot_cond_bits[s, p]
        sd = sd.replace(slot_cond_bits=sd.slot_cond_bits.at[s, p].set(jnp.where(go, bits & ~(U32(1) << _FK), bits)),
                        slot_cond_state=sd.slot_cond_state.at[s, p, _FK].set(
                            jnp.where(go, jnp.zeros(6, I16), sd.slot_cond_state[s, p, _FK])))
        return hc, st.replace(sides=sd), rng


# Scratch.fut slots
F_PENDING, F_RUNNING, F_U, F_T, F_MOVE, F_MASK, F_AB, F_ITEM, F_RS, F_RSI = range(10)


def fget(st, k):
    return st.scratch.fut[k].astype(I32)


def fset(st, upd):
    """Scratch.fut[k] = v for {k: v} (values already conditioned)."""
    f = st.scratch.fut
    for k, v in upd.items():
        f = f.at[k].set(jnp.asarray(v).astype(I16))
    return st.replace(scratch=st.scratch.replace(fut=f))


def future_start(st, dex, on):
    """The scheduled future hit (data/conditions.ts:404-419 futuremove.onEnd -> trySpreadMoveHit([target], source,
    new Move(moveData), notActive) (sim/battle-actions.ts:612-682): no runMove / useMove (no MOVE line, PP, lastMove,
    ModifyType / ModifyMove: no Mold Breaker), no Try / PrepareHit (moveData has no callbacks); steps 0-1 here:
    0 Invulnerability ('-miss' source target, attrLastMove('[miss]')); 1 TryHit: moveData has no protect flag
    (protections skip it); of the abilities' onTryHit only Telepathy can apply to a Psychic special hit on an ally
    ('-activate' target 'ability: Telepathy', null). Steps 2+ and the hit run in move_begin_back / move_hit /
    move_end (fut_running: ignoreImmunity false). An inactive source's ability and item are parked (ignoringAbility /
    ignoringItem: !isActive, sim/pokemon.ts:880, :902) until future_resume. Returns (st, carry)."""
    from vgcjax.battle import state as S
    from vgcjax.effects import locked as LK
    on = jnp.asarray(on)
    if not FUT:
        return st, S.empty_carry()
    u, t, move = fget(st, F_U), fget(st, F_T), jnp.maximum(fget(st, F_MOVE), 1)
    uc, tc = jnp.clip(u, 0, 7), jnp.clip(t, 0, 7)
    targets = jnp.stack([t, jnp.int32(-1), jnp.int32(-1)])
    inv = on & LK.semi_invulnerable(st, dex, u, targets, move)[0]
    st = EV.attr_last_move(st, inv, EV.F_MISS)
    st = EV.emit(st, inv, EV.EV_MISS, actor=uc, target=tc)
    tele = on & ~inv & (t != u) & (tc // 4 == uc // 4) & (mget(st.mons.ability, tc) == _ab("telepathy")) & \
        (dex.move_category[move] != ids.CAT_STATUS)
    st = EV.emit(st, tele, EV.EV_ACTIVATE, actor=tc, move=aeff("telepathy"))
    m = st.mons
    mask = on & ~mget(m.is_active, uc)
    ab, it = mget(m.ability, uc).astype(I16), mget(m.item, uc).astype(I16)
    m = m.replace(ability=mset(m.ability, uc, 0, mask), item=mset(m.item, uc, 0, mask))
    st = fset(st.replace(mons=m), {F_PENDING: (fget(st, F_PENDING) != 0) & ~on, F_RUNNING: (fget(st, F_RUNNING) != 0) | on,
                                   F_MASK: jnp.where(on, mask, fget(st, F_MASK) != 0),
                                   F_AB: jnp.where(mask, ab, fget(st, F_AB)),
                                   F_ITEM: jnp.where(mask, it, fget(st, F_ITEM))})
    tv = jnp.stack([~inv & ~tele, jnp.zeros((), bool), jnp.zeros((), bool)])
    f = jnp.zeros((), bool)
    carry = S.MoveCarry(u=u, move=move, targets=targets, tv=tv, smart=f, spread=f, spread_hit=f, fail0=inv,
                        a_type=dex.move_type[move].astype(I32), a_bp=dex.move_bp[move].astype(I32),
                        a_acc=dex.move_accuracy[move].astype(I32), on=on, go_rm=f, go_um=f, has_t=on, side_field=f,
                        on_ts=on, side_on=f, pre_ok=on, pre_notfail=f, tm_null=f, abort=f)
    return st, carry


def future_exec(st, on):
    """The future hit's Exec has no runMove tail (no resolveFaints / checkWin there: future_resume's checkWin, then
    the walk's resume row's resolveFaints, PS order)."""
    if not FUT:
        return st
    ex = st.exec
    return st.replace(exec=ex.replace(on_rm=ex.on_rm & ~jnp.asarray(on)))


def future_done(st, move_done):
    """The future hit in st.exec finished (no runAction tail: the residual walk resumes). Returns (st, fut_done)."""
    if not FUT:
        return st, jnp.zeros((), bool)
    run = fget(st, F_RUNNING) != 0
    d = jnp.asarray(move_done) & run
    return fset(st, {F_RUNNING: run & ~d}), d


def future_ignore_immunity(st, ign):
    """moveData.ignoreImmunity false (data/moves.ts:6419): the hit's type immunity applies (Dark targets)."""
    if not FUT:
        return ign
    return jnp.where(fget(st, F_RUNNING) != 0, 0, ign)


def future_resume(st, dex, rng, on):
    """Back in futuremove.onEnd after trySpreadMoveHit (data/conditions.ts:420-425): the parked ability / item return
    (Life Orb's AfterMoveSecondarySelf ran at the hit's end: battle/move_pipeline.py move_end, future_lifeorb);
    checkWin(). fieldEvent's resolveFaints (sim/battle.ts:572) follows in the walk (its resume row)."""
    from vgcjax.battle import faint as F
    if not FUT:
        return st, rng
    on = jnp.asarray(on)
    u = jnp.clip(fget(st, F_U), 0, 7)
    m = st.mons
    rest = on & (fget(st, F_MASK) != 0)
    m = m.replace(ability=mset(m.ability, u, fget(st, F_AB), rest), item=mset(m.item, u, fget(st, F_ITEM), rest))
    st = fset(st.replace(mons=m), {F_MASK: (fget(st, F_MASK) != 0) & ~on})
    return F.check_win(st, -1, on), rng


def future_lifeorb(st, end):
    """futuremove.onEnd (data/conditions.ts:420-422): data.source.isActive && hasItem('lifeorb') ->
    singleEvent('AfterMoveSecondarySelf', Life Orb, ..., source, target, Life Orb) right after trySpreadMoveHit,
    whatever its result (the `move` argument is the item: not a Status move). An active source is never parked."""
    if not FUT:
        return jnp.zeros((), bool)
    u = jnp.clip(fget(st, F_U), 0, 7)
    return jnp.asarray(end) & (fget(st, F_RUNNING) != 0) & mget(st.mons.is_active, u)


def suspended(st):
    """A future hit is scheduled or the residual walk waits for it (battle/step.py: no end of turn meanwhile)."""
    if not FUT:
        return jnp.zeros((), bool)
    return (fget(st, F_PENDING) != 0) | (fget(st, F_RS) != 0)


def pending(st):
    return fget(st, F_PENDING) != 0 if FUT else jnp.zeros((), bool)


def rs_pending(st):
    return fget(st, F_RS) != 0 if FUT else jnp.zeros((), bool)
