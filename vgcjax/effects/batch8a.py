"""Batch 8a (Reg M-C ladder blockers): Attract (the move), Rest, Reversal, Fairy Lock, Magic Powder, Minimize (+ the
minimize flag of Body Slam / Dragon Rush / Heat Crash / Heavy Slam / Supercell Slam), Stockpile, Curse (Champions),
Instruct.

PS: data/moves.ts (line numbers below); Champions overrides: data/mods/champions/moves.ts:165-196 curse
(volatileStatus undefined; non-Ghost: onHit boost; Ghost: directDamage + addVolatile), sim/battle-queue.ts:263-266
(resolveAction: a non-Ghost user's Curse targets 'self'), sim/pokemon.ts:1020-1024 (request target 'self'). Call sites
outside this module (minimal hooks, each commented at the site):
- Attract: the 'attract' volatile is effects/batch_passive.py's (Cute Charm); onTryImmunity: battle/move_pipeline.py
  move_begin_back step 3 (try_immunity);
- Reversal: effects/move_callbacks.py base_power (basePowerCallback);
- Fairy Lock: battle/step.py end_turn (onTrapPokemon);
- Minimize: battle/move_pipeline.py accuracy_step (onAccuracy); onSourceModifyDamage is a ModifyDamage mod below;
- Curse: battle/move_pipeline.py move_begin_front (onModifyMove target), effects/move_callbacks.py move_target
  (ActiveMove target 'self'), battle/requests.py move_slot_mask and battle/step.py decode_move_slot / resolve_loc
  (request / resolveAction target 'self');
- Stockpile: '-start' X 'stockpileN' printed by harness/events.py from START v2 = layers;
- Magic Powder: '-start' X 'typechange' printed by harness/events.py (START v2 = 1, as Soak).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset
from vgcjax.core import typechart as TC
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler
from vgcjax.effects.modifiers import mod

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32


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


def _mflag(dex, move, flag):
    w, b = flag
    return ((dex.move_flags[jnp.maximum(jnp.asarray(move, I32), 0), w] >> b) & 1) == 1


def has_type(st, r, ty):
    """pokemon.hasType(type) (sim/pokemon.ts hasType: getTypes() incl. addedType)."""
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    m = st.mons
    return jnp.any(mget(m.types, r) == ty, axis=-1) | (mget(m.added_type, r) == ty)


# ================================================================== Reversal (data/moves.ts:15074-15105)
def reversal_bp(st, u, move, out):
    """basePowerCallback(pokemon) (data/moves.ts:15078-15096): ratio = max(floor(hp * 48 / maxhp), 1); < 2 -> 200,
    < 5 -> 150, < 10 -> 100, < 17 -> 80, < 33 -> 40, else 20."""
    if mid("reversal") <= 0:
        return out
    uc = jnp.clip(jnp.asarray(u, I32), 0, 7)
    hp, mx = mget(st.mons.hp, uc).astype(I32), jnp.maximum(mget(st.mons.maxhp, uc).astype(I32), 1)
    ratio = jnp.maximum((hp * 48) // mx, 1)
    bp = jnp.where(ratio < 2, 200, jnp.where(ratio < 5, 150, jnp.where(ratio < 10, 100, jnp.where(
        ratio < 17, 80, jnp.where(ratio < 33, 40, 20)))))
    return jnp.where(jnp.asarray(move) == mid("reversal"), bp, out)


# ================================================================== Rest (data/moves.ts:14954-14991)
if mid("rest") > 0:
    @handler(ids.HOOK_TRY, meff("rest"), ps="data/moves.ts:14963-14979 rest.onTry")
    def rest_try(hc, st, dex, rng):
        """onTry(source): asleep or Comatose -> false; full hp -> '-fail' source 'heal', null; Insomnia / Vital Spirit
        -> '-fail' source '[from] ability: X' '[of] source' (a plain FAIL record), null."""
        u = jnp.clip(hc.target, 0, 7)
        m = st.mons
        ab = mget(m.ability, u)
        f_false = (mget(m.status, u) == ids.COND_SLP) | (ab == _ab("comatose"))
        f_heal = ~f_false & (mget(m.hp, u) >= mget(m.maxhp, u))
        f_ab = ~f_false & ~f_heal & ((ab == _ab("insomnia")) | (ab == _ab("vitalspirit")))
        st = EV.emit(st, f_heal, EV.EV_FAIL, actor=u, flags=EV.F_HEAL)
        st = EV.emit(st, f_ab, EV.EV_FAIL, actor=u)
        hc = hc.ret(RS.R(RS.BOOL, 0), f_false)
        return hc.ret(RS.R(RS.NULL), f_heal | f_ab), st, rng

    @handler(ids.HOOK_HIT, meff("rest"), ps="data/moves.ts:14980-14986 rest.onHit")
    def rest_hit(hc, st, dex, rng):
        """onHit(target, source, move): setStatus('slp', source, move) ('-status' X 'slp' '[from] move: Rest'; the
        slp onStart's sample([2, 3, 3]) still draws); a falsy result is returned; statusState.time = startTime = 3;
        heal(maxhp) (TryHeal: Heal Block; '-heal' X hp '[silent]': a plain HEAL record)."""
        from vgcjax.battle import move_pipeline as MP
        from vgcjax.effects import statuses as STS
        t = jnp.clip(hc.target, 0, 7)
        st, rng, res = STS.set_status(st, dex, rng, t, ids.COND_SLP, True, source=hc.source,
                                      source_effect=meff("rest"), move_status=0)
        ok = RS.truthy(res)
        st = STS.set_status_field(st, t, CO.slot_of("slp", "time"), 3, ok)
        st = STS.set_status_field(st, t, CO.slot_of("slp", "startTime"), 3, ok)
        st, _ = MP.battle_heal(st, t, mget(st.mons.maxhp, t).astype(I32), ok)
        return hc.ret(res, ~ok), st, rng


# ================================================================== Magic Powder (data/moves.ts:10740-10755)
if mid("magicpowder") > 0:
    @handler(ids.HOOK_HIT, meff("magicpowder"), ps="data/moves.ts:10749-10752 magicpowder.onHit")
    def magicpowder_hit(hc, st, dex, rng):
        """getTypes().join() === 'Psychic' -> false (setType: Arceus / Silvally / Terastallized are not in Reg M-C);
        else setType('Psychic') (addedType cleared), '-start' target 'typechange' 'Psychic' (START move = Magic
        Powder, v2 = 1: harness/events.py prints 'typechange'). Roost's filtered types are the stored ones
        (effects/moves_b2.py roost_start)."""
        t = jnp.clip(hc.target, 0, 7)
        m = st.mons
        ty = mget(m.types, t).astype(I32)
        psy = (ty[0] == ids.TYPE_PSYCHIC) & (ty[1] <= 0) & (mget(m.added_type, t) <= 0)
        new = jnp.stack([jnp.int32(ids.TYPE_PSYCHIC), jnp.int32(-1)]).astype(m.types.dtype)
        m = m.replace(types=mset(m.types, t, new, ~psy), added_type=mset(m.added_type, t, 0, ~psy))
        st = EV.emit(st.replace(mons=m), ~psy, EV.EV_START, actor=t, move=meff("magicpowder"), v1=-1, v2=1)
        return hc.ret(RS.R(RS.BOOL, 0), psy), st, rng


# ================================================================== Minimize (data/moves.ts:11919-11946)
MINI = mid("minimize") > 0 and has("minimize")
if MINI:
    @handler(ids.HOOK_RESTART, CO.eff("minimize"), ps="data/moves.ts:11931 minimize.condition.onRestart")
    def minimize_restart(hc, st, dex, rng):
        """onRestart: () => null."""
        return hc.ret(RS.R(RS.NULL)), st, rng

    @mod("ModifyDamage", CO.eff("minimize"), prefix=ids.PFX_SOURCE,
         ps="data/moves.ts:11932-11936 minimize.condition.onSourceModifyDamage")
    def minimize_damage(ctx, h):
        """The holder (the move's target) takes chainModify(2) from a move with flags.minimize."""
        return jnp.broadcast_to(_mflag(ctx.dex, ctx.move, ids.MF_MINIMIZE), h.shape), jnp.asarray(8192, I32), None


def minimize_accuracy(st, dex, move, targets):
    """minimize.condition.onAccuracy (data/moves.ts:11937-11942; the holder is the target, runEvent('Accuracy')):
    flags.minimize -> true. Returns bool [T]."""
    if not MINI:
        return jnp.zeros(jnp.shape(targets), bool)
    t = jnp.asarray(targets, I32)
    return CO.has_volatile(st, jnp.clip(t, 0, 7), CO.cid("minimize")) & (t >= 0) & _mflag(dex, move, ids.MF_MINIMIZE)


# ================================================================== Stockpile (data/moves.ts:17956-18004)
STOCK = mid("stockpile") > 0 and has("stockpile")
if STOCK:
    CO.fields("stockpile", ("layers", "int"), ("def", "int"), ("spd", "int"))
    _SP = CO.cid("stockpile")
    _SP_L, _SP_D, _SP_S = (CO.slot_of("stockpile", k) for k in ("layers", "def", "spd"))

    @handler(ids.HOOK_TRY, meff("stockpile"), ps="data/moves.ts:17965-17967 stockpile.onTry")
    def stockpile_try(hc, st, dex, rng):
        """volatiles.stockpile.layers >= 3 -> false."""
        u = jnp.clip(hc.target, 0, 7)
        full = CO.has_volatile(st, u, _SP) & (CO.vol_state(st, u, _SP)[_SP_L] >= 3)
        return hc.ret(RS.R(RS.BOOL, 0), full), st, rng

    def _stockpile_layer(hc, st, dex, rng, first):
        """onStart / onRestart body: layers = 1 (def = spd = 0) or layers++ (layers >= 3 -> false on Restart);
        '-start' target 'stockpileN' (START v2 = N); boost({def: 1, spd: 1}, target, target) (effect: the condition:
        a primary boost, zero changes print); each stat the boost did not leave unchanged -> effectState.def / spd--."""
        from vgcjax.battle import move_pipeline as MP
        r = jnp.clip(hc.target, 0, 7)
        s = CO.vol_state(st, r, _SP).astype(I32)
        stop = jnp.asarray(False) if first else s[_SP_L] >= 3
        go = ~stop
        layers = jnp.int32(1) if first else s[_SP_L] + 1
        d0 = jnp.int32(0) if first else s[_SP_D]
        s0 = jnp.int32(0) if first else s[_SP_S]
        st = EV.emit(st, go, EV.EV_START, actor=r, move=CO.eff("stockpile"), v1=-1, v2=layers)
        b0 = mget(st.mons.boosts, r).astype(I32)
        st, _ = MP.boost(st, r, jnp.zeros(7, I32).at[1].set(1).at[3].set(1), go, primary=True, source=r,
                         effect=CO.eff("stockpile"), simple=True)
        b1 = mget(st.mons.boosts, r).astype(I32)
        st = CO.set_vol_field(st, r, _SP, _SP_L, layers, go)
        st = CO.set_vol_field(st, r, _SP, _SP_D, d0 - (b1[1] != b0[1]).astype(I32), go)
        st = CO.set_vol_field(st, r, _SP, _SP_S, s0 - (b1[3] != b0[3]).astype(I32), go)
        return hc.ret(RS.R(RS.BOOL, 0), stop), st, rng

    @handler(ids.HOOK_START, CO.eff("stockpile"), ps="data/moves.ts:17971-17980 stockpile.condition.onStart")
    def stockpile_start(hc, st, dex, rng):
        return _stockpile_layer(hc, st, dex, rng, True)

    @handler(ids.HOOK_RESTART, CO.eff("stockpile"), ps="data/moves.ts:17981-17990 stockpile.condition.onRestart")
    def stockpile_restart(hc, st, dex, rng):
        return _stockpile_layer(hc, st, dex, rng, False)

    # onEnd (data/moves.ts:17991-18001: restore def / spd, '-end' X 'Stockpile') is not registered: only Spit Up /
    # Swallow remove the volatile (outside the impl stage); switching out clears it without End


# ================================================================== Fairy Lock (data/moves.ts:5053-5077)
FL = mid("fairylock") > 0 and has("fairylock")
if FL:
    @handler(ids.HOOK_START, CO.eff("fairylock"), ps="data/moves.ts:5065-5067 fairylock.condition.onFieldStart",
             prefix=ids.PFX_FIELD)
    def fairylock_start(hc, st, dex, rng):
        """'-fieldactivate' 'move: Fairy Lock' (duration 2: the residual walk ends it; no FieldRestart: a second
        Fairy Lock fails)."""
        return hc, EV.emit(st, True, EV.EV_FIELDACTIVATE, move=CO.eff("fairylock")), rng


def fairylock_trapped(st, dex, r):
    """fairylock.condition.onTrapPokemon (data/moves.ts:5068-5070), a field handler of runEvent('TrapPokemon') for
    every active mon: tryTrap() (sim/pokemon.ts tryTrap: runStatusImmunity('trapped'): Ghost types are immune)."""
    if not FL:
        return jnp.zeros((), bool)
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    immune = TC.status_immune(dex, ids.IMMUNITY_KEYS.index("trapped"), mget(st.mons.types, r))
    return CO.has_pseudo(st, CO.cid("fairylock")) & ~immune


# ================================================================== Attract (data/moves.ts:706-761): the volatile is
# effects/batch_passive.py's ('attract': Start / BeforeMove / End / onUpdate)
def try_immunity(st, move, targets, u):
    """attract.onTryImmunity(target, source) (data/moves.ts:754-756): not (M and F) / (F and M) -> '-immune'
    (hitStepTryImmunity). Returns bool [T]."""
    if mid("attract") <= 0:
        return jnp.zeros(jnp.shape(targets), bool)
    from vgcjax.effects import batch_passive as BP
    g = BP._gender(st, jnp.asarray(targets, I32))
    gs = BP._gender(st, u)
    ok = ((g == 0) & (gs == 1)) | ((g == 1) & (gs == 0))
    return (jnp.asarray(move) == mid("attract")) & ~ok


# ================================================================== Curse (Champions: data/mods/champions/moves.ts:165-196;
# the condition: data/moves.ts:3295-3303)
CURSE = mid("curse") > 0 and has("curse")


def curse_self(st, u, move):
    """A non-Ghost user's Curse targets 'self' (request: sim/pokemon.ts:1020-1024; resolveAction:
    sim/battle-queue.ts:263-266; onModifyMove)."""
    if not CURSE:
        return jnp.zeros(jnp.shape(move), bool)
    return (jnp.asarray(move) == mid("curse")) & ~has_type(st, u, ids.TYPE_GHOST)


def curse_tt(st, u, move, tt):
    """The target type of `move` for user u with Curse's non-Ghost 'self'."""
    return jnp.where(curse_self(st, u, move), ids.TARGET_SELF, tt)


def curse_modify_target(st, dex, rng, u, move, target, on):
    """curse.onModifyMove (Champions :167-174) and useMoveInner's retargeting (sim/battle-actions.ts:494-504): a
    non-Ghost user -> move.target 'self' (target = getRandomTarget = the user, no draw); a Ghost user with no target
    or an ally target (source !== target) -> move.target 'randomNormal': baseTarget !== move.target after both the
    ModifyMove singleEvent and runEvent, so getRandomTarget draws twice (RANDOM_TARGET ord 2, same key) and the
    second pick stays. Returns (target, rng)."""
    if not CURSE:
        return target, rng
    from vgcjax.battle import move_pipeline as MP
    target = jnp.asarray(target, I32)
    is_c = jnp.asarray(on) & (jnp.asarray(move) == mid("curse"))
    ghost = has_type(st, u, ids.TYPE_GHOST)
    target = jnp.where(is_c & ~ghost, u, target)
    rand = is_c & ghost & ((target < 0) | ((target != u) & (jnp.clip(target, 0, 7) // 4 == jnp.clip(u, 0, 7) // 4)))
    _, rng = MP.get_random_target(st, dex, rng, u, move, 2, rand)
    r2, rng = MP.get_random_target(st, dex, rng, u, move, 2, rand)
    return jnp.where(rand, r2, target), rng


if CURSE:
    @handler(ids.HOOK_TRYHIT, meff("curse"), ps="data/mods/champions/moves.ts:175-180 curse.onTryHit")
    def curse_try_hit(hc, st, dex, rng):
        """A Ghost source and a cursed target -> false."""
        bad = has_type(st, hc.source, ids.TYPE_GHOST) & CO.has_volatile(st, jnp.clip(hc.target, 0, 7),
                                                                         CO.cid("curse"))
        return hc.ret(RS.R(RS.BOOL, 0), bad), st, rng

    @handler(ids.HOOK_HIT, meff("curse"), ps="data/mods/champions/moves.ts:181-195 curse.onHit")
    def curse_hit(hc, st, dex, rng):
        """Non-Ghost source: return !!boost({spe: -1, atk: 1, def: 1}, source, source) (effect Curse: a primary boost,
        key order spe atk def). Ghost: directDamage(maxhp / 2, source, source) ('-damage' source hp; trunc); an ally
        target -> getRandomTarget(source, 'Curse') (RANDOM_TARGET ord 2; none -> false); delete the target's curse
        (no End), target.addVolatile('curse') (source: the event source)."""
        from vgcjax.battle import faint as F
        from vgcjax.battle import move_pipeline as MP
        u = jnp.clip(hc.source, 0, 7)
        t = jnp.asarray(hc.target, I32)
        ghost = has_type(st, u, ids.TYPE_GHOST)
        st, br = MP.boost(st, u, jnp.zeros(7, I32).at[4].set(-1).at[0].set(1).at[1].set(1), ~ghost,
                          order=jnp.array([4, 0, 1, 2, 3, 5, 6], I32), primary=True, source=u, effect=meff("curse"),
                          simple=True)
        st, _ = F.damage(st, u, mget(st.mons.maxhp, u).astype(I32) // 2, ghost)
        st = EV.emit_hp(st, ghost, EV.EV_DAMAGE, u)
        ally = ghost & (jnp.clip(t, 0, 7) // 4 == u // 4)
        rnd, rng = MP.get_random_target(st, dex, rng, u, mid("curse"), 2, ally)
        t2 = jnp.clip(jnp.where(ally, rnd, t), 0, 7)
        none = ally & (rnd < 0)
        go = ghost & ~none
        c = CO.cid("curse")
        st = CO._set_vol_bit(st, t2, c, False, go)
        st = CO.set_vol_state(st, t2, c, jnp.zeros(6, I16), go)
        st, rng, _ = CO.add_volatile(st, dex, rng, t2, c, go, source=u, source_effect=meff("curse"))
        hc = hc.ret(RS.R(RS.BOOL, RS.truthy(br).astype(I32)), ~ghost)
        return hc.ret(RS.R(RS.BOOL, 0), none), st, rng

    @handler(ids.HOOK_START, CO.eff("curse"), ps="data/moves.ts:3296-3298 curse.condition.onStart")
    def curse_start(hc, st, dex, rng):
        """'-start' pokemon 'Curse' '[of] source'."""
        return hc, EV.emit(st, True, EV.EV_START, actor=hc.target, move=CO.eff("curse"), v1=-1), rng

    @handler(ids.HOOK_RESIDUAL, CO.eff("curse"), ps="data/moves.ts:3299-3302 curse.condition.onResidual")
    def curse_residual(hc, st, dex, rng):
        """onResidualOrder 12: damage(baseMaxhp / 4) ('-damage' X hp '[from] Curse'; Damage event: Magic Guard)."""
        from vgcjax.effects import statuses as STS
        r = jnp.clip(hc.holder, 0, 7)
        st, rng = STS.effect_damage(st, dex, rng, r, mget(st.mons.maxhp, r).astype(I32) // 4, CO.eff("curse"), True)
        return hc, st, rng


# ================================================================== Instruct (data/moves.ts:9643-9672)
if mid("instruct") > 0:
    @handler(ids.HOOK_HIT, meff("instruct"), ps="data/moves.ts:9652-9668 instruct.onHit")
    def instruct_hit(hc, st, dex, rng):
        """No lastMove, flags failinstruct / charge / recharge, a focuspunch volatile (Beak Blast / Shell Trap: not in
        Reg M-C) or a move slot with 0 PP -> false; else '-singleturn' target 'move: Instruct' '[of] source' and
        queue.prioritizeAction(queue.resolveAction({choice: 'move', pokemon: target, moveid: lastMove,
        targetLoc: lastMoveTargetLoc})[0]) (sim/battle-queue.ts:166-275, 282-291): resolveAction (midTurn false;
        draws under step_tag COMMIT of the enclosing action, contract/drawkey.md §3.1) resolves a beforeTurnMove
        (Counter / Mirror Coat) or priorityChargeMove (Chilly Reception) satellite first, which is then element [0]
        and the only action prioritized; runEvent('FractionalPriority') (Quick Claw / Quick Draw draws); a missing
        targetLoc -> getRandomTarget (battle/step.py resolve_loc); getActionSpeed per resolved action (one ModifySpe
        each). The action gets order 3 and no sourceEffect."""
        from vgcjax.battle import queue as Q
        from vgcjax.battle import step as SP
        from vgcjax.battle import move_pipeline as MP
        from vgcjax.effects import batch7c as B7C
        from vgcjax.effects import batch_utility as BU
        t = jnp.clip(hc.target, 0, 7)
        m = st.mons
        lm = mget(m.last_move, t).astype(I32)
        lmc = jnp.maximum(lm, 0)
        mids = mget(m.move_ids, t).astype(I32)
        in_slot = (mids == lm) & (lm > 0)
        has_slot = jnp.any(in_slot)
        slot = jnp.argmax(in_slot).astype(I32)
        pp = mget(m.pp, t)[slot].astype(I32)
        fp = CO.has_volatile(st, t, CO.cid("focuspunch")) if has("focuspunch") else jnp.zeros((), bool)
        fail = (lm <= 0) | _mflag(dex, lmc, ids.MF_FAILINSTRUCT) | _mflag(dex, lmc, ids.MF_CHARGE) | \
            _mflag(dex, lmc, ids.MF_RECHARGE) | fp | (has_slot & (pp <= 0))
        go = ~fail
        st = EV.emit(st, go, EV.EV_SINGLETURN, actor=t, move=meff("instruct"))
        sc = st.scratch
        tag0 = sc.step_tag
        st = C.set_step(st, ids.STEP_COMMIT, sc.step_actor, sc.step_subject, sc.action_ord, go)
        btm = B7C.has_before_turn(lmc)
        pcc = BU.has_priority_charge(lmc)
        st, frac, rng = Q.frac_priority(st, dex, rng, t, lmc, go)
        loc = SP.resolve_loc(st, dex, t, lmc, mget(m.last_move_target_loc, t).astype(I32))
        st = C.mul_tick(st, 1 + (btm | pcc).astype(I32), go)                   # getActionSpeed per resolved action
        spd = C.action_speed(st, dex, t)
        kind = jnp.where(btm, S.Q_BEFORE_TURN_MOVE, jnp.where(pcc, S.Q_PRIORITY_CHARGE, S.Q_MOVE))
        is_move = kind == S.Q_MOVE
        st = Q.add_entry(st, go, kind=kind, side=t // 4, b=t % 4, order=3,
                         priority10=jnp.where(is_move, 10 * Q.move_priority(st, dex, t, lmc) + frac, 0),
                         frac10=jnp.where(is_move, frac, 0), speed=spd,
                         move_slot=jnp.where(is_move & has_slot, slot, -1), move_id=lmc,
                         target_loc=jnp.where(is_move, loc, 0), original_target=jnp.where(is_move, MP.at_loc(
                             st, t, loc), -1))
        sc = st.scratch
        st = st.replace(scratch=sc.replace(step_tag=jnp.where(go, tag0, sc.step_tag)))
        return hc.ret(RS.R(RS.BOOL, 0), fail), st, rng
