"""Utility move batch: Leech Seed, Belly Drum, Focus Energy, Dragon Cheer, No Retreat, Destiny Bond, Quash, Round,
Ally Switch, Chilly Reception.

PS: data/moves.ts (line numbers below), Champions overrides in data/mods/champions/moves.ts: Dragon Cheer
(:268-271, flags only) and No Retreat (:692-700, condition.onStart). Call sites outside this module:
- crit ratio (Focus Energy / Dragon Cheer onModifyCritRatio): battle/move_pipeline.py get_damage;
- Leech Seed onTryImmunity: battle/move_pipeline.py trySpreadMoveHit step 3;
- Destiny Bond onFaint: battle/faint.py resolve_faints; onMoveAborted: battle/move_pipeline.py move_begin;
- Round basePowerCallback (move.sourceEffect): battle/move_pipeline.py move_begin (queue entry source_effect);
- Chilly Reception priorityChargeCallback (the priorityChargeMove action): battle/step.py commit_move / run_light;
- the `trapped` volatile's onTrapPokemon (No Retreat): battle/step.py end_turn.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset
from vgcjax.core import rng as R
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32


def mid(name):
    return getattr(ids, f"MOVE_{name.upper()}", -999)


def has(name):
    return name in ids.CONDITIONS


def meff(name):
    """EffectId of a move (a Python int: usable at trace time; common.eff_move without jnp)."""
    m = mid(name)
    return ids.MOVE_BASE + m - 1 if m > 0 else -1


def _vol(st, r, name):
    if not has(name):
        return jnp.zeros(jnp.shape(r), bool)
    return CO.has_volatile(st, r, CO.cid(name))


def _trap_immune(st, dex, r):
    """runStatusImmunity('trapped') (sim/pokemon.ts:2290-2310): dex.getImmunity('trapped') -> Ghost types."""
    from vgcjax.core import typechart as TC
    return TC.status_immune(dex, ids.IMMUNITY_KEYS.index("trapped"), mget(st.mons.types, jnp.clip(r, 0, 7)))


# ------------------------------------------------------------------ Leech Seed (data/moves.ts:10204-10238)
LS_SLOT = 1                 # raw state slot: effectState.sourceSlot as side * 2 + position (not projected)
if mid("leechseed") > 0 and has("leechseed"):
    CO.fields("leechseed")
    _LS = CO.cid("leechseed")

    @handler(ids.HOOK_START, CO.eff("leechseed"), ps="data/moves.ts:10213-10215 leechseed.condition.onStart")
    def leechseed_start(hc, st, dex, rng):
        """'-start' target 'move: Leech Seed'; addVolatile stored sourceSlot = source.getSlot() (sim/pokemon.ts:2023)."""
        from vgcjax.battle import move_pipeline as MP
        r = hc.target
        src = jnp.clip(hc.source, 0, 7)
        code = (src // 4) * 2 + jnp.maximum(MP.position(st, src), 0)
        st = CO.set_vol_state(st, r, _LS, CO.vol_state(st, r, _LS).at[LS_SLOT].set(code.astype(I16)), True)
        return hc, EV.emit(st, True, EV.EV_START, actor=r, move=CO.eff("leechseed"), v1=-1), rng

    @handler(ids.HOOK_RESIDUAL, CO.eff("leechseed"), ps="data/moves.ts:10216-10228 leechseed.condition.onResidual")
    def leechseed_residual(hc, st, dex, rng):
        """onResidualOrder 8: target = getAtSlot(sourceSlot); none / fainted / hp 0 -> nothing; else
        damage(baseMaxhp / 8, pokemon, target) ('-damage' X hp '[from] Leech Seed' '[of] target') and, if it dealt
        damage, heal(damage, target, pokemon) ('-heal' target hp '[silent]', sim/battle.ts:2293-2296)."""
        from vgcjax.battle import faint as F
        from vgcjax.effects import statuses as STS
        r = hc.holder
        code = CO.vol_state(st, r, _LS)[LS_SLOT].astype(I32)
        side, pos = jnp.clip(code // 2, 0, 1), jnp.clip(code % 2, 0, 1)
        b = st.sides.active[side, pos].astype(I32)
        t = side * 4 + jnp.maximum(b, 0)
        m = st.mons
        ok = (b >= 0) & ~mget(m.fainted, t) & (mget(m.hp, t) > 0)
        hp0 = mget(m.hp, r).astype(I32)
        st, rng = STS.effect_damage(st, dex, rng, r, mget(m.maxhp, r).astype(I32) // 8, CO.eff("leechseed"), ok,
                                    source=t)
        dealt = hp0 - mget(st.mons.hp, r).astype(I32)
        m = st.mons
        thp = mget(m.hp, t)
        from vgcjax.effects import batch_damaging as BD   # battle.heal: runEvent('TryHeal') first (Heal Block: false)
        go = ok & (dealt > 0) & ~BD.heal_blocked(st, t) & (thp > 0) & mget(m.is_active, t) & (thp < mget(m.maxhp, t))
        st, _ = F.heal(st, t, dealt, go)
        return hc, EV.emit_hp(st, go, EV.EV_HEAL, t), rng


# ------------------------------------------------------------------ Focus Energy / Dragon Cheer
if mid("focusenergy") > 0 and has("focusenergy"):
    @handler(ids.HOOK_START, CO.eff("focusenergy"), ps="data/moves.ts:5984-5994 focusenergy.condition.onStart")
    def focusenergy_start(hc, st, dex, rng):
        """A dragoncheer volatile -> false; else '-start' target 'move: Focus Energy' (the [zeffect] / [silent]
        variants come from Z-Power / Psych Up / Transform / Imposter / Costar: not reachable in Reg M-C)."""
        r = hc.target
        dc = _vol(st, r, "dragoncheer")
        st = EV.emit(st, ~dc, EV.EV_START, actor=r, move=CO.eff("focusenergy"), v1=-1)
        return hc.ret(RS.R(RS.BOOL, 0), dc), st, rng

if mid("dragoncheer") > 0 and has("dragoncheer"):
    CO.fields("dragoncheer", ("hasDragonType", "bool"))
    _DC = CO.cid("dragoncheer")

    @handler(ids.HOOK_START, CO.eff("dragoncheer"), ps="data/moves.ts:4069-4082 dragoncheer.condition.onStart")
    def dragoncheer_start(hc, st, dex, rng):
        """A focusenergy volatile -> false; else '-start' target 'move: Dragon Cheer', effectState.hasDragonType =
        target.hasType('Dragon')."""
        r = hc.target
        fe = _vol(st, r, "focusenergy")
        st = EV.emit(st, ~fe, EV.EV_START, actor=r, move=CO.eff("dragoncheer"), v1=-1)
        dragon = jnp.any(mget(st.mons.types, r) == ids.TYPE_DRAGON)
        st = CO.set_vol_field(st, r, _DC, CO.slot_of("dragoncheer", "hasDragonType"), dragon.astype(I32), ~fe)
        return hc.ret(RS.R(RS.BOOL, 0), fe), st, rng


def crit_ratio_bonus(st, u):
    """runEvent('ModifyCritRatio', source) additions of the user's volatiles: Focus Energy +2
    (data/moves.ts:5995-5997); Dragon Cheer +2 with hasDragonType else +1 (data/moves.ts:4083-4085)."""
    out = jnp.zeros((), I32)
    if has("focusenergy"):
        out = out + 2 * _vol(st, u, "focusenergy").astype(I32)
    if has("dragoncheer"):
        hd = CO.vol_state(st, u, CO.cid("dragoncheer"))[CO.slot_of("dragoncheer", "hasDragonType")] != 0
        out = out + jnp.where(_vol(st, u, "dragoncheer"), jnp.where(hd, 2, 1), 0)
    return out


# ------------------------------------------------------------------ Belly Drum (data/moves.ts:1216-1236)
if mid("bellydrum") > 0:
    @handler(ids.HOOK_HIT, meff("bellydrum"), ps="data/moves.ts:1225-1231 bellydrum.onHit")
    def bellydrum_hit(hc, st, dex, rng):
        """hp <= maxhp / 2, Atk +6 or maxhp 1 -> false; directDamage(maxhp / 2) ('-damage' X hp); boost({atk: 12})
        (sim/battle.ts:2037-2106: ChangeBoost (Contrary inverts), capped; '-setboost' target atk N '[from] move: Belly
        Drum', or '-unboost' target atk 0 when nothing changed (Contrary at -6))."""
        from vgcjax.battle import faint as F
        t = hc.target
        m = st.mons
        hp, mx = mget(m.hp, t).astype(I32), mget(m.maxhp, t).astype(I32)
        b = mget(m.boosts, t)
        atk = b[0].astype(I32)
        fail = (2 * hp <= mx) | (atk >= 6) | (mx == 1)
        go = ~fail
        st, _ = F.damage(st, t, jnp.maximum(mx // 2, 1), go)
        st = EV.emit_hp(st, go, EV.EV_DAMAGE, t)
        m = st.mons
        go = go & (mget(m.hp, t) > 0) & mget(m.is_active, t) & (st.sides.pokemon_left[1 - jnp.clip(t, 0, 7) // 4] > 0)
        contrary = mget(m.ability, t) == getattr(ids, "ABILITY_CONTRARY", -999)
        new = jnp.clip(atk + jnp.where(contrary, -12, 12), -6, 6)
        changed = go & (new != atk)
        m = m.replace(boosts=mset(m.boosts, t, b.at[0].set(new.astype(b.dtype)), changed),
                      stats_raised_this_turn=mset(m.stats_raised_this_turn, t, True, changed & ~contrary),
                      stats_lowered_this_turn=mset(m.stats_lowered_this_turn, t, True, changed & contrary))
        st = st.replace(mons=m)
        st = EV.emit(st, changed, EV.EV_SETBOOST, actor=t, v1=0, v2=new, move=meff("bellydrum"))
        st = EV.emit(st, go & ~changed, EV.EV_BOOST, actor=t, v1=0, v2=0, flags=EV.F_UNBOOST)
        return hc.ret(RS.R(RS.BOOL, 0), fail), st, rng


# ------------------------------------------------------------------ No Retreat (data/moves.ts:12789-12817)
if mid("noretreat") > 0 and has("noretreat") and has("trapped"):
    @handler(ids.HOOK_TRY, meff("noretreat"), ps="data/moves.ts:12799-12801 noretreat.onTry")
    def noretreat_try(hc, st, dex, rng):
        """source.volatiles['noretreat'] -> false."""
        return hc.ret(RS.R(RS.BOOL, 0), _vol(st, hc.target, "noretreat")), st, rng

    @handler(ids.HOOK_START, CO.eff("noretreat"), ps="data/mods/champions/moves.ts:692-700 noretreat.condition.onStart")
    def noretreat_start(hc, st, dex, rng):
        """[C] addVolatile('trapped') failing (already trapped, or runStatusImmunity('trapped'): Ghost) returns
        undefined (the volatile stays, no message); else trapped.onStart '-activate' X 'trapped'
        (data/conditions.ts:214-216), then '-start' X 'move: No Retreat'. trapped has no other add-time handler."""
        r = hc.target
        add = ~_trap_immune(st, dex, r) & ~_vol(st, r, "trapped") & (mget(st.mons.hp, r) > 0)
        st = CO.add_plain_volatile(st, r, CO.cid("trapped"), add)
        st = EV.emit(st, add, EV.EV_ACTIVATE, actor=r, move=CO.eff("trapped"))
        return hc, EV.emit(st, add, EV.EV_START, actor=r, move=CO.eff("noretreat"), v1=-1), rng


def trapped_volatile(st, dex, r):
    """trapped.onTrapPokemon (data/conditions.ts:211-213): tryTrap() -> pokemon.trapped = true unless Ghost."""
    if not has("trapped"):
        return jnp.zeros((), bool)
    return _vol(st, r, "trapped") & ~_trap_immune(st, dex, r)


# ------------------------------------------------------------------ Destiny Bond (data/moves.ts:3482-3524)
if mid("destinybond") > 0 and has("destinybond"):
    _DB = CO.cid("destinybond")

    @handler(ids.HOOK_PREPAREHIT, meff("destinybond"), ps="data/moves.ts:3492-3494 destinybond.onPrepareHit")
    def destinybond_prepare_hit(hc, st, dex, rng):
        """return !pokemon.removeVolatile('destinybond') (no End handler)."""
        u = hc.target
        had = CO.has_volatile(st, u, _DB)
        st, rng, _ = CO.remove_volatile(st, dex, rng, u, _DB, had)
        return hc.ret(RS.R(RS.BOOL, (~had).astype(I32))), st, rng

    @handler(ids.HOOK_START, CO.eff("destinybond"), ps="data/moves.ts:3497-3499 destinybond.condition.onStart")
    def destinybond_start(hc, st, dex, rng):
        """'-singlemove' pokemon 'Destiny Bond'."""
        return hc, EV.emit(st, True, EV.EV_SINGLEMOVE, actor=hc.target, move=CO.eff("destinybond")), rng

    @handler(ids.HOOK_BEFOREMOVE, CO.eff("destinybond"), ps="data/moves.ts:3511-3516 destinybond.condition.onBeforeMove")
    def destinybond_before_move(hc, st, dex, rng):
        """onBeforeMovePriority -1: a move other than Destiny Bond -> removeVolatile('destinybond')."""
        st, rng, _ = CO.remove_volatile(st, dex, rng, hc.target, _DB, hc.effect != meff("destinybond"))
        return hc, st, rng


def destinybond_move_aborted(st, dex, rng, u, on):
    """destinybond.condition.onMoveAborted (data/moves.ts:3517-3519): removeVolatile('destinybond')."""
    if not (mid("destinybond") > 0 and has("destinybond")):
        return st, rng
    st, rng, _ = CO.remove_volatile(st, dex, rng, u, CO.cid("destinybond"), on)
    return st, rng


def destinybond_faint(st, ref, src, on):
    """destinybond.condition.onFaint(target, source, effect) (data/moves.ts:3500-3510) for a faint whose source is the
    move user `src` (effect: its move, never a future move in Reg M-C): source has hp and is not an ally ->
    '-activate' target 'move: Destiny Bond'; the caller faints the source. Returns (st, activated)."""
    if not (mid("destinybond") > 0 and has("destinybond")):
        return st, jnp.zeros((), bool)
    s = jnp.clip(jnp.asarray(src, I32), 0, 7)
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    go = jnp.asarray(on) & (jnp.asarray(src) >= 0) & CO.has_volatile(st, r, CO.cid("destinybond")) & \
        (mget(st.mons.hp, s) > 0) & (s // 4 != r // 4)
    return EV.emit(st, go, EV.EV_ACTIVATE, actor=r, move=meff("destinybond")), go


# ------------------------------------------------------------------ Quash (data/moves.ts:14451-14470)
if mid("quash") > 0:
    @handler(ids.HOOK_HIT, meff("quash"), ps="data/moves.ts:14460-14467 quash.onHit")
    def quash_hit(hc, st, dex, rng):
        """queue.willMove(target) (sim/battle-queue.ts:324-332): none -> false; else action.order = 201 and
        '-activate' target 'move: Quash'."""
        from vgcjax.battle import queue as Q
        t = hc.target
        q = st.queue
        cand = q.valid & (q.kind == S.Q_MOVE) & (Q.entry_ref(q) == t) & ~mget(st.mons.fainted, jnp.clip(t, 0, 7))
        found = jnp.any(cand)
        i = jnp.argmin(jnp.where(cand, q.seq.astype(I32), 1 << 30))
        st = st.replace(queue=q.replace(order=q.order.at[i].set(jnp.where(found, 201, q.order[i]).astype(q.order.dtype))))
        st = EV.emit(st, found, EV.EV_ACTIVATE, actor=t, move=meff("quash"))
        return hc.ret(RS.R(RS.BOOL, 0), ~found), st, rng


# ------------------------------------------------------------------ Round (data/moves.ts:15494-15522)
if mid("round") > 0:
    @handler(ids.HOOK_TRY, meff("round"), ps="data/moves.ts:15509-15517 round.onTry")
    def round_try(hc, st, dex, rng):
        """The first queued Round move action (queue list order = the sorted order: best key, then insertion) ->
        queue.prioritizeAction(action, move) (sim/battle-queue.ts:283-293): order 3, sourceEffect Round (stored as
        the MoveId in the entry's source_effect; move_begin doubles the base power, basePowerCallback :15498)."""
        from vgcjax.battle import queue as Q
        q = st.queue
        cand = q.valid & (q.kind == S.Q_MOVE) & (q.move_id == mid("round"))
        found = jnp.any(cand)
        best = Q.best_mask(q.replace(valid=cand))
        i = jnp.argmin(jnp.where(best, q.seq.astype(I32), 1 << 30))
        q = q.replace(order=q.order.at[i].set(jnp.where(found, 3, q.order[i]).astype(q.order.dtype)),
                      source_effect=q.source_effect.at[i].set(
                          jnp.where(found, mid("round"), q.source_effect[i]).astype(q.source_effect.dtype)))
        return hc, st.replace(queue=q), rng


def round_bp(move, source_effect, bp):
    """round.basePowerCallback (data/moves.ts:15498-15504): move.sourceEffect === 'round' -> basePower * 2."""
    if mid("round") <= 0:
        return bp
    return jnp.where((jnp.asarray(move) == mid("round")) & (jnp.asarray(source_effect) == mid("round")), bp * 2, bp)


# ------------------------------------------------------------------ Ally Switch (data/moves.ts:302-358)
if mid("allyswitch") > 0 and has("allyswitch"):
    CO.fields("allyswitch", ("counter", "int"))
    _AS = CO.cid("allyswitch")

    @handler(ids.HOOK_PREPAREHIT, meff("allyswitch"), ps="data/moves.ts:311-313 allyswitch.onPrepareHit")
    def allyswitch_prepare_hit(hc, st, dex, rng):
        """return pokemon.addVolatile('allyswitch')."""
        st, rng, r = CO.add_volatile(st, dex, rng, hc.target, _AS, True, source=hc.source, source_effect=hc.effect)
        return hc.ret(r), st, rng

    @handler(ids.HOOK_START, CO.eff("allyswitch"), ps="data/moves.ts:335-337 allyswitch.condition.onStart")
    def allyswitch_start(hc, st, dex, rng):
        """effectState.counter = 3."""
        return hc, CO.set_vol_field(st, hc.target, _AS, CO.slot_of("allyswitch", "counter"), 3, True), rng

    @handler(ids.HOOK_RESTART, CO.eff("allyswitch"), ps="data/moves.ts:338-352 allyswitch.condition.onRestart")
    def allyswitch_restart(hc, st, dex, rng):
        """randomChance(1, counter || 1) (ALLY_SWITCH_STALL): failure deletes the volatile (no End) and returns false;
        success: counter * 3 below counterMax 729, duration = 2 (undefined -> singleEvent's relay true)."""
        r = hc.target
        k = CO.slot_of("allyswitch", "counter")
        s = CO.vol_state(st, r, _AS)
        cnt = jnp.maximum(s[k].astype(I32), 1)
        key = C.dkey(st, ids.DK_ALLY_SWITCH_STALL, actor=r, subject=CO.eff("allyswitch"))
        ok, rng = R.chance(rng, key, 1, cnt)
        st = CO._set_vol_bit(st, r, _AS, False, ~ok)
        st = CO.set_vol_state(st, r, _AS, jnp.zeros(6, I16), ~ok)
        s = s.at[k].set(jnp.where(s[k] < 729, s[k] * 3, s[k]).astype(I16)).at[0].set(2)
        st = CO.set_vol_state(st, r, _AS, s, ok)
        return hc.ret(RS.R(RS.BOOL, ok.astype(I32))), st, rng

    @handler(ids.HOOK_HIT, meff("allyswitch"), ps="data/moves.ts:314-332 allyswitch.onHit")
    def allyswitch_hit(hc, st, dex, rng):
        """newPosition = the other slot; empty or fainted -> '-fail' pokemon 'move: Ally Switch', attrLastMove('[still]'),
        NOT_FAIL; else swapPosition(pokemon, newPosition, '[from] move: Ally Switch')."""
        from vgcjax.battle import move_pipeline as MP
        u = hc.target
        s = jnp.clip(u, 0, 7) // 4
        np_ = 1 - jnp.maximum(MP.position(st, u), 0)
        b = st.sides.active[s, np_].astype(I32)
        fail = (b < 0) | mget(st.mons.fainted, s * 4 + jnp.maximum(b, 0))
        st = EV.emit(st, fail, EV.EV_FAIL, actor=u, move=meff("allyswitch"), flags=EV.F_FAILSTATUS)
        st = EV.attr_last_move(st, fail, EV.F_STILL)
        st, rng = swap_position(st, dex, rng, u, ~fail, meff("allyswitch"))
        return hc.ret(RS.R(RS.NOTFAIL, 0), fail), st, rng


def swap_position(st, dex, rng, u, on, from_eff):
    """PS: sim/battle.ts:1605-1624 swapPosition(pokemon, newPosition, attributes) in doubles: 'swap' pokemon
    newPosition [from] effect; side.pokemon / side.active / position exchanged; runEvent('Swap', target, pokemon)
    then runEvent('Swap', pokemon, target): the only Reg M-C onSwap is the Healing Wish slot condition at the mon's
    new position (data/moves.ts:8373-8381; effects/moves_b2.py). Returns (st, rng)."""
    from vgcjax.battle import move_pipeline as MP
    from vgcjax.battle.util import tree_where
    on = jnp.asarray(on)
    s = jnp.clip(u, 0, 7) // 4
    p = jnp.maximum(MP.position(st, u), 0)
    np_ = 1 - p
    t = s * 4 + jnp.maximum(st.sides.active[s, np_].astype(I32), 0)
    st = EV.emit(st, on, EV.EV_SWAP, actor=u, v1=np_, move=from_eff)
    sd = st.sides
    act, order = sd.active[s], sd.ps_order[s]
    sd = sd.replace(active=sd.active.at[s].set(jnp.where(on, act[::-1], act)),
                    ps_order=sd.ps_order.at[s].set(jnp.where(on, order.at[0].set(order[1]).at[1].set(order[0]), order)))
    st = st.replace(sides=sd)
    if mid("healingwish") > 0 and has("healingwish"):
        from vgcjax.effects import hooks as H
        from vgcjax.effects import moves_b2 as MB2
        k = CO.cid("healingwish") - ids.SLOT_BASE
        for r, pos in ((t, p), (u, np_)):
            hw = on & (((st.sides.slot_cond_bits[s, pos] >> jnp.uint32(k)) & 1) == 1)
            hc = H.HookCtx.make(r, -1, CO.eff("healingwish"))
            _, st2, rng2 = MB2.healingwish_switch_in(hc, st, dex, rng)
            st, rng = tree_where(hw, (st2, rng2), (st, rng))
    return st, rng


# ------------------------------------------------------------------ Chilly Reception (data/moves.ts:2396-2420)
def has_priority_charge(move_id):
    """move.priorityChargeCallback (sim/battle-queue.ts:236-242 resolveAction): Chilly Reception is the only one the
    engine implements (Focus Punch / Shell Trap / Beak Blast stay outside the impl stage)."""
    if mid("chillyreception") <= 0:
        return jnp.zeros(jnp.shape(move_id), bool)
    return jnp.asarray(move_id) == mid("chillyreception")


def priority_charge(st, dex, rng, ref, move_id, on):
    """The priorityChargeMove action (sim/battle.ts:2792-2797): chillyreception.priorityChargeCallback ->
    source.addVolatile('chillyreception') (data/moves.ts:2405-2407; duration 1, no Start). Returns (st, rng)."""
    if not (mid("chillyreception") > 0 and has("chillyreception")):
        return st, rng
    go = jnp.asarray(on) & has_priority_charge(move_id) & (jnp.asarray(ref) >= 0)
    st, rng, _ = CO.add_volatile(st, dex, rng, jnp.maximum(ref, 0), CO.cid("chillyreception"), go)
    return st, rng


if mid("chillyreception") > 0 and has("chillyreception"):
    @handler(ids.HOOK_BEFOREMOVE, CO.eff("chillyreception"),
             ps="data/moves.ts:2412-2416 chillyreception.condition.onBeforeMove")
    def chillyreception_before_move(hc, st, dex, rng):
        """onBeforeMovePriority 100: move Chilly Reception -> '-prepare' source 'Chilly Reception' '[premajor]'."""
        go = hc.effect == meff("chillyreception")
        return hc, EV.emit(st, go, EV.EV_PREPARE, actor=hc.target, move=mid("chillyreception")), rng
