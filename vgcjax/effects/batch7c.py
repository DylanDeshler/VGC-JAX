"""Batch 7c (Reg M-C ladder blockers): Octolock, Outrage (lockedmove), Last Resort, Synthesis, Moonlight, Mirror Coat and
Counter (beforeTurnCallback + damage tracking), Endure, Electrify, Wish, Supercell Slam (crash damage), Shed Tail,
Simple Beam, Entrainment, Focus Punch; the Simple ability (Simple Beam's result).

PS: data/moves.ts (line numbers below), data/conditions.ts:253-286 lockedmove; Champions overrides in
data/mods/champions/moves.ts: Electrify (:297-300) and Octolock (:709-712) only drop isNonstandard, Wish (:1136-1139)
sets pp 5 (generated data). Call sites outside this module (minimal hooks, each commented at the site):
- lockedmove: effects/locked.py locked() (getLockedMove: forced choice, no PP, trapped);
- Mirror Coat / Counter: battle/step.py commit_move (the beforeTurnMove action) and run_light (beforeTurnCallback),
  battle/move_pipeline.py get_move_targets (onRedirectTarget), effects/move_callbacks.py fixed_damage (damageCallback);
- Endure: battle/move_pipeline.py spread_move_hit (onDamage, priority -10), effects/volatiles.py confusion self-hit;
- Electrify: battle/move_pipeline.py move_begin_front (onModifyType, priority -2);
- Focus Punch: effects/batch_utility.py priority_charge (priorityChargeCallback), battle/move_pipeline.py
  move_begin_front (beforeMoveCallback) and spread_move_hit (condition onHit = runEvent('Hit')),
  effects/ability_blocks.py add_volatile_block (condition onTryAddVolatile);
- Octolock: battle/move_pipeline.py move_begin_back step 3 (onTryImmunity), battle/step.py end_turn (onTrapPokemon);
- Shed Tail: effects/volatiles.py substitute_start ('[from] move: Shed Tail'), battle/move_pipeline.py has_self
  (self.onHit), battle/switching.py switch_in (copyVolatileFrom(old, 'shedtail'));
- Supercell Slam: effects/batch_damaging.py crash_damage (hasCrashDamage);
- Simple Beam / Entrainment: effects/batch_copy.py post_hit (setAbility); Simple's onChangeBoost:
  battle/move_pipeline.py boost_v and the secondary self-boost recurrence.
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


def _vol(st, r, name):
    if not has(name):
        return jnp.zeros(jnp.shape(r), bool)
    return CO.has_volatile(st, jnp.clip(jnp.asarray(r, I32), 0, 7), CO.cid(name))


def true_ability(st, r):
    """pokemon.ability: the Mold Breaker mask (effects/batch_reactive.py mold_start) is not PS state."""
    m = st.mons
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    ab, sv = mget(m.ability, r).astype(I32), mget(m.mold_saved, r).astype(I32)
    return jnp.where((ab == 0) & (sv > 0), sv, ab)


# ================================================================== lockedmove (Outrage)
LM = has("lockedmove") and mid("outrage") > 0
if LM:
    CO.fields("lockedmove", ("trueDuration", "int"), ("move", "move"))
    _LM = CO.cid("lockedmove")
    _LM_TD, _LM_MV = CO.slot_of("lockedmove", "trueDuration"), CO.slot_of("lockedmove", "move")

    @handler(ids.HOOK_START, CO.eff("lockedmove"), ps="data/conditions.ts:264-267 lockedmove.onStart")
    def lockedmove_start(hc, st, dex, rng):
        """effectState.trueDuration = random(2, 4) (LOCKEDMOVE_DURATION); effectState.move = effect.id (the move whose
        self.volatileStatus added it)."""
        r = hc.target
        key = C.dkey(st, ids.DK_LOCKEDMOVE_DURATION, actor=r, subject=CO.eff("lockedmove"))
        td, rng = R.randint(rng, key, 2, 4)
        st = CO.set_vol_field(st, r, _LM, _LM_TD, td, True)
        mv = jnp.maximum(jnp.asarray(hc.effect, I32) - ids.MOVE_BASE + 1, 0)
        return hc, CO.set_vol_field(st, r, _LM, _LM_MV, mv, True), rng

    @handler(ids.HOOK_RESTART, CO.eff("lockedmove"), ps="data/conditions.ts:268-272 lockedmove.onRestart")
    def lockedmove_restart(hc, st, dex, rng):
        """trueDuration >= 2 -> effectState.duration = 2 (returns undefined)."""
        r = hc.target
        s = CO.vol_state(st, r, _LM)
        st = CO.set_vol_state(st, r, _LM, s.at[0].set(2), s[_LM_TD] >= 2)
        return hc, st, rng

    @handler(ids.HOOK_RESIDUAL, CO.eff("lockedmove"), ps="data/conditions.ts:257-263 lockedmove.onResidual")
    def lockedmove_residual(hc, st, dex, rng):
        """(after the framework's duration tick) asleep -> delete the volatile (no End: no confusion);
        trueDuration--."""
        r = hc.holder
        slp = mget(st.mons.status, r) == ids.COND_SLP
        s = CO.vol_state(st, r, _LM)
        st = CO.set_vol_state(st, r, _LM, s.at[_LM_TD].set(s[_LM_TD] - 1), ~slp)
        st = CO._set_vol_bit(st, r, _LM, False, slp)
        return hc, CO.set_vol_state(st, r, _LM, jnp.zeros(6, I16), slp), rng

    @handler(ids.HOOK_AFTERMOVE, CO.eff("lockedmove"), ps="data/conditions.ts:273-277 lockedmove.onAfterMove")
    def lockedmove_after_move(hc, st, dex, rng):
        """effectState.duration === 1 -> removeVolatile('lockedmove')."""
        r = hc.target
        st, rng, _ = CO.remove_volatile(st, dex, rng, r, _LM, CO.vol_state(st, r, _LM)[0] == 1)
        return hc, st, rng

    @handler(ids.HOOK_END, CO.eff("lockedmove"), ps="data/conditions.ts:278-281 lockedmove.onEnd")
    def lockedmove_end(hc, st, dex, rng):
        """trueDuration > 1 -> nothing; else addVolatile('confusion') (sourceEffect lockedmove: '-start' X 'confusion'
        '[fatigue]', effects/volatiles.py)."""
        r = hc.target
        go = CO.vol_state(st, r, _LM)[_LM_TD] <= 1
        st, rng, _ = CO.add_volatile(st, dex, rng, r, CO.cid("confusion"), go, source=r,
                                     source_effect=CO.eff("lockedmove"))
        return hc, st, rng


def locked_move(st, u):
    """lockedmove.onLockMove (data/conditions.ts:282-285): effectState.move (no Dynamax). Returns (held, move id)."""
    if not LM:
        return jnp.zeros((), bool), jnp.int32(0)
    return _vol(st, u, "lockedmove"), CO.vol_state(st, u, _LM)[_LM_MV].astype(I32)


# ================================================================== Last Resort (data/moves.ts:10068-10091)
if mid("lastresort") > 0:
    @handler(ids.HOOK_TRY, meff("lastresort"), ps="data/moves.ts:10077-10087 lastresort.onTry")
    def lastresort_try(hc, st, dex, rng):
        """Fewer than 2 move slots -> false; every slot but Last Resort's must be `used` (deductPP since the switch-in,
        battle/move_pipeline.py move_used); the user must know Last Resort."""
        u = jnp.clip(hc.target, 0, 7)
        m = st.mons
        mids = mget(m.move_ids, u).astype(I32)
        used = mget(m.move_used, u)
        known = mids > 0
        is_lr = mids == mid("lastresort")
        ok = (jnp.sum(known.astype(I32)) >= 2) & jnp.any(is_lr) & ~jnp.any(known & ~is_lr & ~used)
        return hc.ret(RS.R(RS.BOOL, 0), ~ok), st, rng


# ================================================================== Synthesis / Moonlight
def _weather_heal(hc, st, dex, rng):
    """onHit(pokemon): pokemon.effectiveWeather() Sun / Desolate Land -> 0.667, Rain / Primordial Sea / Sandstorm /
    Hail / Snow -> 0.25, else 0.5; heal(modify(maxhp, factor)) ('-heal' X hp: a Move effect); failed -> '-fail' X
    'heal', NOT_FAIL; else true. (Mega Sol's forced sun: not in the impl stage.)"""
    from vgcjax.battle import move_pipeline as MP
    from vgcjax.core import fixedpoint as FP
    from vgcjax.effects import weather as WE
    u = jnp.clip(hc.target, 0, 7)
    w = WE.mon_weather(st, u)
    sun = jnp.zeros((), bool)
    for n in ("sunnyday", "desolateland"):
        if has(n):
            sun = sun | (w == CO.cid(n))
    weak = jnp.zeros((), bool)
    for n in ("raindance", "primordialsea", "sandstorm", "hail", "snowscape"):
        if has(n):
            weak = weak | (w == CO.cid(n))
    m4096 = jnp.where(sun, 2732, jnp.where(weak, 1024, 2048))                     # trunc(factor * 4096)
    st, r = MP.battle_heal(st, u, FP.modify_raw(mget(st.mons.maxhp, u).astype(I32), m4096), True)
    ok = RS.truthy(r)
    st = EV.emit(st, ~ok, EV.EV_FAIL, actor=u, flags=EV.F_HEAL)
    return hc.ret(RS.where(ok, RS.R(RS.BOOL, 1), RS.R(RS.NOTFAIL, 0))), st, rng


for _n, _ln in (("synthesis", "18730-18755"), ("moonlight", "12253-12278")):
    if mid(_n) > 0:
        handler(ids.HOOK_HIT, meff(_n), ps=f"data/moves.ts:{_ln} {_n}.onHit")(_weather_heal)


# ================================================================== Mirror Coat / Counter (data/moves.ts:11985-12028,
# 2987-3031): beforeTurnCallback addVolatile, condition {slot, damage} filled by onDamagingHit, onRedirectTarget,
# onTry, damageCallback
COUNTERS = tuple((n, cat) for n, cat in (("counter", ids.CAT_PHYSICAL), ("mirrorcoat", ids.CAT_SPECIAL))
                 if mid(n) > 0 and has(n))
SLOT_NULL = -1                       # effectState.slot null (presence bit set): harness/project.py kind "slot"
for _n, _cat in COUNTERS:
    CO.fields(_n, ("slot", "slot"), ("damage", "int"))


_CSLOT, _CDMG = 1, 2                 # the declared slots of both conditions ("slot", "damage")


def _cmove_cond(move):
    """The counter-style condition id of a move (0: none)."""
    out = jnp.zeros(jnp.shape(move), I32)
    for n, _ in COUNTERS:
        out = jnp.where(jnp.asarray(move) == mid(n), CO.cid(n), out)
    return out


def counter_start(hc, st, dex, rng):
    """condition.onStart: effectState.slot = null, effectState.damage = 0 (one handler for both conditions: the
    condition is hc.cur_effect)."""
    c = jnp.asarray(hc.cur_effect, I32) - ids.COND_BASE + 1
    st = CO.set_vol_field(st, hc.target, c, _CSLOT, SLOT_NULL, True)
    return hc, CO.set_vol_field(st, hc.target, c, _CDMG, 0, True), rng


def counter_damaging_hit(hc, st, dex, rng):
    """condition.onDamagingHit(damage, target, source, move): a non-ally source and getCategory(move) Physical
    (Counter) / Special (Mirror Coat) -> effectState.slot = source.getSlot(), effectState.damage = 2 * damage."""
    from vgcjax.battle import move_pipeline as MP
    c = jnp.asarray(hc.cur_effect, I32) - ids.COND_BASE + 1
    cat = jnp.int32(-1)
    for n, k in COUNTERS:
        cat = jnp.where(c == CO.cid(n), k, cat)
    t, src = jnp.clip(hc.target, 0, 7), jnp.clip(hc.source, 0, 7)
    move = jnp.maximum(jnp.asarray(hc.effect, I32) - ids.MOVE_BASE + 1, 0)
    go = (hc.source >= 0) & (src // 4 != t // 4) & (dex.move_category[move] == cat)
    code = (src // 4) * 2 + jnp.maximum(MP.position(st, src), 0)
    st = CO.set_vol_field(st, t, c, _CSLOT, code, go)
    return hc, CO.set_vol_field(st, t, c, _CDMG, 2 * hc.relay_v, go), rng


def counter_try(hc, st, dex, rng):
    """onTry(source): no volatile, or effectState.slot === null -> false."""
    u = jnp.clip(hc.target, 0, 7)
    move = jnp.maximum(jnp.asarray(hc.cur_effect, I32) - ids.MOVE_BASE + 1, 0)
    c = _cmove_cond(move)
    bad = ~CO.has_volatile(st, u, c) | (CO.vol_state(st, u, c)[_CSLOT] < 0)
    return hc.ret(RS.R(RS.BOOL, 0), bad), st, rng


for _n, _cat in COUNTERS:
    assert CO.slot_of(_n, "slot") == _CSLOT and CO.slot_of(_n, "damage") == _CDMG
    _ln = "2987-3031" if _n == "counter" else "11985-12028"
    handler(ids.HOOK_START, CO.eff(_n), ps=f"data/moves.ts:{_ln} {_n}.condition.onStart")(counter_start)
    handler(ids.HOOK_DAMAGINGHIT, CO.eff(_n), ps=f"data/moves.ts:{_ln} {_n}.condition.onDamagingHit")(
        counter_damaging_hit)
    handler(ids.HOOK_TRY, meff(_n), ps=f"data/moves.ts:{_ln} {_n}.onTry")(counter_try)


def has_before_turn(move_id):
    """move.beforeTurnCallback (sim/battle-queue.ts:205-209 resolveAction: a 'beforeTurnMove' action, order 5): Counter
    and Mirror Coat are the impl stage's."""
    return _cmove_cond(move_id) > 0


def light_volatile(st, dex, rng, ref, move_id, on):
    """The volatile a light queue action adds, as one addVolatile on a traced condition: the beforeTurnMove action
    (sim/battle.ts:2783-2791; an active, unfainted user; getTarget never returns null for a scripted move in doubles:
    randomFoe() || foe.active[0]) -> beforeTurnCallback addVolatile('counter' / 'mirrorcoat'); the priorityChargeMove
    action (:2792-2797) -> priorityChargeCallback addVolatile('focuspunch' / 'chillyreception', data/moves.ts:6015-6017,
    2405-2407). Returns (st, rng)."""
    names = [n for n, _ in COUNTERS] + [n for n in ("focuspunch", "chillyreception") if mid(n) > 0 and has(n)]
    if not names:
        return st, rng
    c = jnp.zeros((), I32)
    for n in names:
        c = jnp.where(jnp.asarray(move_id) == mid(n), CO.cid(n), c)
    go = jnp.asarray(on) & (c > 0) & (jnp.asarray(ref) >= 0)
    st, rng, _ = CO.add_volatile(st, dex, rng, jnp.maximum(ref, 0), jnp.maximum(c, 1), go,
                                 cands=frozenset(CO.eff(n) for n in names))
    return st, rng


def counter_redirect(st, u, move, target, on):
    """condition.onRedirectTarget (priority -1, after Follow Me / Rage Powder / Lightning Rod) of the user's own
    volatile: move.id is the volatile's move, source is its holder, effectState.slot set -> getAtSlot(slot) (the
    occupant, possibly fainted). Returns (target, redirected)."""
    out, red = jnp.asarray(target, I32), jnp.zeros((), bool)
    u = jnp.clip(jnp.asarray(u, I32), 0, 7)
    for n, _ in COUNTERS:
        code = CO.vol_state(st, u, CO.cid(n))[_CSLOT].astype(I32)
        s, p = jnp.clip(code // 2, 0, 1), jnp.clip(code % 2, 0, 1)
        b = st.sides.active[s, p].astype(I32)
        new = jnp.where(b >= 0, s * 4 + jnp.maximum(b, 0), -1)
        go = jnp.asarray(on) & (move == mid(n)) & _vol(st, u, n) & (code >= 0)
        red = red | (go & (new != out))
        out = jnp.where(go, new, out)
    return out, red


def counter_damage(st, u, move, fixed):
    """damageCallback(pokemon): no volatile -> 0; else effectState.damage || 1."""
    for n, _ in COUNTERS:
        uc = jnp.clip(jnp.asarray(u, I32), 0, 7)
        d = CO.vol_state(st, uc, CO.cid(n))[_CDMG].astype(I32)
        v = jnp.where(_vol(st, uc, n), jnp.maximum(d, 1), 0)
        fixed = jnp.where(move == mid(n), v, fixed)
    return fixed


# ================================================================== Endure (data/moves.ts:4808-4840)
if mid("endure") > 0 and has("endure"):
    from vgcjax.effects import protect as _PR
    handler(ids.HOOK_PREPAREHIT, meff("endure"), ps="data/moves.ts:4819-4821 endure.onPrepareHit")(
        _PR._stall_prepare_hit)
    handler(ids.HOOK_HIT, meff("endure"), ps="data/moves.ts:4822-4824 endure.onHit")(_PR._stall_on_hit)

    @handler(ids.HOOK_START, CO.eff("endure"), ps="data/moves.ts:4827-4829 endure.condition.onStart")
    def endure_start(hc, st, dex, rng):
        """'-singleturn' target 'move: Endure'."""
        return hc, EV.emit(st, True, EV.EV_SINGLETURN, actor=hc.target, move=CO.eff("endure")), rng


def endure_damage(st, targets, amt, go):
    """endure.condition.onDamage (data/moves.ts:4830-4837, priority -10: after Disguise, before Sturdy / Focus Sash): a
    Move effect's damage >= target.hp -> '-activate' target 'move: Endure' (the caller emits it), return hp - 1.
    targets / amt / go [T]. Returns (amt, activated)."""
    if not (mid("endure") > 0 and has("endure")):
        return amt, jnp.zeros(jnp.shape(targets), bool)
    t = jnp.clip(jnp.asarray(targets, I32), 0, 7)
    hp = mget(st.mons.hp, t).astype(I32)
    hit = jnp.asarray(go) & CO.has_volatile(st, t, CO.cid("endure")) & (amt >= hp)
    return jnp.where(hit, hp - 1, amt), hit


def endure_activate(st, r, on):
    if not (mid("endure") > 0 and has("endure")):
        return st
    return EV.emit(st, on, EV.EV_ACTIVATE, actor=r, move=CO.eff("endure"))


# ================================================================== Electrify (data/moves.ts:4559-4590)
if mid("electrify") > 0 and has("electrify"):
    @handler(ids.HOOK_TRYHIT, meff("electrify"), ps="data/moves.ts:4570-4572 electrify.onTryHit")
    def electrify_try_hit(hc, st, dex, rng):
        """!queue.willMove(target) && target.activeTurns -> false."""
        from vgcjax.battle import queue as Q
        t = jnp.clip(hc.target, 0, 7)
        q = st.queue
        will = jnp.any(q.valid & (q.kind == S.Q_MOVE) & (Q.entry_ref(q) == t)) & ~mget(st.mons.fainted, t)
        return hc.ret(RS.R(RS.BOOL, 0), ~will & (mget(st.mons.active_turns, t) > 0)), st, rng

    @handler(ids.HOOK_START, CO.eff("electrify"), ps="data/moves.ts:4575-4577 electrify.condition.onStart")
    def electrify_start(hc, st, dex, rng):
        """'-singleturn' target 'move: Electrify'."""
        return hc, EV.emit(st, True, EV.EV_SINGLETURN, actor=hc.target, move=CO.eff("electrify")), rng


def electrify_type(st, u, move, mtype):
    """electrify.condition.onModifyType (data/moves.ts:4578-4585, priority -2: after -ate / Liquid Voice) on the
    user's runEvent('ModifyType'): a move other than Struggle becomes Electric."""
    if not (mid("electrify") > 0 and has("electrify")):
        return mtype
    return jnp.where(_vol(st, u, "electrify") & (move != ids.MOVE_STRUGGLE), ids.TYPE_ELECTRIC, mtype)


# ================================================================== Focus Punch (data/moves.ts:6004-6041)
FP = mid("focuspunch") > 0 and has("focuspunch")
if FP:
    CO.fields("focuspunch", ("lostFocus", "bool"))

    @handler(ids.HOOK_START, CO.eff("focuspunch"), ps="data/moves.ts:6026-6028 focuspunch.condition.onStart")
    def focuspunch_start(hc, st, dex, rng):
        """'-singleturn' pokemon 'move: Focus Punch'."""
        return hc, EV.emit(st, True, EV.EV_SINGLETURN, actor=hc.target, move=CO.eff("focuspunch")), rng


def focuspunch_cant(st, u, move, on):
    """beforeMoveCallback (data/moves.ts:6018-6023): volatiles.focuspunch.lostFocus -> 'cant' pokemon 'Focus Punch'
    'Focus Punch', true (runMove returns: moveThisTurnResult false, no PP, no lastMove). Returns (st, cant)."""
    if not FP:
        return st, jnp.zeros((), bool)
    uc = jnp.clip(jnp.asarray(u, I32), 0, 7)
    s = CO.vol_state(st, uc, CO.cid("focuspunch"))
    k = CO.slot_of("focuspunch", "lostFocus")
    cant = jnp.asarray(on) & (move == mid("focuspunch")) & _vol(st, uc, "focuspunch") & (s[k] != 0) & \
        (((s[5] >> k) & 1) == 1)
    return EV.emit(st, cant, EV.EV_CANT, actor=uc, move=meff("focuspunch")), cant


def focuspunch_hit(st, dex, t, move, on):
    """focuspunch.condition.onHit (data/moves.ts:6029-6033), through runEvent('Hit', target, source, move): a
    non-Status move -> effectState.lostFocus = true."""
    if not FP:
        return st
    tc = jnp.clip(jnp.asarray(t, I32), 0, 7)
    go = jnp.asarray(on) & (jnp.asarray(t) >= 0) & _vol(st, tc, "focuspunch") & \
        (dex.move_category[move] != ids.CAT_STATUS)
    return CO.set_vol_field(st, tc, CO.cid("focuspunch"), CO.slot_of("focuspunch", "lostFocus"), 1, go)


def focuspunch_flinch_block(st, ref, c):
    """focuspunch.condition.onTryAddVolatile (data/moves.ts:6034-6036): flinch -> null."""
    if not (FP and has("flinch")):
        return jnp.zeros((), bool)
    if isinstance(c, int) and c != CO.cid("flinch"):
        return jnp.zeros((), bool)
    return (jnp.asarray(c, I32) == CO.cid("flinch")) & _vol(st, ref, "focuspunch")


# ================================================================== Octolock (data/moves.ts:12957-12991)
OL = mid("octolock") > 0 and has("octolock")
OL_SRC = 1                           # raw state slot: effectState.source ref (dropped by the projection)
if OL:
    CO.fields("octolock")
    _OL = CO.cid("octolock")

    @handler(ids.HOOK_START, CO.eff("octolock"), ps="data/moves.ts:12972-12974 octolock.condition.onStart")
    def octolock_start(hc, st, dex, rng):
        """'-start' pokemon 'move: Octolock' '[of] source'; effectState.source kept (raw slot)."""
        r = hc.target
        s = CO.vol_state(st, r, _OL).at[OL_SRC].set(jnp.asarray(hc.source, I16))
        st = CO.set_vol_state(st, r, _OL, s, True)
        return hc, EV.emit(st, True, EV.EV_START, actor=r, move=CO.eff("octolock"), v1=-1), rng

    @handler(ids.HOOK_RESIDUAL, CO.eff("octolock"), ps="data/moves.ts:12975-12984 octolock.condition.onResidual")
    def octolock_residual(hc, st, dex, rng):
        """onResidualOrder 14: the source inactive / fainted / no activeTurns -> delete the volatile (no End), '-end'
        pokemon 'Octolock' '[partiallytrapped]' '[silent]'; else boost({def: -1, spd: -1}, pokemon, source, Octolock)
        (a primary boost: zero changes print; TryBoost abilities stay silent for Octolock)."""
        from vgcjax.battle import move_pipeline as MP
        r = hc.holder
        src = CO.vol_state(st, r, _OL)[OL_SRC].astype(I32)
        s = jnp.clip(src, 0, 7)
        m = st.mons
        gone = (src >= 0) & (~mget(m.is_active, s) | (mget(m.hp, s) <= 0) | (mget(m.active_turns, s) == 0))
        st = EV.emit(st, gone, EV.EV_END, actor=r, move=CO.eff("octolock"), v1=-1)
        st = CO._set_vol_bit(st, r, _OL, False, gone)
        st = CO.set_vol_state(st, r, _OL, jnp.zeros(6, I16), gone)
        st, _ = MP.boost(st, r, jnp.zeros(7, I32).at[1].set(-1).at[3].set(-1), ~gone, primary=True, source=src,
                         effect=meff("octolock"), has_secondaries=True)
        return hc, st, rng


def octolock_immune(st, move, targets):
    """onTryImmunity(target): dex.getImmunity('trapped', target) (Ghost) -> '-immune' (hitStepTryImmunity)."""
    if not OL:
        return jnp.zeros(jnp.shape(targets), bool)
    return (move == mid("octolock")) & jnp.any(mget(st.mons.types, jnp.clip(targets, 0, 7)) == ids.TYPE_GHOST, axis=-1)


def octolock_trapped(st, r):
    """octolock.condition.onTrapPokemon (data/moves.ts:12985-12987): effectState.source.isActive -> tryTrap()
    (runStatusImmunity('trapped'): Ghost types are immune)."""
    if not OL:
        return jnp.zeros((), bool)
    m = st.mons
    src = CO.vol_state(st, r, _OL)[OL_SRC].astype(I32)
    ghost = jnp.any(mget(m.types, r) == ids.TYPE_GHOST)
    return _vol(st, r, "octolock") & (src >= 0) & mget(m.is_active, jnp.clip(src, 0, 7)) & ~ghost


# ================================================================== Wish (data/moves.ts:20911-20953)
WISH = mid("wish") > 0 and has("wish")
if WISH:
    CO.fields("wish", ("hp", "half"), ("startingTurn", "int"))
    _WK = CO.cid("wish") - ids.SLOT_BASE
    _W_HP, _W_ST = CO.slot_of("wish", "hp"), CO.slot_of("wish", "startingTurn")

    def _wish_pos(st, r):
        from vgcjax.battle import move_pipeline as MP
        return jnp.clip(r, 0, 7) // 4, jnp.maximum(MP.position(st, r), 0)

    @handler(ids.HOOK_START, CO.eff("wish"), ps="data/moves.ts:20922-20928 wish.condition.onStart")
    def wish_start(hc, st, dex, rng):
        """effectState.hp = source.maxhp / 2 (stored as maxhp: projection kind 'half'), startingTurn =
        getOverflowedTurnCount() = (turn - 1) mod 256 (gen >= 8)."""
        s, p = _wish_pos(st, hc.target)
        sd = st.sides
        v = sd.slot_cond_state[s, p, _WK]
        v = v.at[_W_HP].set(mget(st.mons.maxhp, jnp.clip(hc.source, 0, 7)).astype(I16))
        v = v.at[_W_ST].set(((st.meta.turn.astype(I32) - 1) & 255).astype(I16))
        v = v.at[5].set(v[5] | (1 << _W_HP) | (1 << _W_ST))
        return hc, st.replace(sides=sd.replace(slot_cond_state=sd.slot_cond_state.at[s, p, _WK].set(v))), rng

    @handler(ids.HOOK_RESTART, CO.eff("wish"), ps="data/moves.ts:20929-20934 wish.condition.onRestart")
    def wish_restart(hc, st, dex, rng):
        """return false."""
        return hc.ret(RS.R(RS.BOOL, 0)), st, rng

    @handler(ids.HOOK_RESIDUAL, CO.eff("wish"), ps="data/moves.ts:20935-20951 wish.condition.onResidual / onEnd")
    def wish_residual(hc, st, dex, rng):
        """onResidualOrder 4: getOverflowedTurnCount() <= startingTurn -> nothing; else removeSlotCondition(
        getAtSlot(sourceSlot), 'wish') (the slot it lives in): onEnd(target) runs here, before the delete: target not
        fainted -> heal(hp, target, target) (trunc; TryHeal: Heal Block; no message from heal for 'wish'); healed ->
        '-heal' target hp '[from] move: Wish' '[wisher] <source name>' (HEAL [from] Wish)."""
        from vgcjax.battle import faint as F
        from vgcjax.effects import batch_damaging as BD
        r = jnp.clip(hc.holder, 0, 7)
        s, p = _wish_pos(st, r)
        v = st.sides.slot_cond_state[s, p, _WK].astype(I32)
        go = ((st.meta.turn.astype(I32) - 1) & 255) > v[_W_ST]
        m = st.mons
        hp, mx = mget(m.hp, r).astype(I32), mget(m.maxhp, r).astype(I32)
        amt = jnp.maximum(v[_W_HP] // 2, 1)
        heal = go & ~mget(m.fainted, r) & (hp > 0) & mget(m.is_active, r) & (hp < mx) & ~BD.heal_blocked(st, r)
        st, _ = F.heal(st, r, amt, heal)
        st = EV.emit_hp(st, heal, EV.EV_HEAL, r, move=meff("wish"), flags=EV.F_FROM)
        sd = st.sides                        # the delete (onEnd above; no End dispatch over every slot condition)
        bits = sd.slot_cond_bits[s, p]
        sd = sd.replace(slot_cond_bits=sd.slot_cond_bits.at[s, p].set(jnp.where(go, bits & ~(U32(1) << _WK), bits)),
                        slot_cond_state=sd.slot_cond_state.at[s, p, _WK].set(
                            jnp.where(go, jnp.zeros(6, I16), sd.slot_cond_state[s, p, _WK])))
        return hc, st.replace(sides=sd), rng


# ================================================================== Supercell Slam (data/moves.ts:18441-18456):
# onMoveFail is effects/batch_damaging.py crash_damage (the hasCrashDamage family, with High Jump Kick)


# ================================================================== Shed Tail (data/moves.ts:16157-16193)
SHED = mid("shedtail") > 0 and has("substitute")
if SHED:
    @handler(ids.HOOK_TRYHIT, meff("shedtail"), ps="data/moves.ts:16167-16180 shedtail.onTryHit")
    def shedtail_try_hit(hc, st, dex, rng):
        """!canSwitch(source.side) (no Commander in Reg M-C) -> '-fail' source, NOT_FAIL; a substitute -> '-fail'
        source 'move: Shed Tail', NOT_FAIL; hp <= ceil(maxhp / 2) -> '-fail' source 'move: Shed Tail' '[weak]',
        NOT_FAIL."""
        from vgcjax.battle import faint as F
        u = jnp.clip(hc.target, 0, 7)
        m = st.mons
        hp, mx = mget(m.hp, u).astype(I32), mget(m.maxhp, u).astype(I32)
        f1 = F.can_switch(st, u // 4) <= 0
        f2 = ~f1 & CO.has_volatile(st, u, CO.cid("substitute"))
        f3 = ~f1 & ~f2 & (hp <= (mx + 1) // 2)
        st = EV.emit(st, f1, EV.EV_FAIL, actor=u)
        st = EV.emit(st, f2 | f3, EV.EV_FAIL, actor=u, move=meff("shedtail"), flags=EV.F_FAILSTATUS)
        return hc.ret(RS.R(RS.NOTFAIL, 0), f1 | f2 | f3), st, rng

    @handler(ids.HOOK_HIT, meff("shedtail"), ps="data/moves.ts:16181-16183 shedtail.onHit")
    def shedtail_hit(hc, st, dex, rng):
        """directDamage(ceil(maxhp / 2)): '-damage' X hp (no [from])."""
        from vgcjax.battle import faint as F
        u = hc.target
        st, _ = F.damage(st, u, (mget(st.mons.maxhp, u).astype(I32) + 1) // 2, True)
        return hc, EV.emit_hp(st, True, EV.EV_DAMAGE, u), rng


def is_shedtail(dex, source_effect):
    """switchIn's switchCopyFlag 'shedtail': the instaswitch's sourceEffect move has selfSwitch 'shedtail' (codegen
    SELF_SWITCH 3)."""
    se = jnp.asarray(source_effect, I32)
    return (se > 0) & (dex.move_self_switch[jnp.maximum(se, 0)] == 3)


def shedtail_copy(st, r_in, r_old, on):
    """PS: sim/pokemon.ts:1267-1289 copyVolatileFrom(oldActive, 'shedtail'): no boosts; only the substitute volatile
    (a shallow clone of its state; no onCopy)."""
    if not SHED:
        return st
    c = CO.cid("substitute")
    r_in = jnp.clip(jnp.asarray(r_in, I32), 0, 7)
    r_old = jnp.clip(jnp.asarray(r_old, I32), 0, 7)
    go = jnp.asarray(on) & CO.has_volatile(st, r_old, c)
    st = CO._set_vol_bit(st, r_in, c, True, go)
    return CO.set_vol_state(st, r_in, c, CO.vol_state(st, r_old, c), go)


# ================================================================== Simple Beam / Entrainment (data/moves.ts:16478-16500,
# 4862-4890): the TryHit guards here; setAbility (End, new ability, '-ability', Start) runs in effects/batch_copy.py
# post_hit
def _af(dex, ability, bit):
    return (dex.ability_flags[jnp.maximum(jnp.asarray(ability, I32), 0)].astype(I32) & bit) != 0


if mid("simplebeam") > 0 and _ab("simple") > 0:
    @handler(ids.HOOK_TRYHIT, meff("simplebeam"), ps="data/moves.ts:16487-16491 simplebeam.onTryHit")
    def simplebeam_try_hit(hc, st, dex, rng):
        """target ability cantsuppress, Simple or Truant -> false."""
        ta = true_ability(st, hc.target)
        bad = _af(dex, ta, ids.AF_CANTSUPPRESS) | (ta == _ab("simple")) | (ta == _ab("truant"))
        return hc.ret(RS.R(RS.BOOL, 0), bad), st, rng

if mid("entrainment") > 0:
    @handler(ids.HOOK_TRYHIT, meff("entrainment"), ps="data/moves.ts:4871-4880 entrainment.onTryHit")
    def entrainment_try_hit(hc, st, dex, rng):
        """target === source (no Dynamax) -> false; same ability, target ability cantsuppress or Truant, source
        ability noentrain -> false."""
        ta, sa = true_ability(st, hc.target), true_ability(st, hc.source)
        bad = (hc.target == hc.source) | (ta == sa) | _af(dex, ta, ids.AF_CANTSUPPRESS) | (ta == _ab("truant")) | \
            _af(dex, sa, ids.AF_NOENTRAIN)
        return hc.ret(RS.R(RS.BOOL, 0), bad), st, rng


def set_ability_moves(st, dex, u, t, move, on):
    """(simple beam, entrainment) masks and the new ability for post_hit's setAbility: Simple Beam ->
    target.setAbility('simple') ('-ability' target 'Simple' [from] move: Simple Beam); Entrainment ->
    target.setAbility(source.ability, source) ('... [of] source'). TryHit already failed cantsuppress; setAbility's
    own `!this.hp` gate holds (a hit target)."""
    z = jnp.zeros((), bool)
    sb = (jnp.asarray(on) & (move == mid("simplebeam"))) if (mid("simplebeam") > 0 and _ab("simple") > 0) else z
    en = (jnp.asarray(on) & (move == mid("entrainment"))) if mid("entrainment") > 0 else z
    new = jnp.where(sb, _ab("simple"), mget(st.mons.ability, jnp.clip(u, 0, 7)).astype(I32))
    go = (sb | en) & (mget(st.mons.hp, jnp.clip(t, 0, 7)) > 0)
    return sb & go, en & go, new


def simple_double(st, refs, d):
    """simple.onChangeBoost (data/abilities.ts:4285-4297; breakable: Mold Breaker masks it): every delta x2 (no Z-Power
    in Reg M-C). refs [...], d [..., 7]."""
    if _ab("simple") <= 0:
        return d
    s = mget(st.mons.ability, jnp.clip(jnp.asarray(refs, I32), 0, 7)) == _ab("simple")
    return jnp.where(s[..., None], d * 2, d)
