"""Reactive abilities batch (data/abilities.ts; Champions override: Berserk.onDamage in
data/mods/champions/abilities.ts:12-17) and Chesto Berry (data/items.ts:889-905). Undispatched masked passes called
from the existing event sites:

  Synchronize      AfterSetStatus (berries.after_set_status)          data/abilities.ts:4859-4873
  Moody            Residual (ability_reactions.residual)              data/abilities.ts:2702-2737
  Magician         AfterMoveSecondarySelf (move_pipeline.move_end)    data/abilities.ts:2478-2501
  Berserk [C]      Damage / TryEatItem / AfterMoveSecondary           ch/abilities.ts:12-17, data/abilities.ts:414-440
  Electromorphosis DamagingHit order 1 (damaging_hit_first)           data/abilities.ts:1188-1196 (+ the 'charge'
                   volatile: data/moves.ts:2274-2309)
  Cud Chew         EatItem (berries.eat_item) / Residual              data/abilities.ts:742-770
  Moxie            AfterFaint (faint.resolve_faints)                  data/abilities.ts:2750-2759
  Supreme Overlord SwitchIn / End / BasePower                         data/abilities.ts:4731-4756
  Mold Breaker     SwitchIn / ModifyMove: move.ignoreAbility           data/abilities.ts:2690-2701, sim/battle.ts:366-369,
                   (suppressingAbility -> runEvent skips the handlers  :842-848 (runEvent), :609-613 (singleEvent:
                   of breakable abilities of every mon but the mover)  SwitchIn only)
  Chesto Berry     Update (berries.update) / Eat                      data/items.ts:889-905

Mold Breaker is implemented by masking: from the mover's ModifyMove until clearActiveMove (sim/battle.ts:2874) every
other active mon whose ability has flags.breakable gets ability 0 (its handlers are then not collected, exactly what
runEvent's suppression does), the ability is kept in mons.mold_saved and restored at the move's end, when the masked
mon faints (singleEvent('End') is not suppressed) or when the mover faints (suppressingAbility requires
activePokemon.isActive).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle.util import mget, mset, mset_many
from vgcjax.core import rng as R
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.abilities import FIELDS, aeff, aid, fields
from vgcjax.effects.hooks import handler
from vgcjax.effects.items import holds, ieff, iid
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32

fields("berserk", ("checkedBerserk", "bool"))
fields("cudchew", ("berry", "item"), ("counter", "int"))
fields("supremeoverlord", ("fallen", "int"))


def _slot(name, key):
    return [s for k, s, _ in FIELDS[name] if k == key][0]


def _has(name):
    return aid(name) > 0


def _ab_is(st, refs, name):
    return (mget(st.mons.ability, jnp.clip(refs, 0, 7)) == aid(name)) if _has(name) else \
        jnp.zeros(jnp.shape(refs), bool)


def _field(st, ref, name, key):
    """(present, value) of abilityState[key] of ref (scalar or vector refs)."""
    s = mget(st.mons.ability_state, jnp.clip(ref, 0, 7)).astype(I32)
    sl = _slot(name, key)
    return ((s[..., 3] >> sl) & 1) == 1, s[..., sl]


def _set_fields_v(st, refs, name, values: dict, on):
    """abilityState[key] = value for refs [T] (values: key -> scalar or [T]); presence bits set."""
    m = st.mons
    refs = jnp.asarray(refs, I32)
    s = mget(m.ability_state, jnp.clip(refs, 0, 7))                          # [T, 4]
    for key, v in values.items():
        sl = _slot(name, key)
        v = jnp.broadcast_to(jnp.asarray(v, I16), refs.shape)
        s = s.at[:, sl].set(v).at[:, 3].set(s[:, 3] | (1 << sl))
    return st.replace(mons=m.replace(ability_state=mset_many(m.ability_state, refs, s, on)))


def _clear_fields(st, ref, name, keys, on):
    """delete this.effectState[key] for the given keys (presence bits cleared, values zeroed)."""
    m = st.mons
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    s = mget(m.ability_state, r)
    for key in keys:
        sl = _slot(name, key)
        s = s.at[sl].set(0).at[3].set(s[3] & ~(1 << sl))
    return st.replace(mons=m.replace(ability_state=mset(m.ability_state, r, s, on)))


# ------------------------------------------------------------------ SwitchIn / End (abilities.switch_in_vec / end_vec)
def switch_in(st, r, ab, on):
    """Mold Breaker onStart: '-ability' X 'Mold Breaker'. Supreme Overlord onStart: side.totalFainted -> '-activate' X
    'ability: Supreme Overlord', fallen = min(totalFainted, 5), '-start' X 'fallen<n>' '[silent]' (record: START, move =
    the ability, v2 = n; harness/events.py prints fallen<n>), effectState.fallen = n."""
    on = jnp.asarray(on)
    if _has("moldbreaker"):
        st = EV.emit(st, on & (ab == aid("moldbreaker")), EV.EV_ABILITY, actor=r, move=aeff("moldbreaker"))
    if _has("supremeoverlord"):
        tf = st.sides.total_fainted[jnp.clip(r, 0, 7) // 4].astype(I32)
        so = on & (ab == aid("supremeoverlord")) & (tf > 0)
        n = jnp.minimum(tf, 5)
        st = EV.emit(st, so, EV.EV_ACTIVATE, actor=r, move=aeff("supremeoverlord"))
        st = EV.emit(st, so, EV.EV_START, actor=r, move=aeff("supremeoverlord"), v1=-1, v2=n)
        st = _set_fields_v(st, jnp.asarray(r, I32)[None], "supremeoverlord", {"fallen": n}, so[None])
    return st


def ability_end(st, refs, on):
    """Supreme Overlord onEnd: unless beingCalledBack or !effectState.fallen: '-end' X 'fallen<n>' '[silent]'."""
    if not _has("supremeoverlord"):
        return st
    refs = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    pres, n = _field(st, refs, "supremeoverlord", "fallen")
    hit = jnp.asarray(on) & _ab_is(st, refs, "supremeoverlord") & ~mget(st.mons.being_called_back, refs) & pres & (n > 0)
    for k in range(refs.shape[0]):
        st = EV.emit(st, hit[k], EV.EV_END, actor=refs[k], move=aeff("supremeoverlord"), v1=-1, v2=n[k])
    return st


# (their SwitchIn handlers are registered with abilities.SWITCH_IN)
if _has("supremeoverlord"):
    @mod("BasePower", aeff("supremeoverlord"), ps="data/abilities.ts:4744-4751 supremeoverlord.onBasePower")
    def supreme_overlord_bp(ctx, h):
        """priority 21: effectState.fallen -> chainModify([powMod[fallen], 4096]), powMod = [4096, 4506, 4915, 5325,
        5734, 6144]."""
        pres, n = _field(ctx.st, h, "supremeoverlord", "fallen")
        tab = jnp.array([4096, 4506, 4915, 5325, 5734, 6144], I32)
        return pres & (n > 0), tab[jnp.clip(n, 0, 5)], None


# ------------------------------------------------------------------ Mold Breaker
_BREAK = {}


def _breakable():
    """ability id -> flags.breakable (static numpy table)."""
    if "t" not in _BREAK:
        from vgcjax.core.dex import load_tables_np
        _BREAK["t"] = (np.asarray(load_tables_np()["ability_flags"]).astype(np.int64) & ids.AF_BREAKABLE) != 0
    return _BREAK["t"]


def mold_start(st, dex, u, move, on):
    """Mold Breaker onModifyMove: move.ignoreAbility = true (also a move's own ignoreAbility); from here until
    clearActiveMove, suppressingAbility(holder) holds for every holder but the mover (gen >= 8): mask the breakable
    abilities of the other active mons (ability -> 0, kept in mold_saved; the mover is marked -1)."""
    if not _has("moldbreaker"):
        return st
    m = st.mons
    u = jnp.asarray(u, I32)
    go = jnp.asarray(on) & ((mget(m.ability, u) == aid("moldbreaker")) | (dex.move_ignore_ability[move] == 1))
    ab = m.ability.astype(I32)                                               # [2, 4]
    refs = jnp.arange(8, dtype=I32).reshape(2, 4)
    brk = jnp.asarray(_breakable())[jnp.clip(ab, 0, _breakable().shape[0] - 1)]
    mask = go & m.is_active & (refs != u) & brk & (ab > 0) & (m.mold_saved == 0)
    saved = jnp.where(mask, ab, jnp.where(go & (refs == u), -1, m.mold_saved.astype(I32)))
    return st.replace(mons=m.replace(ability=jnp.where(mask, 0, m.ability).astype(m.ability.dtype),
                                     mold_saved=saved.astype(m.mold_saved.dtype)))


def mold_end(st, on):
    """clearActiveMove (or the mover's faint): masked abilities restored, marks cleared."""
    if not _has("moldbreaker"):
        return st
    m = st.mons
    on = jnp.asarray(on)
    back = on & (m.mold_saved > 0) & (m.ability == 0)
    return st.replace(mons=m.replace(ability=jnp.where(back, m.mold_saved, m.ability).astype(m.ability.dtype),
                                     mold_saved=jnp.where(on, 0, m.mold_saved).astype(m.mold_saved.dtype)))


def mold_unmask(st, refs, on):
    """Before code that reads or replaces abilities mid-move (Skill Swap / Worry Seed / Transform: setAbility):
    PS's suppressingAbility never changes pokemon.ability, so restore the real abilities of `refs` (marks cleared)."""
    if not _has("moldbreaker"):
        return st
    refs = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    on = jnp.broadcast_to(jnp.asarray(on), refs.shape)
    m = st.mons
    sv = mget(m.mold_saved, refs).astype(I32)
    hit = on & (sv > 0) & (mget(m.ability, refs) == 0)
    return st.replace(mons=m.replace(ability=mset_many(m.ability, refs, sv, hit),
                                     mold_saved=mset_many(m.mold_saved, refs, 0, on & (sv > 0))))


def mold_remask(st, refs, on):
    """After such a change: while a mover's suppression is live (some mon marked -1), a non-mover active ref now
    holding a breakable ability is masked again (suppressingAbility is evaluated per event in PS)."""
    if not _has("moldbreaker"):
        return st
    refs = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    on = jnp.broadcast_to(jnp.asarray(on), refs.shape)
    m = st.mons
    live = jnp.any(m.mold_saved == -1)
    ab = mget(m.ability, refs).astype(I32)
    brk = jnp.asarray(_breakable())[jnp.clip(ab, 0, _breakable().shape[0] - 1)]
    sv = mget(m.mold_saved, refs).astype(I32)
    hit = on & live & (sv == 0) & brk & (ab > 0) & mget(m.is_active, refs)
    return st.replace(mons=m.replace(ability=mset_many(m.ability, refs, 0, hit),
                                     mold_saved=mset_many(m.mold_saved, refs, ab, hit)))


def mold_faint(st, refs, do):
    """resolveFaints: a masked mon's End handler is not suppressed (singleEvent suppresses SwitchIn only): restore its
    ability first; the mover's faint (isActive = false) ends the suppression for everyone."""
    if not _has("moldbreaker"):
        return st
    refs = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    do = jnp.asarray(do)
    sv = mget(st.mons.mold_saved, refs).astype(I32)
    st = mold_end(st, jnp.any(do & (sv == -1)))
    m = st.mons
    sv = mget(m.mold_saved, refs).astype(I32)
    hit = do & (sv > 0) & (mget(m.ability, refs) == 0)
    return st.replace(mons=m.replace(ability=mset_many(m.ability, refs, sv, hit),
                                     mold_saved=mset_many(m.mold_saved, refs, 0, do & (sv > 0))))


# ------------------------------------------------------------------ Moxie (runEvent('AfterFaint'))
def moxie(st, src, length, on):
    """Moxie onSourceAfterFaint(length, target, source, effect): the last faint's effect is a Move ->
    boost({atk: length}, source) (effect Moxie: '-ability' X 'Moxie' 'boost'); runEvent skips an inactive source
    (ignoringAbility) and boost needs hp, isActive and a foe left. `src`: faintData.source when the effect is a Move."""
    if not _has("moxie"):
        return st
    from vgcjax.battle import move_pipeline as MP
    src = jnp.asarray(src, I32)
    go = jnp.asarray(on) & (src >= 0) & _ab_is(st, src, "moxie")
    d = jnp.zeros(7, I32).at[0].set(jnp.asarray(length, I32))
    st, _ = MP.boost(st, jnp.clip(src, 0, 7), d, go, source=src, ab_msg=aeff("moxie"), simple=True)
    return st


# ------------------------------------------------------------------ Berserk [C]
def berserk_damage(st, refs, on, checked):
    """[C] berserk.onDamage (ch/abilities.ts:12-17): effectState.checkedBerserk = !(effect is a Move without
    multihit). `checked`: scalar / [T] value for this damage."""
    if not _has("berserk"):
        return st
    refs = jnp.asarray(refs, I32)
    hit = jnp.asarray(on) & (refs >= 0) & _ab_is(st, refs, "berserk")
    return _set_fields_v(st, refs, "berserk", {"checkedBerserk": jnp.asarray(checked).astype(I32)}, hit)


def berserk_try_eat(st, r):
    """berserk.onTryEatItem: a healing berry (Sitrus) -> effectState.checkedBerserk (undefined: no veto)."""
    if not _has("berserk"):
        return jnp.ones(jnp.shape(r), bool)
    pres, v = _field(st, r, "berserk", "checkedBerserk")
    return ~_ab_is(st, r, "berserk") | ~pres | (v != 0)


def berserk_after_move_secondary(st, dex, t, u, move, ex, on):
    """berserk.onAfterMoveSecondary(target, source, move): checkedBerserk = true; source != target, target.hp and
    move.totalDamage: damage = move.multihit && !move.smartTarget ? totalDamage : lastAttackedBy.damage; hp <= maxhp / 2
    < hp + damage -> boost({spa: 1}, target, target) ('-ability' X 'Berserk' 'boost')."""
    if not _has("berserk"):
        return st
    from vgcjax.battle import move_pipeline as MP
    from vgcjax.battle import results as RS
    t = jnp.asarray(t, I32)
    r = jnp.clip(t, 0, 7)
    m = st.mons
    go = jnp.asarray(on) & (t >= 0) & _ab_is(st, r, "berserk") & mget(m.is_active, r)
    st = _set_fields_v(st, r[None], "berserk", {"checkedBerserk": 1}, go[None])
    hp, mx = mget(st.mons.hp, r).astype(I32), mget(st.mons.maxhp, r).astype(I32)
    from vgcjax.effects import batch_damaging as BD                   # Beat Up sets move.multihit in onModifyMove
    multi = (dex.move_multihit[move][0] > 0) | BD.is_beat_up(move) | MP.parental_bond(st, dex, u, move, ex.spread_hit)
    j = jnp.argmax(ex.targets == t)
    last = jnp.where(ex.lt[j] == RS.NUM, ex.lv[j], 0)
    dmg = jnp.where(multi & ~ex.smart, ex.total, last)
    known = (multi & ~ex.smart) | (ex.lt[j] != RS.UNDEF)                    # undefined damage: NaN, no boost
    boost = go & (t != u) & (hp > 0) & (ex.total > 0) & known & (2 * hp <= mx) & (2 * (hp + dmg) > mx)
    st, _ = MP.boost(st, r, jnp.zeros(7, I32).at[2].set(1), boost, source=r, ab_msg=aeff("berserk"), simple=True)
    return st


# ------------------------------------------------------------------ Electromorphosis + the charge volatile
def electromorphosis(st, dex, rng, t, damaged, u):
    """onDamagingHit (order 1): target.addVolatile('charge') (source = the attacker, sourceEffect = the ability)."""
    if not (_has("electromorphosis") and "charge" in ids.CONDITIONS):
        return st, rng
    r = jnp.clip(jnp.asarray(t, I32), 0, 7)
    go = jnp.asarray(damaged) & _ab_is(st, r, "electromorphosis")
    st, rng, _ = CO.add_volatile(st, dex, rng, r, CO.cid("charge"), go, source=u, source_effect=aeff("electromorphosis"))
    return st, rng


def charge_aborted(st, dex, rng, u, move, on):
    """charge.onMoveAborted (data/moves.ts:2296-2300): an Electric move (not Charge) -> removeVolatile('charge')
    (move.type before ModifyType)."""
    if "charge" not in ids.CONDITIONS:
        return st, rng
    go = jnp.asarray(on) & (dex.move_type[move] == ids.TYPE_ELECTRIC) & (move != getattr(ids, "MOVE_CHARGE", -1))
    st, rng, _ = CO.remove_volatile(st, dex, rng, u, CO.cid("charge"), go & CO.has_volatile(st, u, CO.cid("charge")))
    return st, rng


if "charge" in ids.CONDITIONS:
    def _charge_msg(hc, st):
        el = hc.effect == aeff("electromorphosis") if _has("electromorphosis") else jnp.zeros((), bool)
        return EV.emit(st, True, EV.EV_START, actor=hc.target, move=CO.eff("charge"),
                       v1=jnp.where(el, hc.effect, -1), flags=jnp.where(el, EV.F_FROM_ABILITY, 0))

    @handler(ids.HOOK_START, CO.eff("charge"), ps="data/moves.ts:2275-2281 charge.condition.onStart")
    def charge_start(hc, st, dex, rng):
        """'-start' X 'Charge' (activeMove name, '[from] ability: Electromorphosis' / Wind Power when effect is one)."""
        return hc, _charge_msg(hc, st), rng

    @handler(ids.HOOK_RESTART, CO.eff("charge"), ps="data/moves.ts:2282-2288 charge.condition.onRestart")
    def charge_restart(hc, st, dex, rng):
        """Same '-start' line as onStart."""
        return hc, _charge_msg(hc, st), rng

    @handler(ids.HOOK_END, CO.eff("charge"), ps="data/moves.ts:2306-2308 charge.condition.onEnd")
    def charge_end(hc, st, dex, rng):
        """'-end' X 'Charge' '[silent]'."""
        return hc, EV.emit(st, True, EV.EV_END, actor=hc.target, move=CO.eff("charge"), v1=-1), rng

    @handler(ids.HOOK_AFTERMOVE, CO.eff("charge"), ps="data/moves.ts:2301-2305 charge.condition.onAfterMove")
    def charge_after_move(hc, st, dex, rng):
        """move.type === 'Electric' (the active move's type) and not Charge -> removeVolatile('charge')."""
        mv = hc.effect - ids.MOVE_BASE + 1
        go = (st.exec.mtype == ids.TYPE_ELECTRIC) & (mv != getattr(ids, "MOVE_CHARGE", -1)) & \
            (mv != ids.MOVE_STRUGGLE)
        st, rng, _ = CO.remove_volatile(st, dex, rng, hc.target, CO.cid("charge"), go)
        return hc, st, rng

    @mod("BasePower", CO.eff("charge"), ps="data/moves.ts:2289-2295 charge.condition.onBasePower")
    def charge_bp(ctx, h):
        """priority 9: an Electric move -> chainModify(2)."""
        return jnp.broadcast_to(ctx.mtype == ids.TYPE_ELECTRIC, h.shape), jnp.asarray(8192, I32), None


# ------------------------------------------------------------------ Synchronize (runEvent('AfterSetStatus'))
def synchronize(st, dex, rng, msgs, j, ref):
    """onAfterSetStatus(status, target, source, effect): a source other than the holder, effect not Toxic Spikes, not
    slp / frz -> '-activate' X 'ability: Synchronize', source.trySetStatus(status, target, {status, id:
    'synchronize'}) (not a Move / Ability effect: a plain '-status' line; the effect's .status makes the status
    immunities and the same-status '-fail' print). The source's own AfterSetStatus follows: its Synchronize bounces
    back (the holder still has the status: '-fail' holder status), then its Lum Berry."""
    if not _has("synchronize"):
        return st, rng
    from vgcjax.effects import berries as BE
    from vgcjax.effects import statuses as STS
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    src = jnp.asarray(msgs["src"], I32)
    want = jnp.asarray(msgs["want"][j], I32)
    tspikes = CO.eff("toxicspikes") if "toxicspikes" in ids.CONDITIONS else -10**6
    go = msgs["ok"][j] & _ab_is(st, r, "synchronize") & (src >= 0) & (src != r) & \
        (jnp.asarray(msgs["eff"], I32) != tspikes) & (want != ids.COND_SLP) & (want != ids.COND_FRZ)
    st = EV.emit(st, go, EV.EV_ACTIVATE, actor=r, move=aeff("synchronize"))
    s = jnp.clip(src, 0, 7)
    st, rng, _, m2 = STS.try_set_status_v(st, dex, rng, s[None], want, go[None], r, -1, want)
    st = STS.emit_status_msgs(st, m2, 0, s)
    m = st.mons
    back = m2["ok"][0] & _ab_is(st, s, "synchronize") & (mget(m.hp, r) > 0) & mget(m.is_active, r) & \
        (mget(m.status, r) == want)
    st = EV.emit(st, back, EV.EV_ACTIVATE, actor=s, move=aeff("synchronize"))
    st = EV.emit(st, back, EV.EV_FAIL, actor=r, move=C.eff_cond(want), flags=EV.F_FAILSTATUS)
    return BE.lum_eat(st, dex, rng, s, m2["ok"][0])


# ------------------------------------------------------------------ Magician (runEvent('AfterMoveSecondarySelf'))
def magician(st, dex, rng, u, move, targets, kept, on):
    """onAfterMoveSecondarySelf(source, target, move): not switchFlag === true, move.hitTargets, no item, not Fling /
    Status -> speedSort(hitTargets) (ties: SPEED_TIE_ABILITY, sub = the running action's select_ord); for each other
    target: target.takeItem(source) (runEvent('TakeItem'): Unburden, Mega Stone veto; item cleared with its state);
    source.setItem fails (no hp / not active) -> the item is put back, next target; else (Start of the new item)
    '-item' source item '[from] ability: Magician' '[of] target', stop. Returns (st, rng, stole)."""
    z = jnp.zeros((), bool)
    if not _has("magician"):
        return st, rng, z
    from vgcjax.effects import abilities as AB
    from vgcjax.effects import moves_b1 as MB1
    from vgcjax.effects import moves_b2 as MB2
    m = st.mons
    u = jnp.asarray(u, I32)
    targets = jnp.asarray(targets, I32)
    valid = jnp.asarray(kept) & (targets >= 0)
    go = jnp.asarray(on) & _ab_is(st, u, "magician") & (mget(m.switch_flag, u) != 1) & (mget(m.item, u) == 0) & \
        (move != getattr(ids, "MOVE_FLING", -1)) & (dex.move_category[move] != ids.CAT_STATUS) & jnp.any(valid)
    tc = jnp.clip(targets, 0, 7)
    spd = mget(m.speed, tc).astype(I32)
    keyv = jnp.where(valid, -spd * 16 + tc, 1 << 30)
    perm = jnp.argsort(keyv)
    s_ref, s_spd, s_val = tc[perm], spd[perm], valid[perm]
    n = targets.shape[0]
    eqs = [s_val[p] & s_val[p + 1] & (s_spd[p] == s_spd[p + 1]) for p in range(n - 1)]
    # NT = 3: at most one tie group (3 tied, the first two, or the last two); members already in ref order
    e01, e12 = eqs[0], eqs[1] if n > 2 else z
    pos = jnp.arange(n)
    gmask = jnp.where(e01 & e12, pos < 3, jnp.where(e01, pos < 2, jnp.where(e12, (pos >= 1) & (pos < 3), False)))
    start = jnp.where(e01, 0, 1)
    torder, rng = R.tie_order(rng, C.tkey(st, ids.DK_SPEED_TIE_ABILITY, st.scratch.cur_select_ord, 0), s_ref, gmask,
                              active=go & jnp.any(gmask))
    final = jnp.where(gmask, torder[jnp.clip(pos - start, 0, n - 1)], pos)
    order_refs, order_val = s_ref[final], s_val[final]
    done = z
    for p in range(n):
        t = order_refs[p]
        m = st.mons
        cand = go & order_val[p] & (t != u) & ~done
        it = mget(m.item, t).astype(I32)
        has = cand & (it > 0)
        st = AB.unburden(st, t, has & mget(m.is_active, t))                 # runEvent('TakeItem') on the holder
        m = st.mons
        taken = has & MB1.takeable(st, it, t)
        can_set = (mget(m.hp, u) > 0) & mget(m.is_active, u)
        steal = taken & can_set
        zs = jnp.zeros_like(mget(m.item_state, t))
        m = m.replace(item=mset(m.item, t, 0, steal), item_state=mset(m.item_state, t, zs, taken),
                      item_seq=mset(m.item_seq, t, 0, taken))
        seq = st.meta.effect_seq
        m = m.replace(item=mset(m.item, u, it.astype(m.item.dtype), steal),
                      item_state=mset(m.item_state, u, jnp.zeros_like(mget(m.item_state, u)), steal),
                      item_seq=mset(m.item_seq, u, seq, steal))
        st = st.replace(mons=m, meta=st.meta.replace(effect_seq=seq + steal.astype(seq.dtype)))
        st, rng = MB2._item_start(st, dex, rng, u, steal)                   # setItem: singleEvent('Start', item)
        st = EV.emit(st, steal, EV.EV_ITEM, actor=u, target=t, move=ids.ITEM_BASE + it - 1, v1=aeff("magician"),
                     flags=EV.F_FROM_ABILITY)
        done = done | steal
    return st, rng, done


# ------------------------------------------------------------------ Residual: Moody / Cud Chew (order 28, subOrder 2)
for _n in ("moody", "cudchew"):
    if _has(_n):
        handler(ids.HOOK_RESIDUAL, aeff(_n), ps=f"data/abilities.ts {_n}.onResidual (batch_reactive.residual)")(
            lambda hc, st, dex, rng: (hc, st, rng))


def residual(st, dex, rng, r, on):
    """The batch's onResidual handlers for holder r (fieldEvent walk, undispatched)."""
    st, rng = _moody(st, dex, rng, r, on)
    return _cud_chew_residual(st, dex, rng, r, on)


def _moody(st, dex, rng, r, on):
    """Moody onResidual: stats (atk..spe, boosts order) below +6 -> sample one (+2); then the stats above -6 other than
    it -> sample one (-1) (ABILITY_MOODY_STAT; no draw from an empty list); boost(boost,
    pokemon, pokemon) in that key order ('-ability' X 'Moody' 'boost' before the first change)."""
    if not _has("moody"):
        return st, rng
    from vgcjax.battle import move_pipeline as MP
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    go = jnp.asarray(on) & _ab_is(st, r, "moody")
    b = mget(st.mons.boosts, r).astype(I32)[:5]
    plus = b < 6
    n1 = jnp.sum(plus.astype(I32))
    k1 = C.dkey(st, ids.DK_ABILITY_MOODY_STAT, actor=r, subject=aeff("moody"), ord=0)
    i1, rng = R.sample(rng, k1, n1, active=go & (n1 > 0))
    up = jnp.argmax(plus & ((jnp.cumsum(plus.astype(I32)) - 1) == i1))
    has_up = n1 > 0
    minus = (b > -6) & ~((jnp.arange(5) == up) & has_up)
    n2 = jnp.sum(minus.astype(I32))
    # ord: occurrence ordinal of the holder's Moody sample calls in this step (oracle/lib/instrument.js:524): the
    # raise list's sample (even a degenerate one) precedes the lower one only when that list is not empty
    k2 = C.dkey(st, ids.DK_ABILITY_MOODY_STAT, actor=r, subject=aeff("moody"), ord=jnp.where(n1 > 0, 1, 0))
    i2, rng = R.sample(rng, k2, n2, active=go & (n2 > 0))
    dn = jnp.argmax(minus & ((jnp.cumsum(minus.astype(I32)) - 1) == i2))
    has_dn = n2 > 0
    d = jnp.zeros(7, I32)
    d = jnp.where(has_dn, d.at[dn].set(-1), d)
    d = jnp.where(has_up, d.at[up].set(2), d)
    prio = jnp.arange(7) + 8
    prio = jnp.where(has_dn, prio.at[dn].set(1), prio)
    prio = jnp.where(has_up, prio.at[up].set(0), prio)
    order = jnp.argsort(prio)
    st, _ = MP.boost(st, r, d, go & (has_up | has_dn), order=order, source=r, ab_msg=aeff("moody"), simple=True)
    return st, rng


# ------------------------------------------------------------------ Cud Chew
def cud_chew_eat(st, r, item, on):
    """Cud Chew onEatItem (runEvent('EatItem') in eatItem; the item is a berry, effect not Bug Bite / Pluck):
    effectState.berry = item, counter = 2, minus 1 when the queue is empty (eaten during the residuals)."""
    if not _has("cudchew"):
        return st
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    go = jnp.asarray(on) & _ab_is(st, r, "cudchew")
    cnt = jnp.where(jnp.any(st.queue.valid), 2, 1)
    return _set_fields_v(st, r[None], "cudchew", {"berry": item, "counter": cnt}, go[None])


def _cud_chew_residual(st, dex, rng, r, on):
    """Cud Chew onResidual: a berry and hp -> --counter <= 0: '-activate' X 'ability: Cud Chew', '-enditem' X berry
    '[eat]', singleEvent('Eat', berry) (Sitrus heal / Lum / Chesto cures; resist berries: none), runEvent('EatItem')
    (Cud Chew re-arms itself), then berry and counter are deleted."""
    if not _has("cudchew"):
        return st, rng
    from vgcjax.effects import statuses as STS
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    bp, berry = _field(st, r, "cudchew", "berry")
    _, cnt = _field(st, r, "cudchew", "counter")
    go = jnp.asarray(on) & _ab_is(st, r, "cudchew") & bp & (berry > 0) & (mget(st.mons.hp, r) > 0)
    nc = cnt - 1
    st = _set_fields_v(st, r[None], "cudchew", {"counter": nc}, go[None])
    fire = go & (nc <= 0)
    st = EV.emit(st, fire, EV.EV_ACTIVATE, actor=r, move=aeff("cudchew"))
    st = EV.emit(st, fire, EV.EV_ENDITEM, actor=r, move=ids.ITEM_BASE + berry - 1, flags=EV.F_EAT)
    st, rng = _eat_effect(st, dex, rng, r, berry, fire)
    st = _clear_fields(st, r, "cudchew", ("berry", "counter"), fire)
    return st, rng


def _eat_effect(st, dex, rng, r, berry, on):
    """singleEvent('Eat', berry) for the implemented berries: Sitrus (heal(baseMaxhp / 4)), Lum (cureStatus,
    removeVolatile('confusion')), Chesto (slp -> cureStatus)."""
    from vgcjax.effects import statuses as STS
    on = jnp.asarray(on)
    if iid("sitrusberry") > 0:
        st, rng = STS.effect_heal(st, dex, rng, r, mget(st.mons.maxhp, r) // 4, ieff("sitrusberry"),
                                  on & (berry == iid("sitrusberry")))
    if iid("lumberry") > 0:
        lum = on & (berry == iid("lumberry"))
        st = STS.cure_status(st, r, lum)
        if "confusion" in ids.CONDITIONS:
            st, rng, _ = CO.remove_volatile(st, dex, rng, r, CO.cid("confusion"), lum)
    if iid("chestoberry") > 0:
        st = STS.cure_status(st, r, on & (berry == iid("chestoberry")) & (mget(st.mons.status, r) == ids.COND_SLP))
    return st, rng


# ------------------------------------------------------------------ Chesto Berry
def chesto_update(st, dex, rng, r, on):
    """Chesto Berry onUpdate: status slp -> eatItem (TryEatItem: Unnerve) -> onEat: slp -> cureStatus()
    ('-curestatus' X slp '[msg]')."""
    if iid("chestoberry") <= 0:
        return st, rng
    from vgcjax.effects import berries as BE
    from vgcjax.effects import statuses as STS
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    go = jnp.asarray(on) & holds(st, r, "chestoberry") & (mget(st.mons.status, r) == ids.COND_SLP) & BE.can_eat(st, r)
    st = BE.eat_item(st, r, go)
    st = STS.cure_status(st, r, go & (mget(st.mons.status, r) == ids.COND_SLP))
    return st, rng
