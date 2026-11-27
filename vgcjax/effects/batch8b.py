"""Batch 8b (Reg M-C ladder blockers): Avalanche, Pain Split, After You, Misty Explosion, Fly, Gravity, Fling; the
items Iron Ball, Shell Bell, Leppa Berry (+ Feraligite / Glalitite: their Megas' Dragonize / Refrigerate join the -ate
family in effects/move_callbacks.py ATE and effects/ability_mods.py).

PS: data/moves.ts afteryou :195-218, avalanche :908-930, fly :5897-5935, fling :5736-5789, gravity :7757-7835,
mistyexplosion :12134-12151, painsplit :13125-13145; data/items.ts ironball :3062-3079, leppaberry :3370-3391,
shellbell :5674-5687, mentalherb.fling.effect :3910-3926, whiteherb.fling.effect :7685-7702; data/abilities.ts
refrigerate :3813-3832, dragonize :1036-1056. Champions overrides: data/mods/champions/abilities.ts:18-21 dragonize,
data/mods/champions/items.ts:310-313 feraligite / :386-389 glalitite (isNonstandard only); none for the moves.
Call sites outside this module (minimal hooks, each commented at the site):
- Avalanche (basePowerCallback) / Fling (move.basePower = item.fling.basePower, set at onPrepareHit while the item is
  still held): effects/move_callbacks.py base_power; attackedBy: battle/move_pipeline.py hit-loop tail (gotAttacked),
  state.attacked_by cleared by battle/switching.py clear_volatile_many and battle/step.py end_turn;
- Iron Ball: battle/common.py spe_chain (onModifySpe), battle/move_pipeline.py getDamage effectiveness
  (onEffectiveness); its grounding was already in effects/conditions.py is_grounded;
- Shell Bell: battle/move_pipeline.py AfterMoveSecondarySelf (after Magician / Life Orb);
- Leppa Berry: effects/berries.py update (onUpdate), effects/batch_reactive.py _eat_effect (onEat: Cud Chew, Fling);
- Fly: effects/locked.py (charge move + its real move volatile, Invulnerability), effects/conditions.py
  is_semi_invulnerable;
- Gravity: effects/disables.py disable_moves (onDisableMove); grounding: effects/conditions.py is_grounded;
- Fling: battle/move_pipeline.py secondaries (item.fling.status / volatileStatus), effects/berries.py update (the
  'fling' volatile's onUpdate).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset, mset_many
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler
from vgcjax.effects.items import holds, ieff, iid
from vgcjax.effects.modifiers import mod

I8, I16, I32 = jnp.int8, jnp.int16, jnp.int32


def mid(name):
    return getattr(ids, f"MOVE_{name.upper()}", -999)


def meff(name):
    """EffectId of a move (a Python int)."""
    m = mid(name)
    return ids.MOVE_BASE + m - 1 if m > 0 else -1


def has(name):
    return name in ids.CONDITIONS


def _clip(r):
    return jnp.clip(jnp.asarray(r, I32), 0, 7)


def _vol(st, r, name):
    if not has(name):
        return jnp.zeros(jnp.shape(r), bool)
    return CO.has_volatile(st, _clip(r), CO.cid(name))


def _mflag(dex, mids, flag):
    w, b = flag
    return ((dex.move_flags[jnp.clip(mids, 0, len(ids.MOVES) - 1), w] >> b) & 1) == 1


# ================================================================== attackedBy / Avalanche
def got_attacked(st, u, targets, on):
    """target.gotAttacked(move, damage, pokemon) per hit target ([C] data/mods/champions/scripts.ts:559-566,
    sim/pokemon.ts:938-949): an attackedBy entry {source, damage, thisTurn: true}. Only what Avalanche reads is kept:
    bit `u` of target.attacked_by when the entry's numeric damage is > 0 (`on` = numeric damage > 0 per target)."""
    m = st.mons
    t = _clip(targets)
    bit = (jnp.asarray(1, I32) << _clip(u)).astype(I16)
    return st.replace(mons=m.replace(attacked_by=mset_many(m.attacked_by, t, mget(m.attacked_by, t) | bit, on)))


def base_power(st, dex, u, move, targets, bp, out):
    """basePower per target slot for this batch's moves (getDamage, sim/battle-actions.ts:1662-1668); `bp` = the
    ActiveMove basePower, `out` the running value.
    - Avalanche basePowerCallback (data/moves.ts:912-921): pokemon.attackedBy.some(p => p.source === target &&
      p.damage > 0 && p.thisTurn) -> move.basePower * 2.
    - Fling onPrepareHit (data/moves.ts:5750): move.basePower = item.fling.basePower (the item stays held until the
      'fling' volatile's onUpdate after the hit)."""
    m = st.mons
    uu = _clip(u)
    t = _clip(targets)
    if mid("avalanche") > 0:
        hit = ((mget(m.attacked_by, uu).astype(I32) >> t) & 1) == 1
        out = jnp.where((move == mid("avalanche")) & hit & (jnp.asarray(targets) >= 0), jnp.asarray(bp, I32) * 2, out)
    if mid("fling") > 0:
        fbp = dex.item_fling_bp[jnp.clip(mget(m.item, uu).astype(I32), 0, len(ids.ITEMS))].astype(I32)
        out = jnp.where(move == mid("fling"), fbp, out)
    return out


# ================================================================== Misty Explosion
if mid("mistyexplosion") > 0 and has("mistyterrain"):
    @mod("BasePower", meff("mistyexplosion"), ps="data/moves.ts:12144-12149 mistyexplosion.onBasePower")
    def mistyexplosion_bp(ctx, h):
        """field.isTerrain('mistyterrain') (no TryTerrain handler in Reg M-C) and source.isGrounded() ->
        chainModify(1.5). runEvent('BasePower', source, target): the event target (ctx.tgt) is the user."""
        ok = (ctx.st.field.terrain.astype(I32) == CO.cid("mistyterrain")) & CO.is_grounded(ctx.st, ctx.tgt)
        return ok & (ctx.tgt >= 0), jnp.asarray(6144, I32), None


# ================================================================== Pain Split
def _sethp(st, r, d, on):
    """sim/pokemon.ts:1676-1690 sethp(d): no hp -> 0; d < 1 -> 1; hp = min(d, maxhp)."""
    m = st.mons
    go = jnp.asarray(on) & (mget(m.hp, r) > 0)
    v = jnp.minimum(jnp.maximum(jnp.asarray(d, I32), 1), mget(m.maxhp, r).astype(I32))
    return st.replace(mons=m.replace(hp=mset(m.hp, r, v.astype(m.hp.dtype), go)))


if mid("painsplit") > 0:
    @handler(ids.HOOK_HIT, meff("painsplit"), ps="data/moves.ts:13134-13142 painsplit.onHit")
    def painsplit_hit(hc, st, dex, rng):
        """averagehp = floor((target.hp + pokemon.hp) / 2) || 1 (getUndynamaxedHP: no Dynamax);
        target.sethp(averagehp), '-sethp' target hp '[from] move: Pain Split' '[silent]'; pokemon.sethp(averagehp),
        '-sethp' pokemon hp '[from] move: Pain Split'. Returns undefined."""
        t, u = _clip(hc.target), _clip(hc.source)
        m = st.mons
        avg = jnp.maximum((mget(m.hp, t).astype(I32) + mget(m.hp, u).astype(I32)) // 2, 1)
        st = _sethp(st, t, avg, True)
        st = EV.emit_hp(st, True, EV.EV_SETHP, t, move=meff("painsplit"), flags=EV.F_SETHP_SILENT)
        st = _sethp(st, u, avg, True)
        st = EV.emit_hp(st, True, EV.EV_SETHP, u, move=meff("painsplit"))
        return hc, st, rng


# ================================================================== After You
if mid("afteryou") > 0:
    @handler(ids.HOOK_HIT, meff("afteryou"), ps="data/moves.ts:204-213 afteryou.onHit")
    def afteryou_hit(hc, st, dex, rng):
        """activePerHalf 2; queue.willMove(target) (sim/battle-queue.ts:324-332: its 'move' action, null if fainted):
        none -> false; else queue.prioritizeAction(action) (:282-292: order 3, sourceEffect undefined, moved to the
        front: the selection key's order puts it first, as Round does) and '-activate' target 'move: After You'."""
        from vgcjax.battle import queue as Q
        t = hc.target
        q = st.queue
        cand = q.valid & (q.kind == S.Q_MOVE) & (Q.entry_ref(q) == t) & ~mget(st.mons.fainted, _clip(t))
        found = jnp.any(cand)
        i = jnp.argmin(jnp.where(cand, q.seq.astype(I32), 1 << 30))
        q = q.replace(order=q.order.at[i].set(jnp.where(found, 3, q.order[i]).astype(q.order.dtype)),
                      source_effect=q.source_effect.at[i].set(jnp.where(found, 0, q.source_effect[i]).astype(
                          q.source_effect.dtype)))
        st = EV.emit(st.replace(queue=q), found, EV.EV_ACTIVATE, actor=t, move=meff("afteryou"))
        return hc.ret(RS.R(RS.BOOL, 0), ~found), st, rng


# ================================================================== Shell Bell
def shell_bell(st, dex, rng, u, total, on):
    """onAfterMoveSecondarySelf (priority -1, data/items.ts:5680-5684): move.totalDamage and !pokemon.forceSwitchFlag
    -> this.heal(totalDamage / 8, pokemon): battle.heal truncates, a positive amount <= 1 heals 1; '-heal' X hp
    '[from] item: Shell Bell'. Returns (st, rng)."""
    if iid("shellbell") <= 0:
        return st, rng
    from vgcjax.effects import statuses as STS
    uu = _clip(u)
    total = jnp.asarray(total, I32)
    go = jnp.asarray(on) & holds(st, uu, "shellbell") & (total > 0) & ~mget(st.mons.force_switch_flag, uu)
    return STS.effect_heal(st, dex, rng, uu, jnp.maximum(total // 8, 1), ieff("shellbell"), go)


# ================================================================== Leppa Berry
def leppa_eat(st, r, on):
    """leppaberry.onEat (data/items.ts:3383-3389): the first move slot with pp 0, else the first with pp < maxpp;
    none -> nothing; pp = min(pp + (Ripen ? 20 : 10), maxpp); '-activate' X 'item: Leppa Berry' move '[consumed]'."""
    if iid("leppaberry") <= 0:
        return st
    m = st.mons
    rr = _clip(r)
    mids, pp, mx = mget(m.move_ids, rr).astype(I32), mget(m.pp, rr).astype(I32), mget(m.maxpp, rr).astype(I32)
    valid = mids > 0
    zero, low = valid & (pp == 0), valid & (pp < mx)
    k = jnp.where(jnp.any(zero), jnp.argmax(zero), jnp.argmax(low))
    go = jnp.asarray(on) & (jnp.any(zero) | jnp.any(low))
    add = jnp.where(mget(m.ability, rr) == getattr(ids, "ABILITY_RIPEN", -1), 20, 10)
    npp = pp.at[k].set(jnp.minimum(pp[k] + add, mx[k]))
    st = st.replace(mons=m.replace(pp=mset(m.pp, rr, npp.astype(m.pp.dtype), go)))
    return EV.emit(st, go, EV.EV_ACTIVATE, actor=rr, move=ieff("leppaberry"))


def leppa_go(st, r, on):
    """leppaberry.onUpdate (data/items.ts:3378-3382): hp and a move slot with pp 0 -> eatItem() (TryEatItem: Unnerve).
    The caller (berries.update) runs one eat_item for the item onUpdate berries ('-enditem' X 'Leppa Berry' '[eat]'),
    then leppa_eat."""
    if iid("leppaberry") <= 0:
        return jnp.zeros((), bool)
    from vgcjax.effects import berries as BE
    m = st.mons
    rr = _clip(r)
    mids, pp = mget(m.move_ids, rr).astype(I32), mget(m.pp, rr).astype(I32)
    return jnp.asarray(on) & holds(st, rr, "leppaberry") & jnp.any((mids > 0) & (pp == 0)) & BE.can_eat(st, rr)


# ================================================================== Iron Ball (onEffectiveness)
def iron_ball_neutral(st, targets, mtype):
    """ironball.onEffectiveness (data/items.ts:3068-3072), run by runEffectiveness for each of the target's types:
    no ingrain / smackdown volatile and no Gravity, a Ground move and a Flying-type holder -> 0 for every type, i.e.
    the total typeMod is 0 (neutral). Its airborneness negation is conditions.is_grounded. Returns bool[T]."""
    if iid("ironball") <= 0:
        return jnp.zeros(jnp.shape(targets), bool)
    t = _clip(targets)
    exempt = _vol(st, t, "ingrain") | _vol(st, t, "smackdown")
    if has("gravity"):
        exempt = exempt | CO.has_pseudo(st, CO.cid("gravity"))
    flying = jnp.any(mget(st.mons.types, t) == ids.TYPE_FLYING, axis=-1)
    return holds(st, t, "ironball") & ~exempt & flying & (jnp.asarray(mtype) == ids.TYPE_GROUND) & \
        (jnp.asarray(targets) >= 0)


# ================================================================== Gravity (pseudo-weather)
def cancel_move(st, r, on):
    """sim/battle-queue.ts:345-353 cancelMove(pokemon): remove its first 'move' action."""
    from vgcjax.battle import queue as Q
    q = st.queue
    cand = q.valid & (q.kind == S.Q_MOVE) & (Q.entry_ref(q) == r)
    i = jnp.argmin(jnp.where(cand, q.seq.astype(I32), 1 << 30))
    kill = jnp.asarray(on) & jnp.any(cand)
    return st.replace(queue=q.replace(valid=q.valid.at[i].set(jnp.where(kill, False, q.valid[i]))))


def _delete_volatile(st, r, name, on):
    """`delete pokemon.volatiles[name]` (no End)."""
    if not has(name):
        return st, jnp.zeros((), bool)
    c = CO.cid(name)
    go = jnp.asarray(on) & CO.has_volatile(st, r, c)
    st = CO._set_vol_bit(st, r, c, False, go)
    return CO.set_vol_state(st, r, c, jnp.zeros(6, I16), go), go


if has("gravity"):
    @handler(ids.HOOK_START, CO.eff("gravity"), prefix=ids.PFX_FIELD,
             ps="data/moves.ts:7776-7809 gravity.condition.onFieldStart (+ batch8b.gravity_field_start)")
    def gravity_start(hc, st, dex, rng):
        """'-fieldstart' 'move: Gravity' (Persistent: not in Reg M-C). The per-active part runs undispatched right
        after addPseudoWeather returns (gravity_field_start): a FieldStart handler is traced at every Start dispatch
        site (weather / terrain setters too), so its body is kept to the message."""
        return hc, EV.emit(st, True, EV.EV_FIELDSTART, move=CO.eff("gravity")), rng


def gravity_field_start(st, dex, rng, on):
    """The rest of gravity.condition.onFieldStart (data/moves.ts:7782-7808), at the move's addPseudoWeather site
    (nothing runs between the handler and addPseudoWeather's return): for each non-fainted active (getAllActive,
    side order): removeVolatile('bounce') || removeVolatile('fly') -> queue.cancelMove, removeVolatile('twoturnmove');
    (Sky Drop: none in Reg M-C); magnetrise / telekinesis deleted; any of these -> '-activate' X 'move: Gravity'.
    `on` = Gravity was just started. The four actives run in a fori_loop (one traced copy). Returns (st, rng)."""
    if not has("gravity"):
        return st, rng

    def body(i, carry):
        st, rng = carry
        side, slot = i // 2, i % 2
        b = st.sides.active[side, slot].astype(I32)
        r = side * 4 + jnp.maximum(b, 0)
        live = jnp.asarray(on) & (b >= 0) & ~mget(st.mons.fainted, r)
        sky = jnp.zeros((), bool)
        for n in ("bounce", "fly"):
            if has(n):
                st, rng, rem = CO.remove_volatile(st, dex, rng, r, CO.cid(n), live & ~sky)
                sky = sky | rem
        st = cancel_move(st, r, sky)
        st, rng, _ = CO.remove_volatile(st, dex, rng, r, CO.cid("twoturnmove"), sky)
        applies = sky
        for n in ("magnetrise", "telekinesis"):
            st, d = _delete_volatile(st, r, n, live)
            applies = applies | d
        st = EV.emit(st, applies, EV.EV_ACTIVATE, actor=r, move=CO.eff("gravity"))
        return st, rng

    return jax.lax.fori_loop(0, 4, body, (st, rng))


if has("gravity"):
    @handler(ids.HOOK_END, CO.eff("gravity"), prefix=ids.PFX_FIELD,
             ps="data/moves.ts:7830-7832 gravity.condition.onFieldEnd")
    def gravity_end(hc, st, dex, rng):
        """'-fieldend' 'move: Gravity' (residual: onFieldResidualOrder 27, subOrder 2, duration 5)."""
        return hc, EV.emit(st, True, EV.EV_FIELDEND, move=CO.eff("gravity")), rng

    @handler(ids.HOOK_BEFOREMOVE, CO.eff("gravity"), ps="data/moves.ts:7818-7824 gravity.condition.onBeforeMove")
    def gravity_before_move(hc, st, dex, rng):
        """onBeforeMovePriority 6: move.flags.gravity (no Z-moves) -> 'cant' pokemon 'move: Gravity' move; false.
        The twin onModifyMove (:7825-7829) only matters for a gravity move reaching useMove without runMove's
        BeforeMove (a callsMove move): none is implemented, so it is not registered."""
        move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
        block = _mflag(dex, move, ids.MF_GRAVITY)
        st = EV.emit(st, block, EV.EV_CANT, actor=hc.target, move=CO.eff("gravity"))
        return hc.ret(RS.R(RS.BOOL, 0), block), st, rng

    @mod("ModifyAccuracy", CO.eff("gravity"), ps="data/moves.ts:7810-7813 gravity.condition.onModifyAccuracy")
    def gravity_acc(ctx, h):
        """A numeric accuracy -> chainModify([6840, 4096])."""
        return ctx.extra["numeric"], jnp.asarray(6840, I32), None


def gravity_disable(st, dex, r, dis, on):
    """gravity.condition.onDisableMove (data/moves.ts:7814-7820): every move slot whose move has flags.gravity ->
    disableMove (not hidden). `dis` i8[4] -> i8[4]."""
    if not has("gravity"):
        return dis
    mids = mget(st.mons.move_ids, _clip(r)).astype(I32)
    g = jnp.asarray(on) & CO.has_pseudo(st, CO.cid("gravity"))
    return jnp.where(g & _mflag(dex, mids, ids.MF_GRAVITY) & (mids > 0), 1, dis)


# ================================================================== Fling
FL = mid("fling") > 0 and has("fling")


def _item_flag(dex, it, flag):
    return (dex.item_flags[jnp.clip(jnp.asarray(it, I32), 0, len(ids.ITEMS))].astype(I32) & flag) != 0


if FL:
    @handler(ids.HOOK_PREPAREHIT, meff("fling"), ps="data/moves.ts:5745-5774 fling.onPrepareHit")
    def fling_prepare_hit(hc, st, dex, rng):
        """source.ignoringItem(true) (Embargo / Magic Room / Klutz) -> false; singleEvent('TakeItem', item) (a usable
        Mega Stone -> false); no item.fling -> false. Else move.basePower = fling.basePower (move_callbacks
        base_power); a berry: the user's Cud Chew onEatItem (singleEvent('EatItem', ability)), move.onHit = the foe
        eats it (fling_hit); item.fling.effect: move.onHit = it (fling_hit); fling.status / volatileStatus pushed
        into move.secondaries (fling_secondary); source.addVolatile('fling')."""
        from vgcjax.effects import batch_reactive as BR
        from vgcjax.effects import moves_b1 as MB1
        u = _clip(hc.source)
        m = st.mons
        it = mget(m.item, u).astype(I32)
        ign = (mget(m.ability, u) == getattr(ids, "ABILITY_KLUTZ", -1)) | _vol(st, u, "embargo")
        if has("magicroom"):
            ign = ign | CO.has_pseudo(st, CO.cid("magicroom"))
        ok = ~ign & (it > 0) & MB1.takeable(st, it, u) & _item_flag(dex, it, ids.IF_HASFLING)
        st = BR.cud_chew_eat(st, u, it, ok & _item_flag(dex, it, ids.IF_ISBERRY))
        st = CO.add_plain_volatile(st, u, CO.cid("fling"), ok)
        return hc.ret(RS.R(RS.BOOL, 0), ~ok), st, rng


def fling_hit(st, dex, rng, t, u, move, on):
    """PS: data/moves.ts:5752-5765 (Fling's move.onHit). Called undispatched from moveHit's Hit step for
    targets[0] (Fling is single-target; one traced copy instead of a HOOK_HIT branch per target slot).
    The onHit onPrepareHit installed (the item is still held: the 'fling' volatile marks a passed PrepareHit).
    A berry: singleEvent('Eat', item) on the foe (Sitrus / Lum / Chesto / Leppa: batch_reactive._eat_effect; no
    '-enditem', no TryEatItem), runEvent('EatItem', foe) (the foe's Cud Chew). White Herb fling.effect: negative
    boosts -> 0, '-clearnegativeboost' foe '[silent]'. Mental Herb fling.effect: the first of its conditions
    present -> all removed (batch_forme.mental_herb_cure). Other items: no onHit (nothing)."""
    if not FL:
        return st, rng
    from vgcjax.effects import batch_reactive as BR
    t, u = _clip(t), _clip(u)
    m = st.mons
    it = mget(m.item, u).astype(I32)
    gate = jnp.asarray(on) & (move == mid("fling")) & _vol(st, u, "fling")
    berry = gate & _item_flag(dex, it, ids.IF_ISBERRY)
    st, rng = BR._eat_effect(st, dex, rng, t, it, berry)
    st = BR.cud_chew_eat(st, t, it, berry)
    if iid("whiteherb") > 0:
        b = mget(st.mons.boosts, t)
        wh = gate & (it == iid("whiteherb")) & jnp.any(b < 0)
        st = st.replace(mons=st.mons.replace(boosts=mset(st.mons.boosts, t, jnp.where(b < 0, 0, b).astype(
            b.dtype), wh)))
        st = EV.emit(st, wh, EV.EV_CLEARNEGATIVEBOOST, actor=t)
    if iid("mentalherb") > 0:
        from vgcjax.effects import batch_forme as BF
        st, rng = BF.mental_herb_cure(st, dex, rng, t, gate & (it == iid("mentalherb")))
    return st, rng


def fling_secondary(st, dex, u, move):
    """move.secondaries pushed by Fling's onPrepareHit (data/moves.ts:5766-5771): {status: fling.status} or
    {volatileStatus: fling.volatileStatus}, chance undefined (no draw). Added after ModifyMove, so Sheer Force never
    removes them. Returns (on, status ConditionId, volatile ConditionId)."""
    z = jnp.zeros((), bool)
    if not FL:
        return z, jnp.int32(0), jnp.int32(0)
    uu = _clip(u)
    it = jnp.clip(mget(st.mons.item, uu).astype(I32), 0, len(ids.ITEMS))
    eff = dex.item_fling_effect[it].astype(I32)
    on = (move == mid("fling")) & _vol(st, uu, "fling") & ((eff[0] > 0) | (eff[1] > 0))
    return on, eff[0], eff[1]


def fling_update(st, dex, rng, r, on):
    """fling.condition.onUpdate (data/moves.ts:5776-5784): item = getItem(); setItem('') (hp and active gate: the
    old item's End has no Reg M-C handler); lastItem = item; usedItemThisTurn; '-enditem' X item '[from] move:
    Fling'; runEvent('AfterUseItem') (Unburden); removeVolatile('fling'). Returns (st, rng)."""
    if not FL:
        return st, rng
    from vgcjax.effects import abilities as AB
    rr = _clip(r)
    c = CO.cid("fling")
    go = jnp.asarray(on) & CO.has_volatile(st, rr, c)
    m = st.mons
    it = mget(m.item, rr).astype(I32)
    can = go & (mget(m.hp, rr) > 0) & mget(m.is_active, rr)
    m = m.replace(item=mset(m.item, rr, 0, can), item_state=mset(m.item_state, rr, jnp.zeros_like(mget(
        m.item_state, rr)), can), item_seq=mset(m.item_seq, rr, 0, can), last_item=mset(m.last_item, rr, it, go),
        used_item_this_turn=mset(m.used_item_this_turn, rr, True, go))
    st = EV.emit(st.replace(mons=m), go & (it > 0), EV.EV_ENDITEM, actor=rr, move=ids.ITEM_BASE + it - 1)
    st = AB.unburden(st, rr, go)
    return CO._set_vol_bit(st, rr, c, False, go), rng          # removeVolatile: no End handler, no state
