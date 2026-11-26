"""Batch 7b: Stalwart, Heatproof, Magic Guard, Rivalry, Pickpocket, Gluttony, Shed Skin, Quick Draw, Fur Coat; Quick
Claw, Big Root, Red Card.

PS: data/abilities.ts, data/items.ts (no Champions overrides for any of these: data/mods/champions/abilities.ts and
items.ts do not mention them). MUL-hook callbacks are effects/modifiers.py mods; the others are undispatched helpers
called from their PS event sites (perf rule):
- Stalwart onModifyMove (tracksTarget): getTarget (move_pipeline.get_target, already there) and getMoveTargets'
  priorityEvent('RedirectTarget') gate (move_pipeline.get_move_targets -> `tracks_target`);
- Heatproof onSourceModifyAtk / SpA: mods; onDamage (brn -> damage / 2): statuses.effect_damage (`damage_event`);
- Magic Guard onDamage (any non-Move effect -> false): statuses.effect_damage (`damage_event`) and
  move_pipeline.apply_recoil (`magic_guard`): weather, brn / psn / tox, partiallytrapped, recoil, Steel Beam, High
  Jump Kick crash, Life Orb, Stealth Rock / Spikes, Leech Seed, Rocky Helmet, Rough Skin ('-activate'), Spiky Shield.
  Not blocked (as PS): Struggle recoil and every directDamage (Substitute, Belly Drum, Clangorous Soul), confusion
  (its effect is a Move: effectType 'Move');
- Rivalry onBasePower, Fur Coat onModifyDef: mods;
- Gluttony onStart / onDamage (abilityState.gluttony = true; no pinch berry is legal in Reg M-C, so the state is all
  that shows): abilities.switch_in_vec (`switch_in`) and batch_reactive.berserk_damage (`gluttony_damage`, every
  runEvent('Damage') site);
- Shed Skin onResidual (order 5, subOrder 3): ability_reactions.residual (`shed_skin`);
- Quick Draw / Quick Claw onFractionalPriority: battle/queue.frac_priority (`frac_priority`);
- Big Root onTryHeal: the drain (move_pipeline.spread_move_hit), Strength Sap (move_pipeline.battle_heal) and Leech
  Seed (batch_utility) heals (`big_root`);
- Pickpocket / Red Card onAfterMoveSecondary: move_pipeline.move_end after the per-target runEvent loop
  (`after_move_secondary`).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle.util import mget, mset
from vgcjax.core import fixedpoint as fp
from vgcjax.core import rng as R
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.abilities import aeff, aid, fields
from vgcjax.effects.hooks import handler
from vgcjax.effects.items import holds, ieff, iid
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32

fields("gluttony", ("gluttony", "bool"))


def _ab(st, refs, name):
    return (mget(st.mons.ability, jnp.clip(jnp.asarray(refs, I32), 0, 7)) == aid(name)) if aid(name) > 0 else \
        jnp.zeros(jnp.shape(refs), bool)


def _flag(dex, move, flag):
    w, b = flag
    return ((dex.move_flags[move, w] >> b) & 1) == 1


def _const(m):
    return jnp.asarray(m, I32)


def _is_ability_eff(e):
    return ids.ABILITY_BASE <= e < ids.ABILITY_BASE + len(ids.ABILITIES) - 1


# ------------------------------------------------------------------ MUL-hook modifiers
if aid("heatproof") > 0:
    for _hook, _ps in (("ModifyAtk", "data/abilities.ts:1835-1841 heatproof.onSourceModifyAtk"),
                       ("ModifySpA", "data/abilities.ts:1842-1848 heatproof.onSourceModifySpA")):
        @mod(_hook, aeff("heatproof"), prefix=ids.PFX_SOURCE, ps=_ps)
        def heatproof_atk(ctx, h):
            """onSourceModifyAtk (priority 6) / onSourceModifySpA (priority 5), the defender's: a Fire move ->
            chainModify(0.5)."""
            return jnp.broadcast_to(ctx.mtype == ids.TYPE_FIRE, h.shape), _const(2048), None

if aid("furcoat") > 0:
    @mod("ModifyDef", aeff("furcoat"), ps="data/abilities.ts:1579-1583 furcoat.onModifyDef")
    def furcoat_def(ctx, h):
        """onModifyDef (priority 6), the defender's: chainModify(2) (any move; getConfusionDamage runs no Modify*
        event)."""
        return jnp.ones(h.shape, bool), _const(8192), None

if aid("rivalry") > 0:
    @mod("BasePower", aeff("rivalry"), ps="data/abilities.ts:3882-3895 rivalry.onBasePower")
    def rivalry_bp(ctx, h):
        """onBasePower (priority 24), the attacker's: attacker.gender && defender.gender -> same: chainModify(1.25),
        else chainModify(0.75); a genderless side: nothing. Genders from the team sheet (0 M, 1 F, 2 N)."""
        from vgcjax.effects.batch_passive import _gender
        ga, gd = _gender(ctx.st, ctx.tgt), _gender(ctx.st, ctx.src)
        ok = (ga != 2) & (gd != 2) & (ctx.src >= 0)
        return ok, jnp.where(ga == gd, 5120, 3072).astype(I32), None


# ------------------------------------------------------------------ Stalwart (getMoveTargets' RedirectTarget gate)
def tracks_target(st, dex, u, move):
    """move.tracksTarget after ModifyMove (sim/battle-actions.ts:503 runs it before getMoveTargets :531): the move's
    data flag, or Stalwart onModifyMove (priority 1, data/abilities.ts:4504-4509): move.target !== 'scripted'.
    Gates priorityEvent('RedirectTarget') in sim/pokemon.ts:850."""
    t = dex.move_tracks_target[move] == 1
    if aid("stalwart") <= 0:
        return t
    return t | (_ab(st, u, "stalwart") & (dex.move_target[move] != ids.TARGET_SCRIPTED))


# ------------------------------------------------------------------ runEvent('Damage') (spreadDamage, non-move effects)
def magic_guard(st, refs):
    """Magic Guard onDamage (data/abilities.ts:2466-2471): effect.effectType !== 'Move' -> false. True where the
    target holds it (the caller knows the effect is not a Move)."""
    return _ab(st, refs, "magicguard")


def damage_event(st, target, amount, effect, on, source=-1):
    """The batch's runEvent('Damage', target, source, effect, damage) handlers for a non-move effect
    (sim/battle.ts:2132-2140, inside statuses.effect_damage):
      Magic Guard: any effect but confusion (a Move effect, {id: 'confused', effectType: 'Move'}) -> false; an
        Ability effect first adds '-activate' source 'ability: X' (Rough Skin);
      Heatproof: effect.id === 'brn' -> damage / 2 (spreadDamage's clampIntRange: max(floor(d / 2), 1)).
    `effect` is usually a static EffectId (python int): the message / brn branches are only traced where they can
    apply. Returns (st, amount, blocked)."""
    t = jnp.clip(jnp.asarray(target, I32), 0, 7)
    on = jnp.asarray(on)
    static = isinstance(effect, int)
    if aid("magicguard") > 0:
        not_move = (effect != CO.eff("confusion")) if static else (jnp.asarray(effect) != CO.eff("confusion"))
        blocked = on & magic_guard(st, t) & not_move
        if static and _is_ability_eff(effect):
            src = jnp.asarray(source, I32)
            st = EV.emit(st, blocked & (src >= 0), EV.EV_ACTIVATE, actor=src, move=effect)
    else:
        blocked = jnp.zeros((), bool)
    if aid("heatproof") > 0 and "brn" in ids.CONDITIONS and (not static or effect == CO.eff("brn")):
        hp_ = on & _ab(st, t, "heatproof") & (jnp.asarray(effect) == CO.eff("brn"))
        amount = jnp.where(hp_, jnp.maximum(jnp.asarray(amount, I32) // 2, 1), amount)
    return st, amount, blocked


def gluttony_damage(st, refs, on):
    """Gluttony onDamage (data/abilities.ts:1623-1625): pokemon.abilityState.gluttony = true. refs [T]."""
    if aid("gluttony") <= 0:
        return st
    from vgcjax.effects import batch_reactive as BR
    refs = jnp.asarray(refs, I32)
    hit = jnp.asarray(on) & (refs >= 0) & _ab(st, refs, "gluttony")
    return BR._set_fields_v(st, refs, "gluttony", {"gluttony": 1}, hit)


# ------------------------------------------------------------------ Gluttony onStart (abilities.switch_in_vec)
if aid("gluttony") > 0:
    handler(ids.HOOK_SWITCHIN, aeff("gluttony"), ps="data/abilities.ts:1620-1622 gluttony.onStart (batch7b.switch_in)")(
        lambda hc, st, dex, rng: (hc, st, rng))


def switch_in(st, r, ab, on):
    """Gluttony onStart (data/abilities.ts:1620-1622): pokemon.abilityState.gluttony = true (no message)."""
    if aid("gluttony") <= 0:
        return st
    from vgcjax.effects import batch_reactive as BR
    go = jnp.asarray(on) & (ab == aid("gluttony"))
    return BR._set_fields_v(st, jnp.asarray(r, I32)[None], "gluttony", {"gluttony": 1}, go[None])


# ------------------------------------------------------------------ Shed Skin (fieldEvent('Residual'))
if aid("shedskin") > 0:
    handler(ids.HOOK_RESIDUAL, aeff("shedskin"),
            ps="data/abilities.ts:4188-4197 shedskin.onResidual (batch7b.shed_skin)")(lambda hc, st, dex, rng: (hc, st, rng))


def shed_skin(st, dex, rng, r, on):
    """Shed Skin onResidual (order 5, subOrder 3, data/abilities.ts:4188-4197): pokemon.hp && pokemon.status &&
    randomChance(33, 100) -> '-activate' X 'ability: Shed Skin', cureStatus() ('-curestatus' X status '[msg]').
    ABILITY_RESIDUAL_PROC draw: actor holder, subject Shed Skin. Returns (st, rng)."""
    from vgcjax.effects import statuses as STS
    if aid("shedskin") <= 0:
        return st, rng
    m = st.mons
    ok = jnp.asarray(on) & _ab(st, r, "shedskin") & (mget(m.hp, r) > 0) & (mget(m.status, r) > 0)
    key = C.dkey(st, ids.DK_ABILITY_RESIDUAL_PROC, actor=r, target=-1, subject=aeff("shedskin"), hit=0)
    win, rng = R.chance(rng, key, 33, 100, active=ok)
    cure = ok & win
    st = EV.emit(st, cure, EV.EV_ACTIVATE, actor=r, move=aeff("shedskin"))
    return STS.cure_status(st, r, cure), rng


# ------------------------------------------------------------------ runEvent('FractionalPriority') (resolveAction)
def frac_priority(st, dex, rng, ref, move_id, on):
    """runEvent('FractionalPriority', pokemon, null, move, 0) at resolveAction (sim/battle-queue.ts:249), x10.
    Handler order (priority): Stall (0, a constant: -0.1, data/abilities.ts:4497-4498), Quick Draw (-1,
    data/abilities.ts:3736-3743: move.category !== 'Status' && randomChance(3, 10) -> '-activate' X 'ability: Quick
    Draw', 0.1), Quick Claw (-2, data/items.ts:5013-5020: not (Status and Mycelium Might); priority (the relay) <= 0
    && randomChance(1, 5) -> '-activate' X 'item: Quick Claw', 0.1). FRAC_PRIO draws: actor holder, subject the
    ability / item. Returns (st, frac10, rng)."""
    ref = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    on = jnp.asarray(on)
    ab = mget(st.mons.ability, ref)
    relay = jnp.where(ab == ids.ABILITY_STALL, -1, 0)
    status = dex.move_category[jnp.asarray(move_id, I32)] == ids.CAT_STATUS
    if aid("quickdraw") > 0:
        qd = on & (ab == aid("quickdraw")) & ~status
        key = C.dkey(st, ids.DK_FRAC_PRIO, actor=ref, target=-1, subject=aeff("quickdraw"))
        win, rng = R.chance(rng, key, 3, 10, active=qd)
        st = EV.emit(st, qd & win, EV.EV_ACTIVATE, actor=ref, move=aeff("quickdraw"))
        relay = jnp.where(qd & win, 1, relay)
    if iid("quickclaw") > 0:
        myc = status & (ab == aid("myceliummight")) if aid("myceliummight") > 0 else jnp.zeros((), bool)
        qc = on & holds(st, ref, "quickclaw") & ~myc & (relay <= 0)
        key = C.dkey(st, ids.DK_FRAC_PRIO, actor=ref, target=-1, subject=ieff("quickclaw"))
        win, rng = R.chance(rng, key, 1, 5, active=qc)
        st = EV.emit(st, qc & win, EV.EV_ACTIVATE, actor=ref, move=ieff("quickclaw"))
        relay = jnp.where(qc & win, 1, relay)
    return st, relay, rng


# ------------------------------------------------------------------ Big Root (runEvent('TryHeal'))
def big_root(st, ref, amount, on=True):
    """Big Root onTryHeal (priority 1, data/items.ts:482-495): effect.id in drain / leechseed / ingrain / aquaring /
    strengthsap -> chainModify([5324, 4096]); runEvent's final modify(relay, modifier). The caller passes `on` only
    for those effects (Heal Block's false makes the heal fail either way). Returns the heal amount."""
    amount = jnp.asarray(amount, I32)
    if iid("bigroot") <= 0:
        return amount
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    go = jnp.asarray(on) & holds(st, r, "bigroot") & mget(st.mons.is_active, r)
    return jnp.where(go, fp.modify_raw(amount, 5324), amount)


# ------------------------------------------------------------------ Pickpocket / Red Card (runEvent('AfterMoveSecondary'))
def after_move_secondary(st, dex, rng, targets, u, move, ons):
    """The batch's onAfterMoveSecondary handlers (priority 0) of the targets of runEvent('AfterMoveSecondary',
    targets, pokemon, move) ([C] scripts.ts:575 -> sim/battle-actions.ts:873-877). They follow Eject Button (priority
    2, dispatched) and the targets' Berserk (ability), and run per target in speedSort order (holder speed; ties:
    target order, a don't-care tie); for one holder Pickpocket (Ability, subOrder 7) precedes Red Card (Item, 8).
    targets i32[NT], ons bool[NT]. One fori_loop body over the targets. Returns (st, rng)."""
    if aid("pickpocket") <= 0 and iid("redcard") <= 0:
        return st, rng
    targets = jnp.asarray(targets, I32)
    ons = jnp.asarray(ons) & (targets >= 0)
    n = targets.shape[0]
    spd = mget(st.mons.speed, jnp.clip(targets, 0, 7)).astype(I32)
    perm = jnp.argsort(jnp.where(ons, -spd, 1 << 20), stable=True)

    def body(i, carry):
        st, rng = carry
        j = perm[i]
        t, on = targets[j], ons[j]
        st, rng = _pickpocket(st, dex, rng, t, u, move, on)
        st = _red_card(st, dex, t, u, move, on)
        return jax.lax.optimization_barrier((st, rng))

    return jax.lax.fori_loop(0, n, body, (st, rng))


def _pickpocket(st, dex, rng, t, u, move, on):
    """Pickpocket onAfterMoveSecondary (data/abilities.ts:3240-3259): source !== target, contact; not (target.item,
    target.switchFlag, target.forceSwitchFlag, source.switchFlag === true); source.takeItem(target) (no item:
    nothing; runEvent('TakeItem') on the source: Unburden, a Mega Stone of its species vetoes -- the source's
    handlers only while it is active); target.setItem (no hp / inactive: the item goes back, itemState cleared) ->
    Start(item), '-enditem' source item '[silent]' '[from] ability: Pickpocket' '[of] source', '-item' target item
    '[from] ability: Pickpocket' '[of] source'. The holder's handler runs only while it is active (ignoringAbility)."""
    if aid("pickpocket") <= 0:
        return st, rng
    from vgcjax.effects import abilities as AB
    from vgcjax.effects import moves_b1 as MB1
    from vgcjax.effects import moves_b2 as MB2
    m = st.mons
    tt = jnp.clip(t, 0, 7)
    u = jnp.clip(jnp.asarray(u, I32), 0, 7)
    it = mget(m.item, u).astype(I32)
    s_act = mget(m.is_active, u)
    go = jnp.asarray(on) & _ab(st, tt, "pickpocket") & mget(m.is_active, tt) & (tt != u) & \
        _flag(dex, move, ids.MF_CONTACT) & (mget(m.item, tt) == 0) & (mget(m.switch_flag, tt) == 0) & \
        ~mget(m.force_switch_flag, tt) & (mget(m.switch_flag, u) != 1) & (it > 0)
    st = AB.unburden(st, u, go & s_act)                                  # runEvent('TakeItem') on the source
    m = st.mons
    taken = go & (MB1.takeable(st, it, u) | ~s_act)
    steal = taken & (mget(m.hp, tt) > 0) & mget(m.is_active, tt)
    m = m.replace(item=mset(m.item, u, 0, steal), item_state=mset(m.item_state, u, jnp.zeros_like(mget(m.item_state, u)),
                                                                   taken),
                  item_seq=mset(m.item_seq, u, 0, taken))
    seq = st.meta.effect_seq
    m = m.replace(item=mset(m.item, tt, it.astype(m.item.dtype), steal),
                  item_state=mset(m.item_state, tt, jnp.zeros_like(mget(m.item_state, tt)), steal),
                  item_seq=mset(m.item_seq, tt, seq, steal))
    st = st.replace(mons=m, meta=st.meta.replace(effect_seq=seq + steal.astype(seq.dtype)))
    st, rng = MB2._item_start(st, dex, rng, tt, steal)                   # setItem: singleEvent('Start', item)
    st = EV.emit(st, steal, EV.EV_ENDITEM, actor=u, move=ids.ITEM_BASE + it - 1)
    st = EV.emit(st, steal, EV.EV_ITEM, actor=tt, target=u, move=ids.ITEM_BASE + it - 1, v1=aeff("pickpocket"),
                 flags=EV.F_FROM_ABILITY)
    return st, rng


def _red_card(st, dex, t, u, move, on):
    """Red Card onAfterMoveSecondary (data/items.ts:5170-5185): source !== target, source.hp, target.hp, not a
    Status move; source active, canSwitch(source.side), neither forceSwitchFlag -> target.useItem(source)
    ('-enditem' target 'Red Card' '[of] source', sim/pokemon.ts:1845-1846); runEvent('DragOut', source, target, move):
    Guard Dog -> '-activate' source 'ability: Guard Dog', null (Ingrain / Suction Cups: not in the impl pool); else
    source.forceSwitchFlag = true (the runAction tail drags a replacement in, as for Roar). The holder's handler runs
    only while it is active (ignoringItem)."""
    if iid("redcard") <= 0:
        return st
    from vgcjax.battle import faint as F
    from vgcjax.effects import items as IT
    m = st.mons
    tt = jnp.clip(t, 0, 7)
    u = jnp.clip(jnp.asarray(u, I32), 0, 7)
    go = jnp.asarray(on) & holds(st, tt, "redcard") & mget(m.is_active, tt) & (tt != u) & (mget(m.hp, u) > 0) & \
        (mget(m.hp, tt) > 0) & (dex.move_category[move] != ids.CAT_STATUS) & mget(m.is_active, u) & \
        (F.can_switch(st, u // 4) > 0) & ~mget(m.force_switch_flag, u) & ~mget(m.force_switch_flag, tt)
    st = IT.use_item(st, tt, go)
    gd = go & _ab(st, u, "guarddog")
    st = EV.emit(st, gd, EV.EV_ACTIVATE, actor=u, move=aeff("guarddog"))
    m = st.mons
    return st.replace(mons=m.replace(force_switch_flag=mset(m.force_switch_flag, u, True, go & ~gd)))
