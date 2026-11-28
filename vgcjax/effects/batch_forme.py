"""Forme / Mega batch: Fire Mane (Mega Pyroar), Protean (Mega Greninja), Berserk (Mega Drampa), Piercing Drill (Mega
Excadrill), Disguise (Mimikyu), Air Balloon, Mental Herb.

PS (Champions overrides first): data/mods/champions/abilities.ts:12-17 berserk.onDamage, :35-38 firemane, :64-67
piercingdrill (isNonstandard only), data/mods/champions/scripts.ts:56-115 formeChange, :296-305 modifyDamage
bypassProtect; data/abilities.ts berserk / disguise / firemane / piercingdrill / protean; data/items.ts airballoon /
mentalherb; data/mods/champions/moves.ts:310-337 encore (Mental Herb); sim/battle.ts:1314-1323
checkMoveBypassesProtect.

Most work here runs undispatched at its PS call site (perf rule: no dispatch inside a dispatched handler); the call
sites in the shared pipeline are one-liners into this module.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import events as EV
from vgcjax.battle.util import mget, mset, mset_many
from vgcjax.data import ids
from vgcjax.effects import abilities as AB
from vgcjax.effects import conditions as CO
from vgcjax.effects.abilities import aeff, aid
from vgcjax.effects.hooks import handler
from vgcjax.effects.items import holds, ieff, iid
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32
X15 = 6144


def _sid(name):
    return getattr(ids, f"SPECIES_{name.upper()}", -1)


def _flag(dex, move, flag):
    w, b = flag
    return ((dex.move_flags[move, w] >> b) & 1) == 1


def _has_ab(st, refs, name):
    if aid(name) <= 0:
        return jnp.zeros(jnp.shape(refs), bool)
    return mget(st.mons.ability, jnp.clip(jnp.asarray(refs, I32), 0, 7)) == aid(name)


def _field_present(st, refs, name, key):
    """abilityState[key] is set (declared slot presence bit)."""
    slot = [s for k, s, _ in AB.FIELDS[name] if k == key][0]
    s = mget(st.mons.ability_state, jnp.clip(jnp.asarray(refs, I32), 0, 7))
    return ((s[..., 3] >> slot) & 1) == 1, s[..., slot]


def _set_field_many(st, refs, name, key, value, on):
    """abilityState[key] = value for distinct refs [T] (presence bit set)."""
    slot = [s for k, s, _ in AB.FIELDS[name] if k == key][0]
    refs = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    m = st.mons
    s = mget(m.ability_state, refs)
    val = jnp.broadcast_to(jnp.asarray(value, I16), refs.shape)
    s = s.at[..., slot].set(val).at[..., 3].set(s[..., 3] | (1 << slot))
    return st.replace(mons=m.replace(ability_state=mset_many(m.ability_state, refs, s, on)))


# ================================================================== Fire Mane (Mega Pyroar)
if aid("firemane") > 0:
    for _hook in ("ModifyAtk", "ModifySpA"):
        @mod(_hook, aeff("firemane"), ps=f"data/abilities.ts:1296-1316 firemane.on{_hook} ([C] abilities.ts:35-38)")
        def _fire_mane(ctx, h):
            """onModifyAtk / onModifySpA (priority 5): move.type === 'Fire' -> chainModify(1.5). (ADR-019: PS
            behaviour; Flags.fire_mane_atk_mod is not consulted.)"""
            return jnp.broadcast_to(ctx.mtype == ids.TYPE_FIRE, h.shape), jnp.asarray(X15, I32), None


# ================================================================== Protean (Mega Greninja)
if aid("protean") > 0:
    AB.fields("protean", ("protean", "bool"))


def protean(st, dex, u, move, mtype, on):
    """Protean onPrepareHit (data/abilities.ts:3498-3513), run by runEvent('PrepareHit', pokemon, target, move) once
    the move's Try / PrepareHit singleEvents passed: unless effectState.protean, the move bounced / is a future move /
    calls another move, a typed move ('???' excluded) whose type differs from getTypes().join() -> setType(type)
    (addedType cleared), effectState.protean = true, '-start' source 'typechange' type '[from] ability: Protean'
    (START move = Protean, v2 = 1: harness/events.py prints 'typechange')."""
    if aid("protean") <= 0:
        return st
    r = jnp.clip(jnp.asarray(u, I32), 0, 7)
    m = st.mons
    present, val = _field_present(st, r, "protean", "protean")
    done = present & (val != 0)
    calls = dex.move_calls_move[move] == 1
    t = mget(m.types, r).astype(I32)
    added = mget(m.added_type, r).astype(I32)
    mt = jnp.asarray(mtype, I32)
    same = (t[0] == mt) & (t[1] <= 0) & (added <= 0)
    go = jnp.asarray(on) & _has_ab(st, r, "protean") & ~done & ~_flag(dex, move, ids.MF_FUTUREMOVE) & ~calls & \
        (mt != ids.TYPE_TYPELESS) & (mt >= 0) & ~same & (mget(m.hp, r) > 0)
    new = jnp.stack([mt, jnp.int32(-1)]).astype(m.types.dtype)
    m = m.replace(types=mset(m.types, r, new, go), added_type=mset(m.added_type, r, 0, go))
    st = _set_field_many(st.replace(mons=m), r[None], "protean", "protean", 1, go[None])
    e = aeff("protean")
    return EV.emit(st, go, EV.EV_START, actor=r, move=e, v1=e, v2=1, flags=EV.F_FROM_ABILITY)


# ================================================================== Berserk (Mega Drampa)
if aid("berserk") > 0:
    AB.fields("berserk", ("checkedBerserk", "bool"))


def berserk_on_damage(st, refs, checked, on):
    """[C] berserk.onDamage (data/mods/champions/abilities.ts:12-17): every runEvent('Damage') on the holder sets
    effectState.checkedBerserk = !(effect is a Move without multihit). refs [T], checked bool [T]."""
    if aid("berserk") <= 0:
        return st
    refs = jnp.asarray(refs, I32)
    go = jnp.asarray(on) & _has_ab(st, refs, "berserk")
    return _set_field_many(st, refs, "berserk", "checkedBerserk", jnp.asarray(checked).astype(I16), go)


def berserk_blocks_heal(st, r):
    """berserk.onTryEatItem (data/abilities.ts:421-429): a healing berry (Sitrus) -> effectState.checkedBerserk;
    undefined (never damaged) leaves the eat allowed."""
    if aid("berserk") <= 0:
        return jnp.zeros(jnp.shape(r), bool)
    present, val = _field_present(st, r, "berserk", "checkedBerserk")
    return _has_ab(st, r, "berserk") & present & (val == 0)


def move_is_multihit(st, dex, u, move):
    """ActiveMove.multihit: a data multihit or Parental Bond's onPrepareHit (move.multihit = 2)."""
    from vgcjax.battle import move_pipeline as MP
    return (dex.move_multihit[move][0] > 0) | MP.parental_bond(st, dex, u, move, st.exec.spread_hit)


# Berserk (incl. onAfterMoveSecondary): effects/batch_reactive.py (one implementation)


# ================================================================== Piercing Drill (Mega Excadrill)
def hit_protect_bypass(st, dex, u, move, block_status=True):
    """checkMoveBypassesProtect's runEvent('HitProtect', attacker, defender, move) (sim/battle.ts:1314-1323): the
    attacker's Piercing Drill onHitProtect ([C] piercingdrill / data/abilities.ts:3283-3296): a contact move ->
    moveHitData.bypassProtect = Piercing Drill, false (the protection does not apply). Only reached for a move with
    the protect flag that is not Status (or blockStatus). [C] Unseen Fist (data/mods/champions/abilities.ts:91-99,
    batch 7a) has the identical onHitProtect."""
    if aid("piercingdrill") <= 0 and aid("unseenfist") <= 0:
        return jnp.zeros((), bool)
    cat_ok = (dex.move_category[move] != ids.CAT_STATUS) | block_status
    return (_has_ab(st, u, "piercingdrill") | _has_ab(st, u, "unseenfist")) & _flag(dex, move, ids.MF_CONTACT) & _flag(dex, move, ids.MF_PROTECT) & \
        cat_ok


PROTECT_VOLS = ("protect", "spikyshield", "banefulbunker", "kingsshield", "burningbulwark", "obstruct", "silktrap")


def protect_bypassed(st, dex, u, move, targets):
    """Per target [T]: moveHitData.bypassProtect was set by Piercing Drill, i.e. a protection whose onTryHit reached
    checkMoveBypassesProtect is up on the target (a protect-family volatile; Wide Guard vs a spread target; Quick
    Guard vs positive priority) and HitProtect returned false. modifyDamage then applies x0.25 and prints
    '-ability' source 'Piercing Drill', '-zbroken' target ([C] scripts.ts:296-305)."""
    t = jnp.clip(jnp.asarray(targets, I32), 0, 7)
    if aid("piercingdrill") <= 0 and aid("unseenfist") <= 0:
        return jnp.zeros(t.shape, bool)
    from vgcjax.battle import queue as Q
    from vgcjax.effects import move_callbacks as MC
    up = jnp.zeros(t.shape, bool)
    for name in PROTECT_VOLS:
        if name in ids.CONDITIONS:
            up = up | CO.has_volatile(st, t, CO.cid(name))
    tt = MC.move_target(st, dex, u, move)
    spread = (tt == ids.TARGET_ALLADJACENT) | (tt == ids.TARGET_ALLADJACENTFOES)
    side = t // 4
    if "wideguard" in ids.CONDITIONS:
        up = up | (spread & CO.has_side(st, side, CO.cid("wideguard")))
    if "quickguard" in ids.CONDITIONS:
        up = up | ((Q.move_priority(st, dex, u, move) > 0) & CO.has_side(st, side, CO.cid("quickguard")))
    return up & hit_protect_bypass(st, dex, u, move) & (jnp.asarray(targets) >= 0) & (jnp.asarray(targets) != u)


def emit_protect_bypass(st, u, target, on):
    """'-ability' source <bypassProtect: Piercing Drill / [C] Unseen Fist> then '-zbroken' target."""
    ab = ids.ABILITY_BASE + mget(st.mons.ability, jnp.clip(jnp.asarray(u, I32), 0, 7)).astype(I32) - 1
    st = EV.emit(st, on, EV.EV_ABILITY, actor=u, move=ab)
    return EV.emit(st, on, EV.EV_ZBROKEN, actor=target)


# ================================================================== Disguise (Mimikyu)
if aid("disguise") > 0:
    AB.fields("disguise", ("busted", "bool"))


def disguise_intact(st, refs):
    """['mimikyu', 'mimikyutotem'].includes(target.species.id) with Disguise (Totem formes are not in Reg M-C)."""
    if aid("disguise") <= 0 or _sid("mimikyu") <= 0:
        return jnp.zeros(jnp.shape(refs), bool)
    r = jnp.clip(jnp.asarray(refs, I32), 0, 7)
    # ignoringAbility: a notransform ability (Disguise) of a Transformed mon is ignored (sim/pokemon.ts:883)
    return _has_ab(st, r, "disguise") & (mget(st.mons.species, r) == _sid("mimikyu")) & ~mget(st.mons.transformed, r)


def disguise_shield(st, dex, u, move, targets, immune):
    """Disguise onCriticalHit / onEffectiveness (data/abilities.ts:979-1001) per target [T]: an intact Mimikyu not
    behind a substitute the move hits (no bypasssub / infiltrates) and not immune to it (runImmunity) -> the crit is
    vetoed and the effectiveness is 0 (no '-supereffective' / '-resisted', no resist berry / Expert Belt)."""
    from vgcjax.effects import volatiles as VO
    t = jnp.asarray(targets, I32)
    return disguise_intact(st, t) & (t >= 0) & ~VO.sub_blocks(st, dex, u, move, t) & ~jnp.asarray(immune)


def disguise_damage(st, targets, go):
    """Disguise onDamage (priority 1, before Sturdy / Focus Sash; data/abilities.ts:971-978): a move's damage on an
    intact Mimikyu -> '-activate' (the caller emits it), effectState.busted = true, return 0 (the event stops).
    Returns (st, mask [T])."""
    hit = jnp.asarray(go) & disguise_intact(st, targets)
    if aid("disguise") <= 0:
        return st, hit
    return _set_field_many(st, targets, "disguise", "busted", 1, hit), hit


def disguise_update(st, dex, rng, r, on):
    """Disguise onUpdate (data/abilities.ts:1002-1008): an intact Mimikyu with effectState.busted ->
    formeChange('Mimikyu-Busted', Disguise, true) ([C] scripts.ts:56-115: setSpecies, baseSpecies, 'detailschange',
    updateMaxHp (same base HP), no '-formechange' for an Ability source, no setAbility for Disguise), then
    damage(baseMaxhp / 8, pokemon, pokemon, species) -> '-damage' X hp '[from] pokemon: Mimikyu-Busted' (DAMAGE move
    = Disguise: harness/events.py prints 'pokemonmimikyubusted'). Returns (st, rng)."""
    if aid("disguise") <= 0 or _sid("mimikyubusted") <= 0:
        return st, rng
    from vgcjax.battle import switching as SW
    from vgcjax.effects import statuses as STS
    rr = jnp.clip(jnp.asarray(r, I32), 0, 7)
    present, val = _field_present(st, rr, "disguise", "busted")
    go = jnp.asarray(on) & disguise_intact(st, rr) & present & (val != 0) & (mget(st.mons.hp, rr) > 0)
    busted = _sid("mimikyubusted")
    st = SW.set_species(st, dex, rr, busted, go)
    m = st.mons
    st = st.replace(mons=m.replace(base_species=mset(m.base_species, rr, busted, go)))
    st = EV.emit(st, go, EV.EV_DETAILSCHANGE, actor=rr, v1=busted)
    return STS.effect_damage(st, dex, rng, rr, mget(st.mons.maxhp, rr).astype(I32) // 8, aeff("disguise"), go,
                             source=rr)


def effect_damage_hook(st, target, amount, effect, on):
    """runEvent('Damage') of battle.damage for a non-move effect (statuses.effect_damage): Disguise vs the
    confusion self-hit (an ActiveMove-typed effect: '-activate', busted, 0 damage) and Berserk's checkedBerserk
    (false only for the confusion self-hit). Returns (st, amount, disguised)."""
    if aid("disguise") <= 0 and aid("berserk") <= 0:
        return st, amount, jnp.zeros((), bool)
    t = jnp.clip(jnp.asarray(target, I32), 0, 7)
    is_conf = (jnp.asarray(effect) == CO.eff("confusion")) if "confusion" in ids.CONDITIONS else jnp.zeros((), bool)
    st, dis = disguise_damage(st, t[None], (jnp.asarray(on) & is_conf)[None])
    dis = dis[0]
    st = EV.emit(st, dis, EV.EV_ACTIVATE, actor=t, move=aeff("disguise"))
    return st, jnp.where(dis, 0, amount), dis          # Berserk: effects/batch_reactive.py (one implementation)


# ================================================================== Air Balloon
def air_balloon_start(st, r, on):
    """Air Balloon onStart (data/items.ts:191-195; the SwitchIn handler, and setItem's Start): not ignoringItem, no
    Gravity -> '-item' X 'Air Balloon'."""
    if iid("airballoon") <= 0:
        return st
    rr = jnp.clip(jnp.asarray(r, I32), 0, 7)
    grav = CO.has_pseudo(st, CO.cid("gravity")) if "gravity" in ids.CONDITIONS else jnp.zeros((), bool)
    go = jnp.asarray(on) & holds(st, rr, "airballoon") & ~grav & mget(st.mons.is_active, rr)
    return EV.emit(st, go, EV.EV_ITEM, actor=rr, move=ieff("airballoon"))


def air_balloon_pop(st, r, on):
    """Air Balloon onDamagingHit / onAfterSubDamage (a Move effect) (data/items.ts:197-211): '-enditem' X 'Air
    Balloon'; item = '' (no lastItem / usedItemThisTurn); clearEffectState(itemState) (effectOrder 0);
    runEvent('AfterUseItem'): Unburden."""
    if iid("airballoon") <= 0:
        return st
    rr = jnp.clip(jnp.asarray(r, I32), 0, 7)
    go = jnp.asarray(on) & holds(st, rr, "airballoon")
    st = EV.emit(st, go, EV.EV_ENDITEM, actor=rr, move=ieff("airballoon"))
    m = st.mons
    st = st.replace(mons=m.replace(item=mset(m.item, rr, 0, go),
                                   item_state=mset(m.item_state, rr, jnp.zeros_like(mget(m.item_state, rr)), go),
                                   item_seq=mset(m.item_seq, rr, 0, go)))
    return AB.unburden(st, rr, go)


if iid("airballoon") > 0:
    # the onStart row is a fieldEvent('SwitchIn') candidate (effects/static.py lists registered handlers); its work runs
    # undispatched in the walk body (fieldevent.py -> air_balloon_start)
    handler(ids.HOOK_SWITCHIN, ieff("airballoon"), ps="data/items.ts:191-195 airballoon.onStart (air_balloon_start)")(
        lambda hc, st, dex, rng: (hc, st, rng))


# ================================================================== Mental Herb
MENTAL_HERB_CONDS = ("attract", "taunt", "encore", "torment", "disable", "healblock")


def mental_herb(st, dex, rng, r, on):
    """Mental Herb onUpdate (data/items.ts:3927-3941): the first of attract / taunt / encore / torment / disable /
    healblock present -> useItem() ('-enditem' X 'Mental Herb'), then removeVolatile of each in that order (their
    End messages; when attract was the first found, '-end' X 'move: Attract' '[from] item: Mental Herb' follows its
    removal: Cute Charm's attract).
    Returns (st, rng)."""
    if iid("mentalherb") <= 0:
        return st, rng
    from vgcjax.effects import items as IT
    rr = jnp.clip(jnp.asarray(r, I32), 0, 7)
    conds = [c for c in MENTAL_HERB_CONDS if c in ids.CONDITIONS]
    anyv = jnp.zeros((), bool)
    for c in conds:
        anyv = anyv | CO.has_volatile(st, rr, CO.cid(c))
    m = st.mons
    go = jnp.asarray(on) & holds(st, rr, "mentalherb") & anyv & (mget(m.hp, rr) > 0) & mget(m.is_active, rr)
    st = IT.use_item(st, rr, go)
    return mental_herb_cure(st, dex, rng, rr, go)


def mental_herb_cure(st, dex, rng, r, go):
    """The cure shared by Mental Herb's onUpdate (after useItem) and its fling.effect (data/items.ts:3912-3925, Fling
    onHit: no useItem): removeVolatile of attract / taunt / encore / torment / disable / healblock in that order
    (`go` = one of them is present). Returns (st, rng)."""
    rr = jnp.clip(jnp.asarray(r, I32), 0, 7)
    conds = [c for c in MENTAL_HERB_CONDS if c in ids.CONDITIONS]
    anyv = jnp.zeros((), bool)
    for c in conds:
        anyv = anyv | CO.has_volatile(st, rr, CO.cid(c))
    go = jnp.asarray(go) & anyv
    first_attract = CO.has_volatile(st, rr, CO.cid("attract")) if "attract" in conds else jnp.zeros((), bool)
    for c in conds:
        st, rng, _ = CO.remove_volatile(st, dex, rng, rr, CO.cid(c), go & CO.has_volatile(st, rr, CO.cid(c)))
        if c == "attract":           # (data/items.ts:3934-3936) after attract's own silent '-end'
            st = EV.emit(st, go & first_attract, EV.EV_END, actor=rr, move=CO.eff("attract"),
                         v1=ieff("mentalherb"))
    return st, rng


def update_ability(st, dex, rng, r, on):
    """The batch's ability onUpdate handlers (subOrder 7, before the item's): Disguise."""
    return disguise_update(st, dex, rng, r, on)


def update_item(st, dex, rng, r, on):
    """The batch's item onUpdate handlers (subOrder 8): Mental Herb."""
    return mental_herb(st, dex, rng, r, on)


# ================================================================== Mold Breaker (Mega Gyarados)
def mold_breaker_begin(st, dex, u, on):
    """Mold Breaker onModifyMove (data/abilities.ts:2690-2701): move.ignoreAbility = true. While that ActiveMove is
    battle.activeMove (until clearActiveMove in the runAction tail, sim/battle.ts:2874) and its user is active,
    runEvent skips every breakable ability handler of any other holder (sim/battle.ts:843-848 suppressingAbility,
    gen >= 8: not the user's own; singleEvent('SwitchIn') likewise, :609-613). Modelled by parking each other mon's
    breakable ability in scratch.mb_saved (it reads as none) until mold_breaker_end. Not modelled: a breakable
    ability's singleEvent('End') at a faint inside the move (PS runs it; only Flash Fire has one here)."""
    if aid("moldbreaker") <= 0:
        return st
    uu = jnp.clip(jnp.asarray(u, I32), 0, 7)
    go = jnp.asarray(on) & _has_ab(st, uu, "moldbreaker") & ~st.scratch.mb_active
    m = st.mons
    ab = m.ability.astype(I32)                                        # [2, 4]
    brk = (dex.ability_flags[ab].astype(I32) & ids.AF_BREAKABLE) != 0
    park = go & brk & (ab > 0) & (jnp.arange(8).reshape(2, 4) != uu)
    sc = st.scratch
    st = st.replace(mons=m.replace(ability=jnp.where(park, 0, m.ability).astype(m.ability.dtype)),
                    scratch=sc.replace(mb_active=sc.mb_active | go,
                                       mb_saved=jnp.where(park, ab, jnp.where(go, -1, sc.mb_saved)).astype(I16)))
    return st


def mold_breaker_end(st, on):
    """clearActiveMove: the parked abilities return."""
    if aid("moldbreaker") <= 0:
        return st
    sc = st.scratch
    go = jnp.asarray(on) & sc.mb_active
    m = st.mons
    back = go & (sc.mb_saved >= 0)
    st = st.replace(mons=m.replace(ability=jnp.where(back, sc.mb_saved, m.ability).astype(m.ability.dtype)),
                    scratch=sc.replace(mb_active=sc.mb_active & ~go,
                                       mb_saved=jnp.where(go, -1, sc.mb_saved).astype(I16)))
    return st


# ================================================================== Illusion (Events.alias; also game state: transformInto)
def _set_alias(st, ref, val, on):
    from vgcjax.battle import events as EV
    st = EV.flush_alias(st)                     # records so far keep the alias in force when they were written
    ev = st.events
    r = jnp.clip(jnp.asarray(ref, I32), 0, 7)
    hot = (jnp.arange(8) == r) & jnp.asarray(on)
    return st.replace(events=ev.replace(alias=jnp.where(hot, jnp.asarray(val, I32), ev.alias).astype(ev.alias.dtype)))


def illusion_before_switch_in(st, side, pos, b_in, on):
    """Illusion onBeforeSwitchIn (data/abilities.ts:2057-2071), run by switchIn after the party-order swap
    (sim/battle-actions.ts:184-185, pokemon.position = pos): illusion = the last unfainted side.pokemon entry with an
    index > position (the active ally included), else null. pokemon.illusion names the mon in protocol lines
    (sim/pokemon.ts:552-555 toString; the 'switch' details) and also blocks transformInto (sim/pokemon.ts:1294), so
    Events.alias is kept with EngineConfig.events=False too (only the records compile out)."""
    if aid("illusion") <= 0:
        return st
    side = jnp.asarray(side, I32)
    r = side * 4 + jnp.asarray(b_in, I32)
    go = jnp.asarray(on) & _has_ab(st, r, "illusion")
    order = st.sides.ps_order[side].astype(I32)                       # side.pokemon as brought indices
    idx = jnp.arange(4)
    cand = (idx > jnp.asarray(pos, I32)) & ~mget(st.mons.fainted, side * 4 + order)
    last = 3 - jnp.argmax(cand[::-1])
    tgt = jnp.where(jnp.any(cand), side * 4 + order[last], -1)
    return _set_alias(st, r, tgt, go)


def illusion_damaging_hit(st, t, on):
    """Illusion onDamagingHit -> singleEvent('End', Illusion) (data/abilities.ts:2072-2088): an active illusion (the
    holder is not being called back) -> illusion = null, 'replace' X details (the true species), '-end' X
    'Illusion'."""
    if aid("illusion") <= 0:
        return st
    tt = jnp.clip(jnp.asarray(t, I32), 0, 7)
    go = jnp.asarray(on) & (jnp.asarray(t) >= 0) & _has_ab(st, tt, "illusion") & \
        (st.events.alias.astype(I32)[tt] >= 0) & ~mget(st.mons.being_called_back, tt)
    st = _set_alias(st, tt, -1, go)
    st = EV.emit(st, go, EV.EV_REPLACE, actor=tt, v1=mget(st.mons.species, tt))
    return EV.emit(st, go, EV.EV_END, actor=tt, move=aeff("illusion"), v1=-1)


def illusion_clear(st, r, on):
    """faintMessages: after the 'faint' line, runEvent('Faint') (Illusion onFaint) and pokemon.illusion = null
    (sim/battle.ts:2598-2613)."""
    if aid("illusion") <= 0:
        return st
    return _set_alias(st, r, -1, jnp.asarray(on) & (jnp.asarray(r) >= 0))


ACTIVE = any(x > 0 for x in (aid("disguise"), aid("berserk"), iid("mentalherb")))
