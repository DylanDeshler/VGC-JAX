"""Damaging-move batch (data/moves.ts; no Champions overrides in data/mods/champions/moves.ts for any of these):
Beat Up, Psychic Noise (+ the healblock volatile), Raging Bull, High Jump Kick, Ceaseless Edge, Bug Bite, Mortal Spin,
Water Shuriken.

Fixed-data per move id, so most of them are evaluated undispatched at their PS sites in battle/move_pipeline.py:
  - Beat Up: onModifyMove (move.multihit = allies) -> beat_up_hits (hit-loop setup); basePowerCallback -> beat_up_bp
    (move_callbacks.base_power); Parental Bond excluded (move.multihit set before PrepareHit);
  - Water Shuriken: data multihit [2, 5] (the MULTIHIT_COUNT draw); its basePowerCallback only differs for
    Greninja-Ash + Battle Bond, absent from the Champions dex (data/moves.ts:20602-20620): nothing to evaluate;
  - High Jump Kick: onMoveFail -> crash_damage (useMoveInner tail, beside Steel Beam);
  - Ceaseless Edge / Mortal Spin: onAfterHit -> after_hit, onAfterSubDamage -> after_sub_damage (spreadMoveHit);
  - Raging Bull: onModifyType -> raging_bull_type; onTryHit (screen break) dispatched below;
  - Bug Bite: onHit dispatched below;
  - healblock (Psychic Noise's secondary volatile): Start / Restart / End / BeforeMove / durationCallback dispatched;
    onDisableMove -> healblock_disable (disables.disable_moves); onTryHeal -> heal_blocked at every battle.heal site
    (move_pipeline.battle_heal + the inline drain, statuses.effect_heal, the absorb abilities, Sitrus Berry's
    onTryEatItem); Pollen Puff's healblock branches (onTryMove, onTryHeal) -> pollenpuff_try_move / moves_b2.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget, mset
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler

I32 = jnp.int32


def mid(name):
    return getattr(ids, f"MOVE_{name.upper()}", -999)


def has(name):
    return name in ids.CONDITIONS


def _flag(dex, mids, f):
    w, b = f
    return ((dex.move_flags[jnp.clip(mids, 0, len(ids.MOVES) - 1), w] >> b) & 1) == 1


# ------------------------------------------------------------------ Beat Up
def _beat_up_allies(st, u):
    """move.allies = pokemon.side.pokemon.filter(ally => ally === pokemon || !ally.fainted && !ally.status)
    (data/moves.ts:1165-1168), in side.pokemon order (Sides.ps_order). Returns (refs i32[4], eligible bool[4])."""
    m = st.mons
    side = jnp.clip(u, 0, 7) // 4
    refs = side * 4 + st.sides.ps_order[side].astype(I32)
    real = mget(m.sheet_idx, refs) >= 0
    ok = real & ((refs == u) | (~mget(m.fainted, refs) & (mget(m.status, refs) == 0)))
    return refs, ok


def beat_up_hits(st, u, move, hits):
    """Beat Up onModifyMove: move.multihit = move.allies.length (data/moves.ts:1165-1168); the hit loop takes it as
    targetHits ([C] scripts.ts:433). Returns the hit count (i32)."""
    if mid("beatup") < 0:
        return hits
    _, ok = _beat_up_allies(st, u)
    return jnp.where(move == mid("beatup"), jnp.sum(ok.astype(I32)), hits)


def beat_up_bp(st, dex, u, move, bp, hit):
    """Beat Up basePowerCallback (data/moves.ts:1154-1159): move.allies.shift() per getDamage call (one per hit for
    this single-target move): the hit-th ally's set species -> 5 + floor(baseStats.atk / 10)."""
    if mid("beatup") < 0:
        return bp
    refs, ok = _beat_up_allies(st, u)
    rank = jnp.cumsum(ok.astype(I32))                                   # 1-based position among the allies
    pick = ok & (rank == jnp.asarray(hit, I32))
    r = jnp.sum(jnp.where(pick, refs, 0))
    side = jnp.clip(u, 0, 7) // 4
    sp = st.team.species[side, jnp.clip(mget(st.mons.sheet_idx, r).astype(I32), 0, 5)].astype(I32)
    out = 5 + dex.species_base_stats[sp, 1].astype(I32) // 10
    return jnp.where((move == mid("beatup")) & jnp.any(pick), out, bp)


def is_beat_up(move):
    """move.multihit set by Beat Up's onModifyMove ('-hitcount', no Parental Bond)."""
    return move == mid("beatup") if mid("beatup") > 0 else jnp.zeros(jnp.shape(move), bool)


# ------------------------------------------------------------------ Raging Bull
RAGING_BULL_TYPES = (("taurospaldeacombat", "FIGHTING"), ("taurospaldeablaze", "FIRE"), ("taurospaldeaaqua", "WATER"))


def raging_bull_type(st, u, move, mtype):
    """Raging Bull onModifyType (data/moves.ts:14644-14656): pokemon.species.name Tauros-Paldea-Combat / -Blaze /
    -Aqua -> Fighting / Fire / Water (singleEvent, useMoveInner PS sim/battle-actions.ts:494)."""
    if mid("ragingbull") < 0:
        return mtype
    sp = mget(st.mons.species, jnp.clip(u, 0, 7)).astype(I32)
    for name, ty in RAGING_BULL_TYPES:
        s = getattr(ids, f"SPECIES_{name.upper()}", -1)
        if s > 0:
            mtype = jnp.where((move == mid("ragingbull")) & (sp == s), getattr(ids, f"TYPE_{ty}"), mtype)
    return mtype


if mid("ragingbull") > 0:
    @handler(ids.HOOK_TRYHIT, int(C.eff_move(mid("ragingbull"))), ps="data/moves.ts:14639-14643")
    def ragingbull_try_hit(hc, st, dex, rng):
        """onTryHit(pokemon): the target's side loses Reflect, Light Screen, Aurora Veil (through a substitute, before
        the hit)."""
        from vgcjax.effects import sideconds as SC
        side = jnp.clip(hc.target, 0, 7) // 4
        for c in ("reflect", "lightscreen", "auroraveil"):
            if has(c):
                st, rng = SC.remove_side_condition(st, dex, rng, side, CO.cid(c), True)
        return hc, st, rng


# ------------------------------------------------------------------ High Jump Kick
def crash_damage(st, dex, rng, u, move, on):
    """High Jump Kick (data/moves.ts:8911-8913) and Supercell Slam (:18451-18453) onMoveFail: damage(
    source.baseMaxhp / 2, source, source, conditions.get(<move>)): spreadDamage / Pokemon.damage truncate -> floor(maxhp / 2), '-damage' X hp
    '[from] highjumpkick' (no Sturdy / Focus Sash: not a Move effect); then runEvent('EmergencyExit', pokemon) with the
    hp before (PS sim/battle-actions.ts:589-595). Reached when moveResult is false (miss, immunity, Protect), not
    after a no-target / TryMove failure. Returns (st, rng)."""
    if mid("highjumpkick") < 0 and mid("supercellslam") < 0:
        return st, rng
    from vgcjax.effects import ability_reactions as AR
    from vgcjax.effects import statuses as STS
    # the hasCrashDamage moves' identical onMoveFail: High Jump Kick, Supercell Slam (data/moves.ts:18451-18453)
    go = jnp.asarray(on) & (dex.move_has_crash_damage[move] == 1) & (mget(st.mons.hp, u) > 0)
    hp0 = mget(st.mons.hp, u).astype(I32)
    st, rng = STS.effect_damage(st, dex, rng, u, mget(st.mons.maxhp, u).astype(I32) // 2, C.eff_move(move), go)
    return AR.emergency_exit(st, u, hp0, go), rng


# ------------------------------------------------------------------ Ceaseless Edge / Mortal Spin
MORTAL_SPIN_SIDE = ("spikes", "toxicspikes", "stealthrock", "stickyweb", "gmaxsteelsurge")


def _mortal_spin_clear(st, dex, rng, u, on, need_hp):
    """Mortal Spin's clearing (data/moves.ts:12326-12357): removeVolatile('leechseed') -> '-end' X 'Leech Seed'
    '[from] move: Mortal Spin' '[of] X'; each of the user's hazards removed -> '-sideend' side name '[from] move:
    Mortal Spin' '[of] X'; partiallytrapped removed (its End message). `need_hp`: the onAfterSubDamage form checks
    pokemon.hp before each step (onAfterHit only through removeVolatile's own hp gate)."""
    from vgcjax.effects import sideconds as SC
    side = jnp.clip(u, 0, 7) // 4
    alive = mget(st.mons.hp, u) > 0
    gate = on & alive if need_hp else on
    if has("leechseed"):
        ls = CO.cid("leechseed")
        st, rng, rm = CO.remove_volatile(st, dex, rng, u, ls, gate)
        st = EV.emit(st, rm, EV.EV_END, actor=u, move=CO.eff("leechseed"), v1=C.eff_move(mid("mortalspin")))
    for n in MORTAL_SPIN_SIDE:
        if has(n):
            c = CO.cid(n)
            rm = gate & CO.has_side(st, side, c)
            st, rng = SC.remove_side_condition(st, dex, rng, side, c, rm)
            st = EV.emit(st, rm, EV.EV_SIDEEND, actor=side, move=CO.eff(n))
    if has("partiallytrapped"):
        pt = CO.cid("partiallytrapped")
        st, rng, _ = CO.remove_volatile(st, dex, rng, u, pt, gate & CO.has_volatile(st, u, pt))
    return st, rng


def after_hit(st, dex, rng, u, move, targets, damaged):
    """singleEvent('AfterHit', move) per damaged target ([C] scripts.ts:403-418), after DamagingHit:
    Ceaseless Edge (data/moves.ts:2229-2235): !move.hasSheerForce -> foe side addSideCondition('spikes') (source =
    event.target, the hit foe); Mortal Spin (data/moves.ts:12326-12341): !move.hasSheerForce -> clear (a second
    target's call finds nothing left). Returns (st, rng)."""
    from vgcjax.effects import move_callbacks as MC
    from vgcjax.effects import sideconds as SC
    if mid("ceaselessedge") < 0 and mid("mortalspin") < 0:
        return st, rng
    sf = MC.sheer_force(st, dex, u, move)[0]
    side = jnp.clip(u, 0, 7) // 4
    if mid("ceaselessedge") > 0 and has("spikes"):
        for j in range(targets.shape[0]):
            go = damaged[j] & (move == mid("ceaselessedge")) & ~sf
            st, rng, _ = SC.add_side_condition(st, dex, rng, 1 - side, CO.cid("spikes"), go, source=targets[j])
    if mid("mortalspin") > 0:
        st, rng = _mortal_spin_clear(st, dex, rng, u, jnp.any(damaged) & (move == mid("mortalspin")) & ~sf, False)
    return st, rng


def after_sub_damage(st, dex, rng, u, move, t, hit, sf):
    """singleEvent('AfterSubDamage', move) in substitute.onTryPrimaryHit, per substitute hit: Ceaseless Edge
    (data/moves.ts:2236-2242): !hasSheerForce && source.hp -> foe side spikes; Mortal Spin (data/moves.ts:12342-12357):
    !hasSheerForce -> clear, each step behind pokemon.hp. Returns (st, rng)."""
    from vgcjax.effects import sideconds as SC
    side = jnp.clip(u, 0, 7) // 4
    if mid("ceaselessedge") > 0 and has("spikes"):
        go = hit & (move == mid("ceaselessedge")) & ~sf & (mget(st.mons.hp, u) > 0)
        st, rng, _ = SC.add_side_condition(st, dex, rng, 1 - side, CO.cid("spikes"), go, source=t)
    if mid("mortalspin") > 0:
        st, rng = _mortal_spin_clear(st, dex, rng, u, hit & (move == mid("mortalspin")) & ~sf, True)
    return st, rng


# ------------------------------------------------------------------ Bug Bite
if mid("bugbite") > 0:
    @handler(ids.HOOK_HIT, int(C.eff_move(mid("bugbite"))), ps="data/moves.ts:1921-1931")
    def bugbite_hit(hc, st, dex, rng):
        """source.hp && item.isBerry && target.takeItem(source) -> '-enditem' target item '[from] stealeat' '[move]
        Bug Bite' '[of] source'; singleEvent('Eat', item, ..., source, source, move): the berry's onEat on the user
        (Sitrus: heal(baseMaxhp / 4) '[from] item: Sitrus Berry' (TryHeal: Heal Block); Lum: cureStatus [msg] +
        removeVolatile('confusion'); resist berries: empty); runEvent('EatItem'): no Reg M-C handler. takeItem
        (PS sim/pokemon.ts takeItem): runEvent('TakeItem') -> Unburden; item cleared, no lastItem."""
        from vgcjax.effects import abilities as AB
        from vgcjax.effects import statuses as STS
        from vgcjax.effects.items import ieff, iid
        t, u = jnp.clip(hc.target, 0, 7), jnp.clip(hc.source, 0, 7)
        m = st.mons
        it = mget(m.item, t).astype(I32)
        berry = (it > 0) & ((dex.item_flags[jnp.clip(it, 0, len(ids.ITEMS))].astype(I32) & ids.IF_ISBERRY) != 0)
        go = (mget(m.hp, u) > 0) & berry & mget(m.is_active, t)
        st = AB.unburden(st, t, go)
        m = st.mons
        m = m.replace(item=mset(m.item, t, 0, go), item_state=mset(m.item_state, t, jnp.zeros_like(mget(m.item_state, t)),
                                                                 go), item_seq=mset(m.item_seq, t, 0, go))
        st = EV.emit(st.replace(mons=m), go, EV.EV_ENDITEM, actor=t, move=ids.ITEM_BASE + it - 1)
        if iid("sitrusberry") > 0:
            st, rng = STS.effect_heal(st, dex, rng, u, mget(st.mons.maxhp, u) // 4, ieff("sitrusberry"),
                                      go & (it == iid("sitrusberry")))
        if iid("lumberry") > 0:
            lum = go & (it == iid("lumberry"))
            st = STS.cure_status(st, u, lum)
            if has("confusion"):
                st, rng, _ = CO.remove_volatile(st, dex, rng, u, CO.cid("confusion"), lum)
        return hc, st, rng


# ------------------------------------------------------------------ healblock (Psychic Noise)
def heal_blocked(st, refs):
    """healblock.onTryHeal (data/moves.ts:8327-8336) -> false for every non-Z heal (battle.heal: runEvent('TryHeal')
    before the hp gates; Pokemon.heal (Healing Wish, Regenerator) runs no event)."""
    if not has("healblock"):
        return jnp.zeros(jnp.shape(refs), bool)
    return CO.has_volatile(st, jnp.clip(jnp.asarray(refs, I32), 0, 7), CO.cid("healblock"))


def healblock_disable(st, dex, r, dis, on):
    """healblock.onDisableMove (data/moves.ts:8304-8310): every heal-flag move slot."""
    if not has("healblock"):
        return dis
    mids = mget(st.mons.move_ids, r).astype(I32)
    hb = jnp.asarray(on) & CO.has_volatile(st, r, CO.cid("healblock"))
    return jnp.where(hb & _flag(dex, mids, ids.MF_HEAL) & (mids > 0), 1, dis)


def pollenpuff_try_move(st, u, target, move, on):
    """Pollen Puff onTryMove (data/moves.ts:13574-13580): an ally target and a heal-blocked user -> attrLastMove
    ('[still]'), 'cant' source 'move: Heal Block' move; false. Returns (st, failed)."""
    if mid("pollenpuff") < 0 or not has("healblock"):
        return st, jnp.zeros((), bool)
    t = jnp.asarray(target, I32)
    ally = (t >= 0) & (jnp.clip(t, 0, 7) // 4 == jnp.clip(u, 0, 7) // 4)
    go = jnp.asarray(on) & (move == mid("pollenpuff")) & ally & heal_blocked(st, u)
    st = EV.attr_last_move(st, go, EV.F_STILL)
    st = EV.emit(st, go, EV.EV_CANT, actor=u, move=CO.eff("healblock"))
    return st, go


def pollenpuff_heal_block_msg(st, t, u, on):
    """healblock.onTryHeal's Pollen Puff branch (data/moves.ts:8329-8334): source !== target, target not at full hp
    -> attrLastMove('[still]'), 'cant' source 'move: Heal Block' (heal returns null)."""
    if not has("healblock"):
        return st
    m = st.mons
    go = jnp.asarray(on) & heal_blocked(st, t) & (t != u) & (mget(m.hp, t) != mget(m.maxhp, t))
    st = EV.attr_last_move(st, go, EV.F_STILL)
    return EV.emit(st, go, EV.EV_CANT, actor=u, move=CO.eff("healblock"))


if has("healblock"):
    @handler(CO.DURATION_CALLBACK, CO.eff("healblock"), ps="data/moves.ts:8289-8298")
    def healblock_duration(hc, st, dex, rng):
        """Psychic Noise -> 2; a Persistent source -> '-activate' + 7 (no Reg M-C holder); else 5."""
        pn = (hc.effect == ids.MOVE_BASE + mid("psychicnoise") - 1) & (mid("psychicnoise") > 0)
        pers = getattr(ids, "ABILITY_PERSISTENT", -1)
        src = jnp.clip(hc.source, 0, 7)
        p = ~pn & (pers > 0) & (hc.source >= 0) & (mget(st.mons.ability, src) == pers)
        st = EV.emit(st, p, EV.EV_ACTIVATE, actor=src, move=ids.ABILITY_BASE + pers - 1)
        return hc.ret(RS.R(RS.NUM, jnp.where(pn, 2, jnp.where(p, 7, 5)))), st, rng

    @handler(ids.HOOK_START, CO.eff("healblock"), ps="data/moves.ts:8299-8302")
    def healblock_start(hc, st, dex, rng):
        """'-start' X 'move: Heal Block' (source.moveThisTurnResult = true: the user's move succeeds anyway)."""
        return hc, EV.emit(st, True, EV.EV_START, actor=hc.target, move=CO.eff("healblock"), v1=-1), rng

    @handler(ids.HOOK_RESTART, CO.eff("healblock"), ps="data/moves.ts:8337-8344")
    def healblock_restart(hc, st, dex, rng):
        """Psychic Noise -> nothing (undefined); else '-fail' X 'move: Heal Block' (the Heal Block move: Past)."""
        pn = (hc.effect == ids.MOVE_BASE + mid("psychicnoise") - 1) & (mid("psychicnoise") > 0)
        st = EV.emit(st, ~pn, EV.EV_FAIL, actor=hc.target, move=CO.eff("healblock"), flags=EV.F_FAILSTATUS)
        return hc, st, rng

    @handler(ids.HOOK_END, CO.eff("healblock"), ps="data/moves.ts:8323-8326")
    def healblock_end(hc, st, dex, rng):
        """'-end' X 'move: Heal Block'."""
        return hc, EV.emit(st, True, EV.EV_END, actor=hc.target, move=CO.eff("healblock"), v1=-1), rng

    @handler(ids.HOOK_BEFOREMOVE, CO.eff("healblock"), ps="data/moves.ts:8311-8317")
    def healblock_before_move(hc, st, dex, rng):
        """priority 6: a heal-flag move -> 'cant' X 'move: Heal Block' move; false. (onModifyMove repeats the check
        for moves that got past BeforeMove; with no Reg M-C move caller it is unreachable.)"""
        move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
        block = _flag(dex, move, ids.MF_HEAL)
        st = EV.emit(st, block, EV.EV_CANT, actor=hc.target, move=CO.eff("healblock"))
        return hc.ret(RS.R(RS.BOOL, 0), block), st, rng
