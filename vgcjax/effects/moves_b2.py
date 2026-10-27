"""Move batch B2 (data/moves.ts; Champions overrides in data/mods/champions/moves.ts where noted): Perish Song.
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler

I16, I32 = jnp.int16, jnp.int32


def mid(name):
    return getattr(ids, f"MOVE_{name.upper()}", -999)


def has(name):
    return name in ids.CONDITIONS


# ------------------------------------------------------------------ Perish Song
# '-start' X 'perishN': START with move = perishsong and v2 = N (harness/events.py prints 'perishN')
if mid("perishsong") > 0 and has("perishsong"):
    @handler(ids.HOOK_HITFIELD, int(C.eff_move(mid("perishsong"))), ps="data/moves.ts perishsong.onHitField")
    def perishsong_hit_field(hc, st, dex, rng):
        """Each active (getAllActive order): Invulnerability (no Reg M-C semi-invulnerable state) / runEvent('TryHit')
        null (Soundproof, Good as Gold: '-immune') -> counted; else no perishsong -> addVolatile, '-start' X 'perish3'
        '[silent]'. Nothing counted -> false; a volatile added -> '-fieldactivate' 'move: Perish Song'."""
        from vgcjax.effects import ability_reactions as AR
        u = hc.source
        move = jnp.int32(mid("perishsong"))
        c = CO.cid("perishsong")
        result = message = jnp.zeros((), bool)
        for k in range(4):
            b = st.sides.active[k // 2, k % 2].astype(I32)
            r = (k // 2) * 4 + jnp.maximum(b, 0)
            live = (b >= 0) & ~mget(st.mons.fainted, r)
            th = (jnp.full(1, RS.BOOL, I32), jnp.ones(1, I32))
            th, st, rng, _ = AR.try_hit(st, dex, rng, r[None], live[None], th, u, move, dex.move_type[move].astype(I32))
            blocked = live & (th[0][0] == RS.NULL)
            add = live & ~blocked & ~CO.has_volatile(st, r, c)
            st, rng, _ = CO.add_volatile(st, dex, rng, r, c, add, source=u, source_effect=C.eff_move(move))
            st = EV.emit(st, add, EV.EV_START, actor=r, move=CO.eff("perishsong"), v1=-1, v2=3)
            result = result | blocked | add
            message = message | add
        st = EV.emit(st, message, EV.EV_FIELDACTIVATE, move=C.eff_move(move))
        return hc.ret(RS.R(RS.BOOL, 0), ~result), st, rng

    @handler(ids.HOOK_RESIDUAL, CO.eff("perishsong"), ps="data/moves.ts perishsong.condition.onResidual")
    def perishsong_residual(hc, st, dex, rng):
        """onResidualOrder 24: '-start' pokemon 'perish<duration>'."""
        r = hc.target
        n = CO.vol_state(st, r, CO.cid("perishsong"))[0].astype(I32)
        return hc, EV.emit(st, True, EV.EV_START, actor=r, move=CO.eff("perishsong"), v1=-1, v2=n), rng

    @handler(ids.HOOK_END, CO.eff("perishsong"), ps="data/moves.ts perishsong.condition.onEnd")
    def perishsong_end(hc, st, dex, rng):
        """'-start' target 'perish0'; target.faint()."""
        from vgcjax.battle import faint as F
        r = hc.target
        st = EV.emit(st, True, EV.EV_START, actor=r, move=CO.eff("perishsong"), v1=-1, v2=0)
        return hc, F.faint(st, r, True), rng


# ------------------------------------------------------------------ Yawn
YAWN_SRC = 1                                 # raw yawn state slot: effectState.source (dropped by the projection)
if mid("yawn") > 0 and has("yawn"):
    @handler(ids.HOOK_TRYHIT, int(C.eff_move(mid("yawn"))), ps="data/moves.ts yawn.onTryHit")
    def yawn_try_hit(hc, st, dex, rng):
        """target.status or !target.runStatusImmunity('slp') (no Reg M-C slp immunity) -> false."""
        fail = mget(st.mons.status, hc.target) > 0
        return hc.ret(RS.R(RS.BOOL, 0), fail), st, rng

    @handler(ids.HOOK_START, CO.eff("yawn"), ps="data/moves.ts yawn.condition.onStart")
    def yawn_start(hc, st, dex, rng):
        """'-start' target 'move: Yawn' '[of] source'."""
        r, c = hc.target, CO.cid("yawn")
        s = CO.vol_state(st, r, c)
        st = CO.set_vol_state(st, r, c, s.at[YAWN_SRC].set(jnp.asarray(hc.source, I16)), True)
        return hc, EV.emit(st, True, EV.EV_START, actor=r, move=CO.eff("yawn"), v1=-1), rng

    @handler(ids.HOOK_END, CO.eff("yawn"), ps="data/moves.ts yawn.condition.onEnd")
    def yawn_end(hc, st, dex, rng):
        """'-end' target 'move: Yawn' '[silent]'; target.trySetStatus('slp', effectState.source) (sourceEffect: the
        yawn condition -> a plain '-status' X 'slp'; Electric / Misty Terrain and Sweet Veil block it)."""
        from vgcjax.effects import statuses as STS
        r = hc.target
        src = CO.vol_state(st, r, CO.cid("yawn"))[YAWN_SRC].astype(I32)
        st = EV.emit(st, True, EV.EV_END, actor=r, move=CO.eff("yawn"), v1=-1)
        st, rng, _, msgs = STS.try_set_status_v(st, dex, rng, jnp.asarray(r, I32)[None], ids.COND_SLP, True, src,
                                                CO.eff("yawn"), 0)
        return hc, STS.emit_status_msgs(st, msgs, 0, r), rng


# ------------------------------------------------------------------ Clangorous Soul (Champions: accuracy true)
if mid("clangoroussoul") > 0:
    _cs = int(C.eff_move(mid("clangoroussoul")))

    @handler(ids.HOOK_TRY, _cs, ps="data/moves.ts clangoroussoul.onTry")
    def clangoroussoul_try(hc, st, dex, rng):
        """source.hp <= source.maxhp * 33 / 100 (100 hp <= 33 maxhp) or maxhp 1 -> false."""
        m = st.mons
        u = hc.target
        hp, mx = mget(m.hp, u).astype(I32), mget(m.maxhp, u).astype(I32)
        fail = (100 * hp <= 33 * mx) | (mx == 1)
        return hc.ret(RS.R(RS.BOOL, 0), fail), st, rng

    @handler(ids.HOOK_TRYHIT, _cs, ps="data/moves.ts clangoroussoul.onTryHit")
    def clangoroussoul_try_hit(hc, st, dex, rng):
        """boost(move.boosts) on the user (+1 Atk / Def / SpA / SpD / Spe); no change -> null; else delete
        move.boosts (moveHit applies none: battle/move_pipeline.py)."""
        from vgcjax.battle import move_pipeline as MP
        mv = mid("clangoroussoul")
        st, r = MP.boost(st, hc.target, dex.move_boosts[mv].astype(I32), True, primary=True,
                         source=hc.source, effect=_cs, simple=True)              # +1 only
        return hc.ret(RS.R(RS.NULL), ~RS.truthy(r)), st, rng

    @handler(ids.HOOK_HIT, _cs, ps="data/moves.ts clangoroussoul.onHit")
    def clangoroussoul_hit(hc, st, dex, rng):
        """directDamage(pokemon.maxhp * 33 / 100): max(floor, 1); '-damage' pokemon (no [from])."""
        from vgcjax.battle import faint as F
        u = hc.target
        d = jnp.maximum(mget(st.mons.maxhp, u).astype(I32) * 33 // 100, 1)
        on = mget(st.mons.hp, u) > 0
        st, _ = F.damage(st, u, d, on)
        return hc, EV.emit_hp(st, on, EV.EV_DAMAGE, u), rng


# ------------------------------------------------------------------ Roost
# onType (priority -1) filters Flying out of getTypes() while the volatile lasts (an empty result is Normal); the
# engine keeps pokemon.types as getTypes() would return it: rewritten at Start, restored from the species at End
# (switching out resets types via setSpecies anyway). effectState.typeWas is not modelled (the volatile never
# survives to a decision point except a mid-turn switch request).
if mid("roost") > 0 and has("roost"):
    @handler(ids.HOOK_START, CO.eff("roost"), ps="data/moves.ts roost.condition.onStart")
    def roost_start(hc, st, dex, rng):
        """'-singleturn' target 'move: Roost' (Terastallized: not in Reg M-C); types without Flying."""
        from vgcjax.battle.util import mset
        r = hc.target
        m = st.mons
        t = mget(m.types, r).astype(I32)
        fly = t == ids.TYPE_FLYING
        kept = jnp.where(fly[0], jnp.stack([t[1], 0]), jnp.where(fly[1], jnp.stack([t[0], 0]), t))
        kept = jnp.where(kept[0] == 0, jnp.stack([jnp.int32(ids.TYPE_NORMAL), 0]), kept)
        st = st.replace(mons=m.replace(types=mset(m.types, r, kept.astype(m.types.dtype), jnp.any(fly))))
        return hc, EV.emit(st, True, EV.EV_SINGLETURN, actor=r, move=CO.eff("roost")), rng

    @handler(ids.HOOK_END, CO.eff("roost"), ps="data/moves.ts roost.condition (onType ends)")
    def roost_end(hc, st, dex, rng):
        """The species' types again."""
        from vgcjax.battle.util import mset
        r = hc.target
        m = st.mons
        sp = mget(m.species, r).astype(I32)
        return hc, st.replace(mons=m.replace(types=mset(m.types, r, dex.species_types[sp].astype(m.types.dtype),
                                                        True))), rng


# ------------------------------------------------------------------ Encore (Champions: changeAction, no OverrideAction)
if mid("encore") > 0 and has("encore"):
    CO.fields("encore", ("move", "move"))

    def _encore_slot(st, r, move):
        mids = mget(st.mons.move_ids, r).astype(I32)
        hit = (mids == move) & (move > 0)
        slot = jnp.argmax(hit).astype(I32)
        return jnp.any(hit), slot

    @handler(ids.HOOK_START, CO.eff("encore"), ps="data/mods/champions/moves.ts encore.condition.onStart")
    def encore_start(hc, st, dex, rng):
        """target.lastMove (none, flags.failencore, no move slot or 0 PP -> false); effectState.move; '-start' target
        'Encore'; queue.willMove(target): none -> duration++; a different move (no Mental Herb in Reg M-C) ->
        queue.changeAction(target, {choice: 'move', moveid}): cancelAction, then insertChoice -> resolveAction (no
        targetLoc: getRandomTarget, RANDOM_TARGET phase 0; priority, fractional priority, speed)."""
        from vgcjax.battle import common as C_
        from vgcjax.battle import move_pipeline as MP
        from vgcjax.battle import queue as Q
        from vgcjax.battle import state as S
        from vgcjax.effects import disables as DS
        r, c = hc.target, CO.cid("encore")
        m = st.mons
        move = mget(m.last_move, r).astype(I32)
        known, slot = _encore_slot(st, r, move)
        pp = mget(m.pp, r)[slot]
        w, b = ids.MF_FAILENCORE
        fe = ((dex.move_flags[move, w] >> b) & 1) == 1
        ok = (move > 0) & known & (pp > 0) & ~fe
        st = CO.set_vol_field(st, r, c, CO.slot_of("encore", "move"), move, ok)
        st = EV.emit(st, ok, EV.EV_START, actor=r, move=CO.eff("encore"), v1=-1)
        q = st.queue
        mine = q.valid & (q.kind == S.Q_MOVE) & (Q.entry_ref(q) == r) & ~mget(m.fainted, r)
        will = DS.will_move(st, r) & ~mget(m.fainted, r)
        from vgcjax.effects import locked as LK
        cur = jnp.max(jnp.where(mine, jnp.where(q.move_slot == LK.RECHARGE_SLOT, -1, q.move_id.astype(I32)), 0))
        s = CO.vol_state(st, r, c)
        st = CO.set_vol_state(st, r, c, s.at[0].set(s[0] + 1), ok & ~will)
        change = ok & will & (cur != move)
        st = Q.cancel_action(st, r, change)
        st = C_.update_speed_one(st, dex, r, change)                 # insertChoice: choice.pokemon.updateSpeed()
        # resolve-time draws carry step_tag COMMIT (contract §3.1: insertChoice / changeAction resolve)
        tag0 = st.scratch.step_tag
        st = st.replace(scratch=st.scratch.replace(step_tag=jnp.asarray(ids.STEP_COMMIT, tag0.dtype)))
        tgt, rng = MP.get_random_target(st, dex, rng, r, move, 0, change)
        tside = jnp.clip(tgt, 0, 7) // 4
        tloc = jnp.where(tgt < 0, 0, jnp.where(tside == r // 4, -(MP.position(st, tgt) + 1), MP.position(st, tgt) + 1))
        st = C_.mul_tick(st, 2, change)                 # updateSpeed + resolveAction's getActionSpeed (ModifySpe x2)
        frac, rng = Q.frac_priority(st, dex, rng, r, move, change)
        prio = 10 * Q.move_priority(st, dex, r, move) + frac
        st = st.replace(scratch=st.scratch.replace(step_tag=tag0))
        st = Q.add_entry(st, change, kind=S.Q_MOVE, side=r // 4, b=r % 4, priority10=prio, frac10=frac,
                         speed=C_.action_speed(st, dex, r), move_slot=slot, move_id=move, target_loc=tloc,
                         original_target=MP.at_loc(st, r, tloc))
        return hc.ret(RS.R(RS.BOOL, 0), ~ok), st, rng

    @handler(ids.HOOK_END, CO.eff("encore"), ps="data/moves.ts encore.condition.onEnd")
    def encore_end(hc, st, dex, rng):
        """'-end' target 'Encore'."""
        return hc, EV.emit(st, True, EV.EV_END, actor=hc.target, move=CO.eff("encore"), v1=-1), rng

    @handler(ids.HOOK_RESIDUAL, CO.eff("encore"), ps="data/moves.ts encore.condition.onResidual")
    def encore_residual(hc, st, dex, rng):
        """onResidualOrder 16: the encored move's slot is gone or has 0 PP -> removeVolatile('encore')."""
        r, c = hc.target, CO.cid("encore")
        move = CO.vol_state(st, r, c)[CO.slot_of("encore", "move")].astype(I32)
        known, slot = _encore_slot(st, r, move)
        gone = ~known | (mget(st.mons.pp, r)[slot] <= 0)
        st, rng, _ = CO.remove_volatile(st, dex, rng, r, c, gone)
        return hc, st, rng


def encore_disable(st, dex, r, dis, on):
    """encore.condition.onDisableMove: the encored move known -> every other move slot disabled."""
    if not (mid("encore") > 0 and has("encore")):
        return dis
    c = CO.cid("encore")
    move = CO.vol_state(st, r, c)[CO.slot_of("encore", "move")].astype(I32)
    mids = mget(st.mons.move_ids, r).astype(I32)
    act = jnp.asarray(on) & CO.has_volatile(st, r, c) & (move > 0) & jnp.any(mids == move)
    return jnp.where(act & (mids != move) & (mids > 0), 1, dis)


# ------------------------------------------------------------------ Imprison
if mid("imprison") > 0 and has("imprison"):
    @handler(ids.HOOK_START, CO.eff("imprison"), ps="data/moves.ts imprison.condition.onStart")
    def imprison_start(hc, st, dex, rng):
        """'-start' target 'move: Imprison'."""
        return hc, EV.emit(st, True, EV.EV_START, actor=hc.target, move=CO.eff("imprison"), v1=-1), rng

    @handler(ids.HOOK_BEFOREMOVE, CO.eff("imprison"), prefix=ids.PFX_FOE,
             ps="data/moves.ts imprison.condition.onFoeBeforeMove")
    def imprison_foe_before_move(hc, st, dex, rng):
        """priority 4: not Struggle and the Imprison user knows the move -> 'cant' attacker 'move: Imprison' move;
        false."""
        move = jnp.maximum(hc.effect - ids.MOVE_BASE + 1, 0)
        src = jnp.clip(hc.holder, 0, 7)
        block = (move != ids.MOVE_STRUGGLE) & jnp.any(mget(st.mons.move_ids, src).astype(I32) == move)
        st = EV.emit(st, block, EV.EV_CANT, actor=hc.target, move=CO.eff("imprison"))
        return hc.ret(RS.R(RS.BOOL, 0), block), st, rng


def imprison_disable(st, r, dis, on):
    """imprison.condition.onFoeDisableMove: each move the Imprison user knows -> disableMove(id, true) ('hidden'; a
    flag already true stays true)."""
    if not (mid("imprison") > 0 and has("imprison")):
        return dis
    m = st.mons
    mids = mget(m.move_ids, r).astype(I32)
    fs = 1 - jnp.clip(r, 0, 7) // 4
    hit = jnp.zeros(4, bool)
    for b in range(2):
        a = st.sides.active[fs, b].astype(I32)
        h = fs * 4 + jnp.maximum(a, 0)
        held = jnp.asarray(on) & (a >= 0) & (mget(m.hp, h) > 0) & CO.has_volatile(st, h, CO.cid("imprison"))
        known = jnp.any(mids[:, None] == mget(m.move_ids, h).astype(I32)[None, :], axis=1)
        hit = hit | (held & known & (mids > 0))
    return jnp.where(hit & (dis == 0), 2, dis)


# ------------------------------------------------------------------ Trick (Sticky Hold onTryImmunity: not in Reg M-C)
if mid("trick") > 0:
    @handler(ids.HOOK_HIT, int(C.eff_move(mid("trick"))), ps="data/moves.ts trick.onHit")
    def trick_hit(hc, st, dex, rng):
        """yourItem = target.takeItem(source), myItem = source.takeItem() (no item: undefined; a Mega Stone of its
        holder's species: false); either false or neither item -> items put back (itemState stays cleared), false.
        Then each item must pass singleEvent('TakeItem') for its new holder, else the same. Else '-activate' source
        'move: Trick' '[of] target'; each side setItem (new itemState: effectOrder) + '-item' X item '[from] move:
        Trick', or '-enditem' X oldItem '[silent]' '[from] move: Trick' when it receives nothing."""
        from vgcjax.battle.util import mset
        from vgcjax.effects import moves_b1 as MB1
        t, u = jnp.clip(hc.target, 0, 7), jnp.clip(hc.source, 0, 7)
        m = st.mons
        yi, mi = mget(m.item, t).astype(I32), mget(m.item, u).astype(I32)
        y_false = (yi > 0) & ~MB1.takeable(st, yi, t)
        m_false = (mi > 0) & ~MB1.takeable(st, mi, u)
        taken_y = (yi > 0) & ~y_false                # takeItem cleared the item and its state
        taken_m = (mi > 0) & ~m_false
        fail1 = y_false | m_false | ((yi == 0) & (mi == 0))
        fail2 = ~fail1 & (((mi > 0) & ~MB1.takeable(st, mi, t)) | ((yi > 0) & ~MB1.takeable(st, yi, u)))
        ok = ~fail1 & ~fail2
        zs = jnp.zeros_like(mget(m.item_state, t))
        # takeItem: itemState cleared (effectOrder 0) for each taken item, kept cleared when put back
        m = m.replace(item_state=mset(mset(m.item_state, t, zs, taken_y), u, zs, taken_m),
                      item_seq=mset(mset(m.item_seq, t, 0, taken_y), u, 0, taken_m))
        st = st.replace(mons=m)
        st = EV.emit(st, ok, EV.EV_ACTIVATE, actor=u, move=C.eff_move(mid("trick")))
        seq = st.meta.effect_seq
        for holder, new, old in ((t, mi, yi), (u, yi, mi)):
            m = st.mons
            gets = ok & (new > 0)
            empties = ok & (new == 0)
            m = m.replace(item=mset(m.item, holder, new.astype(m.item.dtype), ok),
                          item_seq=mset(m.item_seq, holder, seq, gets))
            seq = seq + gets.astype(seq.dtype)
            st = st.replace(mons=m)
            st = EV.emit(st, gets, EV.EV_ITEM, actor=holder, move=ids.ITEM_BASE + new - 1, v1=-1)
            st = EV.emit(st, empties, EV.EV_ENDITEM, actor=holder, move=ids.ITEM_BASE + old - 1)
        st = st.replace(meta=st.meta.replace(effect_seq=seq))
        return hc.ret(RS.R(RS.BOOL, 0), ~ok), st, rng


# ------------------------------------------------------------------ Steel Roller
if mid("steelroller") > 0:
    _sr = int(C.eff_move(mid("steelroller")))

    @handler(ids.HOOK_TRY, _sr, ps="data/moves.ts steelroller.onTry")
    def steelroller_try(hc, st, dex, rng):
        """No terrain -> false."""
        return hc.ret(RS.R(RS.BOOL, 0), st.field.terrain == 0), st, rng

    @handler(ids.HOOK_HIT, _sr, ps="data/moves.ts steelroller.onHit")
    def steelroller_hit(hc, st, dex, rng):
        """field.clearTerrain() (FieldEnd: '-fieldend'; TerrainChange: no Reg M-C handler)."""
        from vgcjax.effects import sideconds as SC
        st, rng = SC.clear_terrain(st, dex, rng, True)
        return hc, st, rng


# ------------------------------------------------------------------ Double Shock (Champions: punch flag)
def double_shock_try(st, u, move, on):
    """Double Shock onTryMove: a user without the Electric type -> '-fail' pokemon 'move: Double Shock',
    attrLastMove('[still]'), null. Returns (st, null)."""
    if mid("doubleshock") < 0:
        return st, jnp.zeros((), bool)
    fail = jnp.asarray(on) & (move == mid("doubleshock")) & ~jnp.any(mget(st.mons.types, u) == ids.TYPE_ELECTRIC)
    st = EV.emit(st, fail, EV.EV_FAIL, actor=u, move=C.eff_move(mid("doubleshock")), flags=EV.F_FAILSTATUS)
    st = EV.attr_last_move(st, fail, EV.F_STILL)
    return st, fail


def double_shock_self(st, u, move, on):
    """self.onHit (selfDrops -> moveHit(source, source, move.self)): setType(types with Electric -> '???'); '-start'
    pokemon 'typechange' <types> '[from] move: Double Shock' (START move/v1 = Double Shock, v2 = 1:
    harness/events.py prints 'typechange')."""
    if mid("doubleshock") < 0:
        return st
    from vgcjax.battle.util import mset
    go = jnp.asarray(on) & (move == mid("doubleshock")) & (mget(st.mons.hp, u) > 0)
    m = st.mons
    t = mget(m.types, u)
    new = jnp.where(t == ids.TYPE_ELECTRIC, ids.TYPE_TYPELESS, t).astype(t.dtype)
    m = m.replace(types=mset(m.types, u, new, go), added_type=mset(m.added_type, u, 0, go))
    st = st.replace(mons=m)
    e = C.eff_move(mid("doubleshock"))
    return EV.emit(st, go, EV.EV_START, actor=u, move=e, v1=e, v2=1)


# ------------------------------------------------------------------ Glaive Rush
if mid("glaiverush") > 0 and has("glaiverush"):
    from vgcjax.effects.modifiers import mod

    @handler(ids.HOOK_START, CO.eff("glaiverush"), ps="data/moves.ts glaiverush.condition.onStart")
    def glaiverush_start(hc, st, dex, rng):
        """'-singlemove' pokemon 'Glaive Rush' '[silent]'."""
        return hc, EV.emit(st, True, EV.EV_SINGLEMOVE, actor=hc.target, move=CO.eff("glaiverush")), rng

    @handler(ids.HOOK_BEFOREMOVE, CO.eff("glaiverush"), ps="data/moves.ts glaiverush.condition.onBeforeMove")
    def glaiverush_before_move(hc, st, dex, rng):
        """priority 100: removeVolatile('glaiverush')."""
        st, rng, _ = CO.remove_volatile(st, dex, rng, hc.target, CO.cid("glaiverush"), True)
        return hc, st, rng

    @mod("ModifyDamage", CO.eff("glaiverush"), prefix=ids.PFX_SOURCE,
         ps="data/moves.ts glaiverush.condition.onSourceModifyDamage")
    def glaiverush_damage(ctx, h):
        """The holder takes chainModify(2)."""
        return jnp.ones(h.shape, bool), jnp.asarray(8192, I32), None


def glaive_rush_target(st, targets):
    """glaiverush.condition.onAccuracy (the holder is the target): accuracy true."""
    if not (mid("glaiverush") > 0 and has("glaiverush")):
        return jnp.zeros(jnp.shape(targets), bool)
    return CO.has_volatile(st, jnp.clip(targets, 0, 7), CO.cid("glaiverush")) & (jnp.asarray(targets) >= 0)
