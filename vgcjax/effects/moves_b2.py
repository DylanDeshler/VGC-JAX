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
                         source=hc.source, effect=_cs)
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
