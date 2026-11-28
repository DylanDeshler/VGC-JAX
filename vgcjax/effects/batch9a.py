"""Batch 9a (Reg M-C ladder blockers): Clear Smog, Burning Jealousy, Fickle Beam, Speed Swap, Barb Barrage, Shell Side
Arm, Defog, Flail, Metal Burst (+ the attackedBy list behind getLastDamagedBy).

PS: data/moves.ts (line numbers below); no Champions override (data/mods/champions/moves.ts has none of these moves).
Call sites outside this module (minimal hooks, each commented at the site):
- Burning Jealousy: effects/moves_b1.py secondary_special (secondary.onHit);
- Fickle Beam: battle/move_pipeline.py get_damage (the draw before runEvent('BasePower'), ctx.extra['fickle']) and
  emit_damage_msgs ('-activate' before the getDamage messages of that target);
- Shell Side Arm: battle/move_pipeline.py move_begin_front (onModifyMove after the ModifyMove events) and the tops of
  move_begin_back / move_hit / move_end (ssa_dex: the ActiveMove's category / contact flag for the whole execution);
  battle/state.py MoveCarry.a_phys / Exec.phys;
- Flail: effects/batch8a.py reversal_bp (the same basePowerCallback as Reversal);
- Metal Burst: battle/move_pipeline.py move_begin_front (runEvent('ModifyTarget')) and move_end (gotAttacked),
  effects/move_callbacks.py fixed_damage (damageCallback), battle/step.py end_turn (attackedBy thisTurn / splice),
  battle/switching.py clear_volatile_many (attackedBy = []), battle/state.py Mons.dmg_by, battle/snapshot.py;
- Clear Smog: '-clearboost' is events.py CLEARBOOST (oracle/lib/protocol.js maps the PS line).
"""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget, mset, mset_many
from vgcjax.core import rng as R
from vgcjax.core import stats as ST
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import handler
from vgcjax.effects.modifiers import mod

I16, I32 = jnp.int16, jnp.int32


def mid(name):
    return getattr(ids, f"MOVE_{name.upper()}", -999)


def has(name):
    return name in ids.CONDITIONS


def meff(name):
    """EffectId of a move (a Python int)."""
    m = mid(name)
    return ids.MOVE_BASE + m - 1 if m > 0 else -1


# ================================================================== Clear Smog (data/moves.ts:2554-2570)
if mid("clearsmog") > 0:
    @handler(ids.HOOK_HIT, meff("clearsmog"), ps="data/moves.ts:2563-2566 clearsmog.onHit")
    def clearsmog_hit(hc, st, dex, rng):
        """target.clearBoosts() (sim/pokemon.ts clearBoosts: every boost 0), '-clearboost' target."""
        t = jnp.clip(hc.target, 0, 7)
        m = st.mons
        st = st.replace(mons=m.replace(boosts=mset(m.boosts, t, jnp.zeros(7, m.boosts.dtype))))
        return hc, EV.emit(st, True, EV.EV_CLEARBOOST, actor=t), rng


# ================================================================== Burning Jealousy (data/moves.ts:2071-2091)
def burning_jealousy(st, dex, rng, u, move, t, ok):
    """secondary.onHit(target, source, move) (data/moves.ts:2082-2086; chance 100: the SECONDARY roll is the generic
    one): target.statsRaisedThisTurn -> target.trySetStatus('brn', source, move) (sourceEffect = the move: no
    '[from]' on the brn line, data/conditions.ts:5-13). Returns (st, rng)."""
    if mid("burningjealousy") <= 0:
        return st, rng
    from vgcjax.effects import berries as BE
    from vgcjax.effects import statuses as STS
    tr = jnp.clip(jnp.asarray(t, I32), 0, 7)
    on = ok & (move == mid("burningjealousy")) & (jnp.asarray(t, I32) >= 0) & mget(st.mons.stats_raised_this_turn, tr)
    st, rng, _, msgs = STS.try_set_status_v(st, dex, rng, tr[None], ids.COND_BRN, on[None], u,
                                            C.eff_move(mid("burningjealousy")), 0)
    st = STS.emit_status_msgs(st, msgs, 0, tr)
    return BE.after_set_status(st, dex, rng, msgs, 0, tr)


# ================================================================== Fickle Beam (data/moves.ts:5221-5239)
FICKLE = mid("ficklebeam") > 0


def fickle_draw(st, rng, u, move, targets, hit_no, reach):
    """onBasePower(basePower, pokemon) (data/moves.ts:5230-5236): this.randomChance(3, 10) each time the move's
    BasePower handler runs (runEvent('BasePower') per target reaching it; MOVE_FICKLE_BEAM: actor user, target, the
    move, hit). Returns (bool [T], rng): the chainModify(2) / '-activate' outcome."""
    if not FICKLE:
        return jnp.zeros(jnp.shape(targets), bool), rng
    on = jnp.asarray(reach) & (move == mid("ficklebeam"))
    key = C.dkey(st, ids.DK_MOVE_FICKLE_BEAM, actor=u, target=targets, subject=C.eff_move(move), hit=hit_no)
    ok, rng = R.chance_v(rng, key, 3, 10, active=on)
    return on & ok, rng


if FICKLE:
    @mod("BasePower", int(C.eff_move(mid("ficklebeam"))), ps="data/moves.ts:5230-5236 ficklebeam.onBasePower")
    def ficklebeam_bp(ctx, h):
        """The drawn randomChance(3, 10) (fickle_draw, ctx.extra['fickle']) -> chainModify(2)."""
        f = ctx.extra.get("fickle")
        f = jnp.zeros(h.shape, bool) if f is None else f
        return f, jnp.asarray(8192, I32), None


def fickle_emit(st, u, f):
    """attrLastMove('[anim] Fickle Beam All Out') (not compared), '-activate' pokemon 'move: Fickle Beam' (pokemon:
    the BasePower event target = the user), inside getDamage before modifyDamage's messages."""
    if not FICKLE:
        return st
    return EV.emit(st, f, EV.EV_ACTIVATE, actor=u, move=meff("ficklebeam"))


# ================================================================== Speed Swap (data/moves.ts:17426-17445)
if mid("speedswap") > 0:
    @handler(ids.HOOK_HIT, meff("speedswap"), ps="data/moves.ts:17435-17440 speedswap.onHit")
    def speedswap_hit(hc, st, dex, rng):
        """Swap target.storedStats.spe and source.storedStats.spe (no updateSpeed here); '-activate' source
        'move: Speed Swap' '[of] target'."""
        t, u = jnp.clip(hc.target, 0, 7), jnp.clip(hc.source, 0, 7)
        m = st.mons
        ts, us = mget(m.stats, t), mget(m.stats, u)
        stats = mset(m.stats, t, ts.at[4].set(us[4]))
        stats = mset(stats, u, mget(stats, u).at[4].set(ts[4]))
        st = st.replace(mons=m.replace(stats=stats))
        return hc, EV.emit(st, True, EV.EV_ACTIVATE, actor=u, move=meff("speedswap")), rng


# ================================================================== Barb Barrage (data/moves.ts:1038-1057)
if mid("barbbarrage") > 0:
    from vgcjax.effects import moves_b2 as _MB2
    # onBasePower(basePower, pokemon, target): target.status psn / tox -> chainModify(2): Venoshock's handler
    mod("BasePower", int(C.eff_move(mid("barbbarrage"))), ps="data/moves.ts:1047-1051 barbbarrage.onBasePower")(
        _MB2.venoshock_bp)


# ================================================================== Shell Side Arm (data/moves.ts:16210-16251)
SSA = mid("shellsidearm") > 0


def ssa_physical(st, rng, u, target, move, on):
    """onModifyMove(move, pokemon, target) (data/moves.ts:16224-16236; the move's ModifyMove singleEvent with
    useMoveInner's target): no target -> unchanged; else with getStat(stat, false, true) (stored stat x boost stage,
    no ModifyBoost / Modify* events: sim/pokemon.ts:617-657) physical = floor(floor(floor(22 * 90 * atk) / def) / 50),
    special likewise with spa / spd (level 50); physical > special, or a tie and randomChance(1, 2)
    (MOVE_SHELLSIDEARM_TIE), -> category Physical, flags.contact. Returns (bool, rng)."""
    if not SSA:
        return jnp.zeros((), bool), rng
    m = st.mons
    tgt = jnp.asarray(target, I32)
    go = jnp.asarray(on) & (jnp.asarray(move) == mid("shellsidearm")) & (tgt >= 0)
    uc, tc = jnp.clip(jnp.asarray(u, I32), 0, 7), jnp.clip(tgt, 0, 7)
    us, ub = mget(m.stats, uc).astype(I32), mget(m.boosts, uc).astype(I32)
    ts, tb = mget(m.stats, tc).astype(I32), mget(m.boosts, tc).astype(I32)
    a = (2 * 50) // 5 + 2
    atk, spa = ST.boosted(us[0], ub[0]), ST.boosted(us[2], ub[2])
    df, sd = jnp.maximum(ST.boosted(ts[1], tb[1]), 1), jnp.maximum(ST.boosted(ts[3], tb[3]), 1)
    phys = ((a * 90 * atk) // df) // 50
    spec = ((a * 90 * spa) // sd) // 50
    key = C.dkey(st, ids.DK_MOVE_SHELLSIDEARM_TIE, actor=uc, target=tc, subject=C.eff_move(move))
    tie, rng = R.chance(rng, key, 1, 2, active=go & (phys == spec))
    return go & ((phys > spec) | ((phys == spec) & tie)), rng


def ssa_dex(dex, phys):
    """The Dex seen by the rest of a Shell Side Arm execution whose onModifyMove made it Physical: its category row
    Physical and its contact flag set (every getCategory / flags.contact read of the ActiveMove: damage stats, burn,
    screens, Counter / Mirror Coat, contact abilities / items / protections)."""
    if not SSA:
        return dex
    from vgcjax.core.dex import Dex
    mv = mid("shellsidearm")
    phys = jnp.asarray(phys)
    cat = dex.move_category
    cat = cat.at[mv].set(jnp.where(phys, ids.CAT_PHYSICAL, cat[mv]).astype(cat.dtype))
    w, b = ids.MF_CONTACT
    fl = dex.move_flags
    fl = fl.at[mv, w].set(jnp.where(phys, fl[mv, w] | jnp.asarray(1 << b, fl.dtype), fl[mv, w]))
    return Dex(tables={**dex.tables, "move_category": cat, "move_flags": fl})


# ================================================================== Defog (data/moves.ts:3447-3481)
DEFOG_ALL = ("spikes", "toxicspikes", "stealthrock", "stickyweb", "gmaxsteelsurge")
DEFOG_TARGET = ("reflect", "lightscreen", "auroraveil", "safeguard", "mist") + DEFOG_ALL

if mid("defog") > 0:
    @handler(ids.HOOK_HIT, meff("defog"), ps="data/moves.ts:3456-3474 defog.onHit")
    def defog_hit(hc, st, dex, rng):
        """No substitute on the target (or move.infiltrates: Infiltrator) -> success = !!boost({evasion: -1}) (target,
        source, effect Defog); each of reflect, lightscreen, auroraveil, safeguard, mist, then the hazards removed
        from target.side (removeSideCondition: SideEnd, e.g. '-sideend' X 'Reflect'); a hazard also prints '-sideend'
        side name '[from] move: Defog' '[of] source' and sets success; then each hazard removed from source.side
        likewise; field.clearTerrain(); return success."""
        from vgcjax.battle import move_pipeline as MP
        from vgcjax.effects import ability_mods as AM
        from vgcjax.effects import sideconds as SC
        t, u = jnp.clip(hc.target, 0, 7), jnp.clip(hc.source, 0, 7)
        sub = CO.has_volatile(st, t, CO.cid("substitute")) if has("substitute") else jnp.zeros((), bool)
        can = ~sub | AM.infiltrates(st, u)
        st, br = MP.boost(st, t, jnp.zeros(7, I32).at[6].set(-1), can, primary=True, source=u, effect=meff("defog"))
        success = can & RS.truthy(br)
        for side, names in ((t // 4, DEFOG_TARGET), (u // 4, DEFOG_ALL)):
            for n in names:
                if not has(n):
                    continue
                c = CO.cid(n)
                rm = CO.has_side(st, side, c)
                st, rng = SC.remove_side_condition(st, dex, rng, side, c, rm)
                if n in DEFOG_ALL:
                    st = EV.emit(st, rm, EV.EV_SIDEEND, actor=side, move=CO.eff(n))
                    success = success | rm
        st, rng = SC.clear_terrain(st, dex, rng, True)
        return hc.ret(RS.R(RS.BOOL, success.astype(I32))), st, rng


# ================================================================== Metal Burst (data/moves.ts:11657-11685) and the
# attackedBy list (sim/pokemon.ts:938-963 gotAttacked / getLastDamagedBy)
MB = mid("metalburst") > 0
DB_SEQ, DB_DMG, DB_SLOT, DB_THIS = 0, 1, 2, 3     # Mons.dmg_by[..., src, :] columns (seq 0: no entry)


def got_attacked(st, u, targets, num, dmg):
    """target.gotAttacked(move, damage, pokemon) for each target with a numeric damage ([C] scripts.ts:559-566;
    sim/pokemon.ts:938-949): an attackedBy entry {source, damage, thisTurn: true, slot: source.getSlot()}. Only the
    latest numeric entry per source is kept (getLastDamagedBy filters numeric damageValue; the end-of-turn splice
    removes all entries of a source at once), ordered by a per-target sequence number. Non-numeric entries are not
    kept: no Reg M-C reader of attackedBy sees them besides getLastAttackedBy (knownType, not modelled)."""
    from vgcjax.battle import move_pipeline as MP
    m = st.mons
    targets = jnp.asarray(targets, I32)
    uc = jnp.clip(jnp.asarray(u, I32), 0, 7)
    rows = mget(m.dmg_by, targets).astype(I32)                                # [T, 8, 4]
    seq = jnp.max(rows[:, :, DB_SEQ], axis=1) + 1
    code = (uc // 4) * 2 + jnp.maximum(MP.position(st, uc), 0)
    T = targets.shape[0]
    new = jnp.stack([seq, jnp.asarray(dmg, I32), jnp.full(T, code, I32), jnp.ones(T, I32)], axis=1)
    rows = rows.at[:, uc, :].set(new)
    return st.replace(mons=m.replace(dmg_by=mset_many(m.dmg_by, targets, rows.astype(m.dmg_by.dtype), num)))


def end_turn_attacked(st, m, r, go):
    """endTurn (sim/battle.ts:1720-1727) for an active pokemon: each attackedBy entry whose source is active gets
    thisTurn = false, the others are spliced out. Returns the Mons."""
    rows = mget(m.dmg_by, r)
    act = m.is_active.reshape(8)
    rows = jnp.where(act[:, None], rows.at[:, DB_THIS].set(0), 0)
    return m.replace(dmg_by=mset(m.dmg_by, r, rows, go))


def last_damaged_by(st, r):
    """getLastDamagedBy(true) (sim/pokemon.ts:956-963): the latest numeric entry from a non-ally source. Returns
    (exists, damage, slot code side*2+pos, thisTurn)."""
    r = jnp.clip(jnp.asarray(r, I32), 0, 7)
    rows = mget(st.mons.dmg_by, r).astype(I32)
    ok = (rows[:, DB_SEQ] > 0) & (jnp.arange(8) // 4 != r // 4)
    i = jnp.argmax(jnp.where(ok, rows[:, DB_SEQ], -1))
    e = rows[i]
    return jnp.any(ok), e[DB_DMG], e[DB_SLOT], jnp.any(ok) & (e[DB_THIS] > 0)


def metal_burst_target(st, u, move, target, on):
    """onModifyTarget(targetRelayVar, source) (data/moves.ts:11677-11682; useMoveInner's runEvent('ModifyTarget'),
    sim/battle-actions.ts:479-481, before the move line): a lastDamagedBy -> getAtSlot(slot) (the occupant, possibly
    fainted; an empty slot -> no target)."""
    if not MB:
        return target
    ex, _, code, _ = last_damaged_by(st, u)
    s, p = jnp.clip(code // 2, 0, 1), jnp.clip(code % 2, 0, 1)
    b = st.sides.active[s, p].astype(I32)
    new = jnp.where(b >= 0, s * 4 + jnp.maximum(b, 0), -1)
    return jnp.where(jnp.asarray(on) & (jnp.asarray(move) == mid("metalburst")) & ex, new, jnp.asarray(target, I32))


def metal_burst_damage(st, u, move, fixed):
    """damageCallback(pokemon) (data/moves.ts:11661-11667): lastDamagedBy -> (damage * 1.5) || 1, else 0; the
    fraction is floored by spreadDamage's clampIntRange (sim/battle.ts:2130)."""
    if not MB:
        return fixed
    ex, d, _, _ = last_damaged_by(st, u)
    v = jnp.where(ex, jnp.maximum((3 * d) // 2, 1), 0)
    return jnp.where(move == mid("metalburst"), v, fixed)


if MB:
    @handler(ids.HOOK_TRY, meff("metalburst"), ps="data/moves.ts:11673-11676 metalburst.onTry")
    def metalburst_try(hc, st, dex, rng):
        """!source.getLastDamagedBy(true)?.thisTurn -> false."""
        _, _, _, this_turn = last_damaged_by(st, hc.target)
        return hc.ret(RS.R(RS.BOOL, 0), ~this_turn), st, rng
