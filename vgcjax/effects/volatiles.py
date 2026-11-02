"""Core volatiles (05 §4 step 2): flinch, confusion. PS: data/conditions.ts (no Champions overrides)."""
from __future__ import annotations

import jax.numpy as jnp

from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle import results as RS
from vgcjax.battle.util import mget
from vgcjax.core import rng as R
from vgcjax.core import stats as ST
from vgcjax.data import ids
from vgcjax.effects import conditions as CO
from vgcjax.effects.hooks import HookCtx, handler

I32 = jnp.int32
CO.fields("confusion", ("time", "int"))
LEVEL = 50


# ------------------------------------------------------------------ flinch
@handler(ids.HOOK_BEFOREMOVE, CO.eff("flinch"), ps="data/conditions.ts:211-216")
def flinch_before_move(hc, st, dex, rng):
    st = EV.emit(st, True, EV.EV_CANT, actor=hc.target, move=CO.eff("flinch"))
    from vgcjax.effects import ability_reactions as AR
    st = AR.steadfast(st, hc.target, True)                    # runEvent('Flinch', pokemon): Steadfast
    return hc.ret(RS.R(RS.BOOL, 0)), st, rng


# ------------------------------------------------------------------ confusion
@handler(ids.HOOK_START, CO.eff("confusion"), ps="data/conditions.ts:165-178")
def confusion_start(hc, st, dex, rng):
    ref = hc.target
    fatigue = hc.effect == CO.eff("lockedmove")
    is_ability = (hc.effect >= ids.ABILITY_BASE) & (hc.effect < ids.ABILITY_BASE + len(ids.ABILITIES) - 1)
    st = EV.emit(st, True, EV.EV_START, actor=ref, move=CO.eff("confusion"),
                 v1=jnp.where(is_ability, hc.effect, -1),
                 flags=jnp.where(fatigue, EV.F_FATIGUE, jnp.where(is_ability, EV.F_FROM_ABILITY, 0)))
    lo = jnp.where(hc.effect == C.eff_move(getattr(ids, "MOVE_AXEKICK", -1)), 3, 2)
    key = C.dkey(st, ids.DK_CONFUSION_DURATION, actor=ref, subject=CO.eff("confusion"))
    t, rng = R.randint(rng, key, lo, 6)
    st = CO.set_vol_field(st, ref, CO.cid("confusion"), CO.slot_of("confusion", "time"), t, True)
    return hc, st, rng


@handler(ids.HOOK_END, CO.eff("confusion"), ps="data/conditions.ts:179-181")
def confusion_end(hc, st, dex, rng):
    return hc, EV.emit(st, True, EV.EV_END, actor=hc.target, move=CO.eff("confusion")), rng


def confusion_damage(st, ref, roll):
    """PS: sim/battle-actions.ts:1904-1916 getConfusionDamage(pokemon, 40): calculateStat with boosts (no Modify*
    events), base damage + 2, 16-bit truncation, randomizer, at least 1."""
    m = st.mons
    stats, boosts = mget(m.stats, ref), mget(m.boosts, ref)
    atk = ST.boosted(stats[0].astype(I32), boosts[0].astype(I32))
    dfn = ST.boosted(stats[1].astype(I32), boosts[1].astype(I32))
    base = (((2 * LEVEL) // 5 + 2) * 40 * atk // dfn) // 50 + 2
    d = base & 0xFFFF
    d = (d * (100 - roll)) // 100                         # PS battle.randomizer: tr(tr(d * (100 - r)) / 100)
    return jnp.maximum(d, 1)


@handler(ids.HOOK_BEFOREMOVE, CO.eff("confusion"), ps="data/conditions.ts:182-198")
def confusion_before_move(hc, st, dex, rng):
    from vgcjax.effects import statuses as STS
    ref = hc.target
    c = CO.cid("confusion")
    slot = CO.slot_of("confusion", "time")
    t = CO.vol_state(st, ref, c)[slot].astype(I32) - 1
    st = CO.set_vol_field(st, ref, c, slot, t, True)
    end = t == 0
    st, rng, _ = CO.remove_volatile(st, dex, rng, ref, c, end)
    st = EV.emit(st, ~end, EV.EV_ACTIVATE, actor=ref, move=CO.eff("confusion"))
    key = C.dkey(st, ids.DK_CONFUSION_SELFHIT, actor=ref, subject=CO.eff("confusion"))
    hit, rng = R.chance(rng, key, 33, 100, active=~end)
    selfhit = ~end & hit
    rkey = C.dkey(st, ids.DK_CONFUSION_SELFHIT_DAMAGE_ROLL, actor=ref, target=ref, subject=CO.eff("confusion"))
    roll, rng = R.randint(rng, rkey, 0, 16, active=selfhit)
    # battle.damage(d, pokemon, pokemon, {id: 'confused'}) -> '-damage' X hp '[from] confusion'
    st, rng = STS.effect_damage(st, dex, rng, ref, confusion_damage(st, ref, roll), CO.eff("confusion"), selfhit,
                                source=ref)
    return hc.ret(RS.R(RS.BOOL, 0), selfhit), st, rng


# ------------------------------------------------------------------ partiallytrapped (Bind, Fire Spin, Infestation, ...)
if "partiallytrapped" in ids.CONDITIONS:
    CO.fields("partiallytrapped", ("boundDivisor", "int"))
    PT_SRC, PT_MOVE = 2, 3                    # raw slots: effectState.source / sourceEffect (dropped by the projection)
    _PT = CO.cid("partiallytrapped")

    @handler(CO.DURATION_CALLBACK, CO.eff("partiallytrapped"), ps="data/conditions.ts:225-228")
    def partiallytrapped_duration(hc, st, dex, rng):
        """source.hasItem('gripclaw') -> 8 (not Reg M-C legal); else random(5, 7)."""
        key = C.dkey(st, ids.DK_PARTIAL_TRAP_DURATION, actor=hc.target, subject=CO.eff("partiallytrapped"))
        d, rng = R.randint(rng, key, 5, 7)
        return hc.ret(RS.R(RS.NUM, d)), st, rng

    @handler(ids.HOOK_START, CO.eff("partiallytrapped"), ps="data/conditions.ts:229-232")
    def partiallytrapped_start(hc, st, dex, rng):
        """'-activate' pokemon 'move: <sourceEffect>' '[of] source'; boundDivisor = Binding Band ? 6 : 8."""
        r, src = hc.target, jnp.clip(hc.source, 0, 7)
        band = mget(st.mons.item, src).astype(I32) == getattr(ids, "ITEM_BINDINGBAND", -1)
        s = CO.vol_state(st, r, _PT)
        s = s.at[PT_SRC].set(jnp.asarray(hc.source, s.dtype)).at[PT_MOVE].set(jnp.asarray(hc.effect, s.dtype))
        st = CO.set_vol_state(st, r, _PT, s, True)
        st = CO.set_vol_field(st, r, _PT, CO.slot_of("partiallytrapped", "boundDivisor"), jnp.where(band, 6, 8), True)
        return hc, EV.emit(st, True, EV.EV_ACTIVATE, actor=r, move=hc.effect), rng

    def _pt_source_gone(st, r):
        src = CO.vol_state(st, r, _PT)[PT_SRC].astype(I32)
        s = jnp.clip(src, 0, 7)
        m = st.mons
        return (src >= 0) & (~mget(m.is_active, s) | (mget(m.hp, s) <= 0) | (mget(m.active_turns, s) == 0))

    @handler(ids.HOOK_RESIDUAL, CO.eff("partiallytrapped"), ps="data/conditions.ts:233-244")
    def partiallytrapped_residual(hc, st, dex, rng):
        """onResidualOrder 13: the source left / fainted / has no activeTurns -> delete the volatile, '-end' pokemon
        <sourceEffect> '[partiallytrapped]' '[silent]'; else damage(baseMaxhp / boundDivisor) ('-damage' X hp
        '[from] <sourceEffect>' '[partiallytrapped]')."""
        from vgcjax.effects import statuses as STS
        r = hc.holder
        s = CO.vol_state(st, r, _PT).astype(I32)
        gone = _pt_source_gone(st, r)
        st = EV.emit(st, gone, EV.EV_END, actor=r, move=s[PT_MOVE], v1=-1)
        st = CO._set_vol_bit(st, r, _PT, False, gone)
        div = jnp.maximum(s[CO.slot_of("partiallytrapped", "boundDivisor")], 1)
        st, rng = STS.effect_damage(st, dex, rng, r, mget(st.mons.maxhp, r).astype(I32) // div, s[PT_MOVE], ~gone)
        return hc, st, rng

    @handler(ids.HOOK_END, CO.eff("partiallytrapped"), ps="data/conditions.ts:245-247")
    def partiallytrapped_end(hc, st, dex, rng):
        """'-end' pokemon <sourceEffect> '[partiallytrapped]'."""
        r = hc.target
        mv = CO.vol_state(st, r, _PT)[PT_MOVE].astype(I32)
        return hc, EV.emit(st, True, EV.EV_END, actor=r, move=mv, v1=-1), rng


def partially_trapped(st, r):
    """partiallytrapped onTrapPokemon (data/conditions.ts:248-251): effectState.source.isActive -> tryTrap()
    (runStatusImmunity('trapped'): Ghost types are immune)."""
    if "partiallytrapped" not in ids.CONDITIONS:
        return jnp.zeros((), bool)
    m = st.mons
    src = CO.vol_state(st, r, _PT)[PT_SRC].astype(I32)
    ghost = jnp.any(mget(m.types, r) == ids.TYPE_GHOST)
    return CO.has_volatile(st, r, _PT) & (src >= 0) & mget(m.is_active, jnp.clip(src, 0, 7)) & ~ghost
