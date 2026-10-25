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
