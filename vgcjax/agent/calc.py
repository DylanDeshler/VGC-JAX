"""The engine as an exact damage calculator (belief filter likelihoods, the DamageCalc tool).

A two-mon "duel" state holds the attacker at p1a (ref 0) and the defender at p2a (ref 4), built by the engine's own
preview path (INIT.apply_preview / build_mon) and made active without switch-in hooks (no Intimidate etc.: boosts,
status and field are whatever the caller sets). damage_rolls runs the real move pipeline's getDamage
(battle/move_pipeline.py get_damage: basePowerCallback, the modifier chain of every implemented ability / item / field
effect, crit and type effectiveness) with the DAMAGE_ROLL and CRIT draws forced through the semantic-key RNG
(core/rng.py ForcedOutcomes), once per roll 0..15 (vmapped). So every mechanic the engine reproduces from PS is in
the calculator; tests/agent/test_calc.py checks it against PS's own getDamage through the oracle's 'damage' RPC.

Hypotheses are applied by overwriting fields of the duel state (set_stats / set_item / set_ability), so one compiled
function serves every (skeleton, stat) grid point under vmap.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vgcjax.battle import common as C
from vgcjax.battle import init as INIT
from vgcjax.battle import move_pipeline as MP
from vgcjax.battle import state as S
from vgcjax.battle.util import mget, mset
from vgcjax.core import drawkey as dk
from vgcjax.core import rng as R
from vgcjax.core import stats as ST
from vgcjax.data import ids
from vgcjax.env import actions as A
from vgcjax.effects import registry  # noqa: F401  (handler registration must happen before tracing)

I32 = jnp.int32
ATK, DEF = 0, 4                      # refs of the duel's two active mons (p1a, p2a)
NT = 3


def team_arrays(atk: dict, dfn: dict) -> S.Team:
    """Engine Team with `atk` in every sheet slot of side 0 and `dfn` in every slot of side 1 (sets as id dicts:
    species, item, ability, nature, gender, moves[4], sp[6]). The copies only fill the sheet (the PS 'damage' RPC uses
    the attacker set as filler too)."""
    def side(x):
        return {k: np.repeat(np.asarray(x[k])[None], 6, axis=0) for k in ("species", "item", "ability", "nature",
                                                                          "gender", "moves", "sp")}
    a, d = side(atk), side(dfn)
    st = lambda k, dt: jnp.asarray(np.stack([a[k], d[k]]).astype(dt))  # noqa: E731
    return S.Team(species=st("species", np.int16), item=st("item", np.int16), ability=st("ability", np.int16),
                  nature=st("nature", np.int8), gender=st("gender", np.int8), moves=st("moves", np.int16),
                  sp=st("sp", np.int8))


def duel_state(dex, team: S.Team) -> S.State:
    """Brought order 0..3 for both sides, then sheet 0 of each side active at slot a (no switch-in hooks)."""
    st = INIT.init(team)
    order = jnp.asarray(A.PREVIEW_ORDERS, I32)
    pick = jnp.argmax(jnp.all(order == jnp.arange(4, dtype=I32)[None, :], axis=1))
    st = INIT.apply_preview(st, dex, jnp.stack([pick, pick]))
    sd, m = st.sides, st.mons
    active = sd.active.at[0, 0].set(0).at[1, 0].set(0)
    is_act = m.is_active.at[0, 0].set(True).at[1, 0].set(True)
    st = st.replace(sides=sd.replace(active=active.astype(sd.active.dtype)), mons=m.replace(is_active=is_act))
    return st.replace(meta=st.meta.replace(phase=jnp.asarray(S.MOVE, st.meta.phase.dtype), turn=jnp.asarray(1, st.meta.turn.dtype)))


def set_stats(st: S.State, dex, ref, species, sp6, nature):
    """Recompute ref's stats / maxhp (and full hp) for an SP spread and nature (core/stats.calc_stats)."""
    maxhp, stats = ST.calc_stats(dex.species_base_stats[species], jnp.asarray(sp6, I32), dex.nature_plus[nature],
                                 dex.nature_minus[nature])
    m = st.mons
    return st.replace(mons=m.replace(stats=mset(m.stats, ref, stats.astype(m.stats.dtype), True),
                                     maxhp=mset(m.maxhp, ref, jnp.asarray(maxhp, m.maxhp.dtype), True),
                                     hp=mset(m.hp, ref, jnp.asarray(maxhp, m.hp.dtype), True)))


def set_item(st: S.State, ref, item):
    m = st.mons
    return st.replace(mons=m.replace(item=mset(m.item, ref, jnp.asarray(item, m.item.dtype), True)))


def set_ability(st: S.State, ref, ability):
    m = st.mons
    ab = jnp.asarray(ability, m.ability.dtype)
    return st.replace(mons=m.replace(ability=mset(m.ability, ref, ab, True),
                                     ability_state_of=mset(m.ability_state_of, ref, ab.astype(m.ability_state_of.dtype),
                                                           True)))


def _forced(st, u, t, move, roll, crit):
    ek = C.eff_move(move)
    rk = C.dkey(st, ids.DK_DAMAGE_ROLL, actor=u, target=t, subject=ek, hit=1)
    ck = C.dkey(st, ids.DK_CRIT, actor=u, target=t, subject=ek, hit=1)
    pk = lambda k: jnp.stack([jnp.asarray(w).astype(jnp.uint32) for w in dk.pack(k, jnp)])  # noqa: E731
    keys = jnp.stack([pk(rk), pk(ck)])
    return R.ForcedOutcomes(keys=keys, value=jnp.stack([jnp.asarray(roll, I32), jnp.asarray(crit, I32)]),
                            valid=jnp.ones(2, bool))


def damage_rolls(st: S.State, dex, cfg, move, crit=False, u=ATK, t=DEF, spread=False):
    """i32[16] damage for rolls 0..15 (PS random(16): damage * (100 - roll) / 100) of `move` from u to t, and its
    result kind (move_pipeline result tags: NUM / immune-false / ...), via the engine's getDamage."""
    targets = jnp.asarray([t, -1, -1], I32)
    on = jnp.asarray([True, False, False])

    def one(r):
        rng = R.make_rng(jax.random.PRNGKey(0), st.meta.step_counter, forced=_forced(st, u, t, move, r, crit))
        mt, bp, _ = MP.modified_type_bp(st, dex, jnp.asarray(u, I32), jnp.asarray(t, I32), jnp.asarray(move, I32))
        _, (tags, vals), _, _, _ = MP.get_damage(st, dex, rng, cfg, jnp.asarray(u, I32), jnp.asarray(move, I32),
                                                 targets, jnp.asarray(spread), 1, on, mtype=mt, bp=bp)
        return vals[0], tags[0]
    vals, tags = jax.vmap(one)(jnp.arange(16, dtype=I32))
    return vals, tags[0]
