"""Symbolic Bayesian belief over the opponent's hidden sets (Bo1, closed team sheets).

Per opponent sheet slot i (species known from team preview):
  skeleton belief  logw[i, K]      over K prior set skeletons (item, ability, 4 moves, nature; tools/belief/build_priors)
  SP belief        sp[i, 6, 33]    per-stat marginals over SP 0..32 (hp, atk, def, spa, spd, spe), chaos prior
The joint is factorised b(k) * prod_s b_s(v); every observation depends on the skeleton and at most two SP stats, so its
update is exact on that sub-grid and folded back into the factors (assumed-density filtering):
  reveal(slot, item | ability | move)    hard masks on skeletons (a move outside every skeleton: 'off_prior' flag)
  damage_taken(...)                      the opponent hit our mon (our HP is exact): grid K x atk-SP(33) x 16 rolls,
                                         likelihood = P(roll gives exactly the observed damage)
  damage_dealt(...)                      we hit the opponent (public HP% only): grid K x hp-SP x def-SP x 16 rolls,
                                         likelihood = P(predicted HP% after == observed HP%)
  speed_order(...)                       turn order in one priority bracket: opponent speed per K x spe-SP vs ours
Damage comes from the engine itself (vgcjax/agent/calc.py: exact vs PS), on a hypothetical copy of the real board with
the opponent mon's item / ability / stats replaced by the hypothesis, so field, boosts and status are exact. A
hypothesis the evidence makes impossible gets probability 0 (with an epsilon floor on the factor so a set outside
the prior degrades gracefully instead of zeroing the belief: off_prior is raised when that happens).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from vgcjax.agent import calc as CA
from vgcjax.battle import common as C
from vgcjax.battle import events as EV
from vgcjax.battle.util import mget, mset
from vgcjax.core import stats as ST
from vgcjax.data import ids

F32, I32 = jnp.float32, jnp.int32
NSP = 33
EPS = 1e-6                   # likelihood floor for a factor (keeps an off-prior truth from zeroing everything)
HP, ATK, DEF, SPA, SPD, SPE = range(6)


@struct.dataclass
class Priors:
    skel_item: jax.Array      # i32[NS, K]
    skel_ability: jax.Array   # i32[NS, K]
    skel_moves: jax.Array     # i32[NS, K, 4]
    skel_nature: jax.Array    # i32[NS, K]
    skel_w: jax.Array         # f32[NS, K]
    skel_n: jax.Array         # i32[NS]
    sp_marg: jax.Array        # f32[NS, 6, 33]


def load_priors(path) -> Priors:
    z = np.load(path)
    return Priors(**{k: jnp.asarray(z[k]) for k in ("skel_item", "skel_ability", "skel_moves", "skel_nature", "skel_w",
                                                     "skel_n", "sp_marg")})


@struct.dataclass
class Belief:
    species: jax.Array        # i32[6] opponent sheet species (team preview)
    logw: jax.Array           # f32[6, K] skeleton log-weights (-inf = ruled out)
    sp: jax.Array             # f32[6, 6, 33] per-stat SP marginals
    off_prior: jax.Array      # bool[6] evidence fell outside the prior's support for this mon


def init(pri: Priors, species) -> Belief:
    species = jnp.asarray(species, I32)
    K = pri.skel_w.shape[1]
    valid = jnp.arange(K)[None, :] < pri.skel_n[species][:, None]
    logw = jnp.where(valid, jnp.log(jnp.maximum(pri.skel_w[species], 1e-12)), -jnp.inf)
    return Belief(species=species, logw=logw, sp=pri.sp_marg[species], off_prior=jnp.zeros(6, bool))


def _renorm(logw):
    m = jnp.max(logw, axis=-1, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    z = jnp.log(jnp.sum(jnp.exp(logw - m), axis=-1, keepdims=True)) + m
    return logw - z


def skel_probs(b: Belief):
    """f32[6, K]."""
    w = jnp.exp(_renorm(b.logw))
    return jnp.where(jnp.isfinite(b.logw), w, 0.0)


def _mask_update(b: Belief, slot, keep):
    """Hard evidence on skeletons of `slot` (keep: bool[K]); all ruled out -> keep the old weights, flag off_prior."""
    lw = b.logw[slot]
    new = jnp.where(keep, lw, -jnp.inf)
    any_left = jnp.any(jnp.isfinite(new))
    lw2 = jnp.where(any_left, _renorm(new[None])[0], lw)
    return b.replace(logw=b.logw.at[slot].set(lw2), off_prior=b.off_prior.at[slot].set(b.off_prior[slot] | ~any_left))


def reveal_item(b: Belief, pri: Priors, slot, item):
    return _mask_update(b, slot, pri.skel_item[b.species[slot]] == item)


def reveal_ability(b: Belief, pri: Priors, slot, ability):
    return _mask_update(b, slot, pri.skel_ability[b.species[slot]] == ability)


def reveal_move(b: Belief, pri: Priors, slot, move):
    return _mask_update(b, slot, jnp.any(pri.skel_moves[b.species[slot]] == move, axis=-1))


def _apply_joint(b: Belief, slot, lik_k_a, stat_a, lik_full=None, stat_b=None):
    """Fold a likelihood over (skeleton K, SP of stat_a [, SP of stat_b]) into the factors of `slot`.
    lik_k_a: f32[K, 33] (or lik_full: f32[K, 33, 33] over (stat_a, stat_b))."""
    pk = skel_probs(b)[slot]                                        # [K]
    pa = b.sp[slot, stat_a]                                         # [33]
    if lik_full is None:
        joint = pk[:, None] * pa[None, :] * jnp.maximum(lik_k_a, 0.0)
    else:
        pb = b.sp[slot, stat_b]
        joint = pk[:, None, None] * pa[None, :, None] * pb[None, None, :] * jnp.maximum(lik_full, 0.0)
    z = jnp.sum(joint)
    ok = z > 0
    if lik_full is None:
        new_k, new_a = jnp.sum(joint, axis=1), jnp.sum(joint, axis=0)
    else:
        new_k, new_a, new_b = jnp.sum(joint, axis=(1, 2)), jnp.sum(joint, axis=(0, 2)), jnp.sum(joint, axis=(0, 1))
    zk = jnp.maximum(jnp.sum(new_k), 1e-30)
    lw = jnp.where(new_k > 0, jnp.log(jnp.maximum(new_k / zk, 1e-30)), -jnp.inf)
    lw = jnp.where(ok, lw, b.logw[slot])
    sp = b.sp.at[slot, stat_a].set(jnp.where(ok, (new_a + EPS) / jnp.sum(new_a + EPS), pa))
    if lik_full is not None:
        sp = sp.at[slot, stat_b].set(jnp.where(ok, (new_b + EPS) / jnp.sum(new_b + EPS), b.sp[slot, stat_b]))
    return b.replace(logw=b.logw.at[slot].set(lw), sp=sp, off_prior=b.off_prior.at[slot].set(b.off_prior[slot] | ~ok))



def off_stat(dex, move):
    """Belief stat index (HP = 0) of the attacking stat: overrideOffensiveStat (Body Press) or Atk / SpA by category
    (PS sim/battle-actions.ts getDamage; battle/damage.py raw_stats)."""
    ovr = dex.move_override_off_stat[move].astype(I32)
    return jnp.where(ovr > 0, ovr, jnp.where(dex.move_category[move] == ids.CAT_PHYSICAL, ATK, SPA))


def def_stat(dex, move):
    """Belief stat index of the defending stat: overrideDefensiveStat (Psyshock) or Def / SpD by category."""
    ovr = dex.move_override_def_stat[move].astype(I32)
    return jnp.where(ovr > 0, ovr, jnp.where(dex.move_category[move] == ids.CAT_PHYSICAL, DEF, SPD))

# ------------------------------------------------------------------ hypothetical boards
def _hyp_state(st, dex, pri: Priors, species, opp_ref, k, sp6, nature=None):
    """The real board with the opponent mon `opp_ref` replaced by skeleton k (item, ability, nature) and SP spread
    sp6 (hp kept at the same fraction of the new max hp: the public percentage)."""
    item = pri.skel_item[species, k]
    ab = pri.skel_ability[species, k]
    nat = pri.skel_nature[species, k] if nature is None else nature
    frac_num, frac_den = mget(st.mons.hp, opp_ref).astype(I32), jnp.maximum(mget(st.mons.maxhp, opp_ref).astype(I32), 1)
    st = CA.set_item(CA.set_ability(st, opp_ref, ab), opp_ref, item)
    maxhp, stats = ST.calc_stats(dex.species_base_stats[mget(st.mons.species, opp_ref).astype(I32)], sp6,
                                 dex.nature_plus[nat], dex.nature_minus[nat])
    m = st.mons
    hp = jnp.maximum((maxhp * frac_num + frac_den - 1) // frac_den, jnp.where(frac_num > 0, 1, 0))
    return st.replace(mons=m.replace(stats=mset(m.stats, opp_ref, stats.astype(m.stats.dtype), True),
                                     maxhp=mset(m.maxhp, opp_ref, jnp.asarray(maxhp, m.maxhp.dtype), True),
                                     hp=mset(m.hp, opp_ref, jnp.asarray(hp, m.hp.dtype), True)))


def _sp_vec(b: Belief, slot, stat, v):
    """A spread with `stat` = v and every other stat at its current belief mode."""
    modes = jnp.argmax(b.sp[slot], axis=-1).astype(I32)
    return modes.at[stat].set(v)


def damage_taken(b: Belief, pri: Priors, st, dex, cfg, slot, opp_ref, our_ref, move, crit, observed):
    """The opponent mon (sheet `slot`, battle ref `opp_ref`) hit our mon `our_ref` with `move` (crit known from the
    '-crit' line) for exactly `observed` hp. Grid: skeleton K x attacking-stat SP (Atk / SpA by the move's category)."""
    species = b.species[slot]
    stat = off_stat(dex, move)                     # Body Press: Def; Foul Play reads ours (likelihood flat in v)
    K = pri.skel_w.shape[1]

    def lik(k, v):
        sp6 = _sp_vec(b, slot, stat, v)
        h = _hyp_state(st, dex, pri, species, opp_ref, k, sp6)
        vals, _ = CA.damage_rolls(h, dex, cfg, move, crit, u=opp_ref, t=our_ref)
        return jnp.mean((vals == observed).astype(F32))
    L = jax.vmap(lambda k: jax.vmap(lambda v: lik(k, v))(jnp.arange(NSP)))(jnp.arange(K))
    return _apply_joint(b, slot, L, stat)


def damage_dealt(b: Belief, pri: Priors, st, dex, cfg, slot, opp_ref, our_ref, move, crit, pct_before, pct_after):
    """Our mon hit the opponent mon for a public HP% change pct_before -> pct_after. Grid: skeleton K x HP-SP x
    defensive-stat SP (Def / SpD by category; Psyshock-style moves are not special-cased yet)."""
    species = b.species[slot]
    dstat = def_stat(dex, move)                    # Psyshock / Psystrike: Def
    K = pri.skel_w.shape[1]

    def lik(k, h, d):
        sp6 = _sp_vec(b, slot, dstat, d).at[HP].set(h)
        hs = _hyp_state(st, dex, pri, species, opp_ref, k, sp6)
        mx = mget(hs.mons.maxhp, opp_ref).astype(I32)
        hp0 = (pct_before * mx + 99) // 100                             # an HP consistent with the shown percentage
        vals, _ = CA.damage_rolls(hs, dex, cfg, move, crit, u=our_ref, t=opp_ref)
        after = jnp.maximum(hp0 - vals, 0)
        pct, _ = EV.hp_public(after, mx)
        return jnp.mean((jnp.abs(pct.astype(I32) - pct_after) <= 1).astype(F32))
    L = jax.vmap(lambda k: jax.vmap(lambda h: jax.vmap(lambda d: lik(k, h, d))(jnp.arange(NSP)))(jnp.arange(NSP)))(
        jnp.arange(K))                                                  # [K, hp, def]
    return _apply_joint(b, slot, None, HP, L, dstat)


def speed_order(b: Belief, pri: Priors, st, dex, slot, opp_ref, our_ref, opp_first):
    """Within one priority bracket the opponent mon moved before (opp_first) or after our mon, with no unseen speed
    effects (the caller checks Trick Room / Quick Claw-type cases). Opponent speed per skeleton x Spe-SP via the
    engine's action speed (item: Choice Scarf / Iron Ball, abilities, boosts, paralysis all from the hypothesis)."""
    species = b.species[slot]
    K = pri.skel_w.shape[1]
    ours = C.action_speed(st, dex, our_ref)

    def lik(k, v):
        h = _hyp_state(st, dex, pri, species, opp_ref, k, _sp_vec(b, slot, SPE, v))
        spd = C.action_speed(h, dex, opp_ref)
        return jnp.where(opp_first, spd >= ours, spd <= ours).astype(F32)   # ties: either order possible
    L = jax.vmap(lambda k: jax.vmap(lambda v: lik(k, v))(jnp.arange(NSP)))(jnp.arange(K))
    return _apply_joint(b, slot, L, SPE)


# ------------------------------------------------------------------ features for the policy
def features(b: Belief, pri: Priors, dex):
    """Per opponent sheet slot: skeleton probabilities (K), P(item = each skeleton item) folded into the skeleton
    axis, expected SP per stat (6), SP std (6), expected stats need the species base: f32[6, K + 12 + 1]."""
    pk = skel_probs(b)                                                  # [6, K]
    v = jnp.arange(NSP, dtype=F32)
    mean = jnp.sum(b.sp * v, axis=-1)                                   # [6, 6]
    std = jnp.sqrt(jnp.maximum(jnp.sum(b.sp * v ** 2, axis=-1) - mean ** 2, 0.0))
    return jnp.concatenate([pk, mean / 32.0, std / 16.0, b.off_prior[:, None].astype(F32)], axis=-1)
