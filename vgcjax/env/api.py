"""RL-facing API (08 §1-5): pure, single-battle functions; vmap outside."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from vgcjax.battle import init as INIT
from vgcjax.battle import requests as REQ
from vgcjax.battle import state as S
from vgcjax.battle import step as STEP
from vgcjax.core import rng as R
from vgcjax.env import actions as A
from vgcjax.env import obs as O

F32 = jnp.float32
I32 = jnp.int32


@struct.dataclass
class StepOut:
    obs: O.Obs                 # stacked over players [2, ...]
    final_obs: O.Obs           # == obs except on auto-reset steps (env/autoreset.py)
    reward: jax.Array          # f32[2], zero-sum terminal only
    terminated: jax.Array      # bool
    truncated: jax.Array       # bool
    has_decision: jax.Array    # bool[2]
    phase: jax.Array           # i8
    error_flags: jax.Array     # u32
    shaping: jax.Array         # f32[2,4]: HP-fraction sum, mons remaining, fainted, mega used


def init(key, team: S.Team, cfg, dex) -> S.State:
    """State at the PREVIEW decision. Cheap (no hooks) so select-based auto-reset can run it every step (08 §6)."""
    del key, cfg, dex
    return INIT.init(team)


def preview_index(order):
    """i32 index into PREVIEW_ORDERS of an ordered 4-tuple of sheet indices."""
    tab = jnp.asarray(A.PREVIEW_ORDERS, I32)
    return jnp.argmax(jnp.all(tab == jnp.asarray(order, I32)[None, :], axis=1))


def init_from_order(key, team, order, cfg, dex) -> S.State:
    """Skip preview: both sides' ordered 4-of-6 (sheet indices, i32[2,4])."""
    st = init(key, team, cfg, dex)
    acts = jnp.stack([preview_index(order[0]), preview_index(order[1])])
    st, _ = STEP.step(st, acts, key, cfg, dex)
    return st


def legal_masks(state, cfg, dex):
    """Masks {slot: bool[2,2,29], pair_ok: bool[2,29,29], preview: bool[2,360]} (all-true when DONE; 08 §1)."""
    mk = REQ.legal_masks(state, dex)
    if cfg.preview_canonical:
        prev = jnp.stack([A.preview_mask(state.team.ability[s], True) for s in range(2)])
        mk = dict(mk, preview=mk["preview"] & prev)
    done = state.meta.phase == S.DONE
    return {k: jnp.where(done, True, v) for k, v in mk.items()}


def has_decision(state, masks):
    preview = state.meta.phase == S.PREVIEW
    out = []
    for s in range(2):
        nj = jnp.sum(A.joint_mask(masks, s).astype(I32))
        np_ = jnp.sum(masks["preview"][s].astype(I32))
        out.append(jnp.where(preview, np_ > 1, nj > 1))
    return jnp.stack(out) & (state.meta.phase != S.DONE)


def observe(state, player, cfg, dex, masks=None) -> O.Obs:
    if masks is None:
        masks = legal_masks(state, cfg, dex)
    return O.observe(state, player, cfg, masks)


def shaping(state):
    m = state.mons
    hpf = jnp.sum(m.hp.astype(F32) / jnp.maximum(m.maxhp.astype(F32), 1), axis=1)
    left = state.sides.pokemon_left.astype(F32)
    fainted = state.sides.total_fainted.astype(F32)
    mega = state.sides.mega_used.astype(F32)
    return jnp.stack([hpf, left, fainted, mega], axis=1)


def outcome(state, cfg):
    """(reward f32[2], terminated, truncated) per 08 §5."""
    meta = state.meta
    done = meta.phase == S.DONE
    cap = done & (meta.ended_reason == S.END_TURN_CAP)
    w = meta.winner.astype(I32)
    # tiebreak at the cap: remaining mons, then HP-fraction sum, then draw
    sh = shaping(state)
    left = sh[:, 1]
    hpf = sh[:, 0]
    tb = jnp.where(left[0] != left[1], jnp.where(left[0] > left[1], 0, 1),
                   jnp.where(hpf[0] != hpf[1], jnp.where(hpf[0] > hpf[1], 0, 1), 2))
    if cfg.turn_cap_outcome == "tiebreak":
        w = jnp.where(cap, tb, w)
    elif cfg.turn_cap_outcome == "draw":
        w = jnp.where(cap, 2, w)
    win = jnp.stack([w == 0, w == 1])
    lose = jnp.stack([w == 1, w == 0])
    if cfg.reward_01:
        r = win.astype(F32)
    else:
        r = win.astype(F32) - lose.astype(F32)
    truncate = cap & (cfg.turn_cap_outcome == "truncate")
    r = jnp.where(done & ~truncate, r, 0.0)
    return r, done & ~truncate, truncate


def make_stepout(state, cfg, dex, final_obs=None) -> StepOut:
    masks = legal_masks(state, cfg, dex)
    obs = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), observe(state, 0, cfg, dex, masks),
                                 observe(state, 1, cfg, dex, masks))
    r, term, trunc = outcome(state, cfg)
    return StepOut(obs=obs, final_obs=obs if final_obs is None else final_obs, reward=r, terminated=term,
                   truncated=trunc, has_decision=has_decision(state, masks), phase=state.meta.phase,
                   error_flags=state.meta.error_flags, shaping=shaping(state))


def step(state, actions, key, cfg, dex, forced: R.ForcedOutcomes | None = None):
    """Apply both players' actions (08 §1). DONE states are a no-op."""
    done = state.meta.phase == S.DONE
    st2, _ = STEP.step(state, actions, key, cfg, dex, forced)
    st2 = jax.tree_util.tree_map(lambda a, b: jnp.where(done, b, a), st2, state)
    return st2, make_stepout(st2, cfg, dex)


def random_actions(key, masks, phase):
    """Uniform legal actions for both players (testing / baselines)."""
    out = []
    for s in range(2):
        k = jax.random.fold_in(key, s)
        jm = A.joint_mask(masks, s)
        pj = jax.random.categorical(k, jnp.where(jm, 0.0, -jnp.inf))
        pp = jax.random.categorical(k, jnp.where(masks["preview"][s], 0.0, -jnp.inf))
        out.append(jnp.where(phase == S.PREVIEW, pp, pj))
    return jnp.stack(out).astype(I32)
