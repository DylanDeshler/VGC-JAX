"""Auto-reset env (08 §6): select-based reset when a battle ends, final_obs, EpisodeStats."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from vgcjax.battle import state as S
from vgcjax.env import api
from vgcjax.env import teambank as TB


@struct.dataclass
class EnvState:
    state: S.State
    team_ids: jax.Array        # i32[2]
    opp_policy_id: jax.Array   # i32
    steps: jax.Array           # i32 env steps in this episode
    preview_actions: jax.Array  # i32[2]


class EpisodeStats(NamedTuple):   # valid where `done`
    done: jax.Array
    winner: jax.Array
    turns: jax.Array
    steps: jax.Array
    end_reason: jax.Array
    team_ids: jax.Array
    opp_policy_id: jax.Array
    preview_actions: jax.Array
    megas: jax.Array
    remaining: jax.Array
    hp_frac: jax.Array
    error_flags: jax.Array


class AutoResetEnv:
    def __init__(self, cfg, dex, bank: TB.TeamBank):
        self.cfg, self.dex, self.bank = cfg, dex, bank

    def _fresh(self, key, opp_policy_id):
        team, ids = TB.sample_team_pair(key, self.bank)
        st = api.init(key, team, self.cfg, self.dex)
        return EnvState(state=st, team_ids=ids, opp_policy_id=jnp.asarray(opp_policy_id, jnp.int32),
                        steps=jnp.zeros((), jnp.int32), preview_actions=jnp.full(2, -1, jnp.int32))

    def reset(self, key, opp_policy_id=0):
        es = self._fresh(key, opp_policy_id)
        return es, api.make_stepout(es.state, self.cfg, self.dex)

    def step(self, es: EnvState, actions, key):
        k_step, k_reset = jax.random.split(key)
        was_preview = es.state.meta.phase == S.PREVIEW
        st, out = api.step(es.state, actions, k_step, self.cfg, self.dex)
        done = out.terminated | out.truncated
        pa = jnp.where(was_preview, jnp.asarray(actions, jnp.int32), es.preview_actions)
        sh = out.shaping
        stats = EpisodeStats(done=done, winner=st.meta.winner, turns=st.meta.turn, steps=es.steps + 1,
                             end_reason=st.meta.ended_reason, team_ids=es.team_ids, opp_policy_id=es.opp_policy_id,
                             preview_actions=pa, megas=st.sides.mega_used.astype(jnp.int8),
                             remaining=st.sides.pokemon_left, hp_frac=sh[:, 0], error_flags=st.meta.error_flags)
        cont = EnvState(state=st, team_ids=es.team_ids, opp_policy_id=es.opp_policy_id, steps=es.steps + 1,
                        preview_actions=pa)
        fresh = self._fresh(k_reset, es.opp_policy_id)
        nxt = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), fresh, cont)
        out_new = api.make_stepout(nxt.state, self.cfg, self.dex, final_obs=out.obs)
        out_new = out_new.replace(reward=out.reward, terminated=out.terminated, truncated=out.truncated)
        return nxt, out_new, stats
