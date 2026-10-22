"""Side swap of a State (for perspective-symmetry tests, 08 §4 leak test 2, and seat randomisation checks)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from vgcjax.battle import state as S


def _flip_ref(r):
    r = jnp.asarray(r)
    return jnp.where(r >= 0, (r + 4) % 8, r).astype(r.dtype)


def swap_sides(st: S.State) -> S.State:
    flip0 = lambda x: x[::-1] if x.ndim >= 1 and x.shape[0] == 2 else x  # noqa: E731
    team = jax.tree_util.tree_map(flip0, st.team)
    sides = jax.tree_util.tree_map(flip0, st.sides)
    mons = jax.tree_util.tree_map(flip0, st.mons)
    meta = st.meta
    w = meta.winner
    meta = meta.replace(winner=jnp.where((w == 0) | (w == 1), 1 - w, w).astype(w.dtype),
                        faint_queue=_flip_ref(meta.faint_queue))
    q = st.queue
    q = q.replace(side=(1 - q.side).astype(q.side.dtype), original_target=_flip_ref(q.original_target))
    sc = st.scratch
    sc = sc.replace(move_actions=sc.move_actions[::-1], step_actor=_flip_ref(sc.step_actor))
    return st.replace(team=team, sides=sides, mons=mons, meta=meta, queue=q, scratch=sc)
