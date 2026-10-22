"""TeamBank (08 §6, 06 §4.3): device-resident validated teams; pair sampling for resets."""
from __future__ import annotations

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from vgcjax.battle.state import Team
from vgcjax.interop.team import team_from_sets

K_SP = 4


@struct.dataclass
class TeamBank:
    species: jax.Array   # i16[N,6]
    item: jax.Array
    ability: jax.Array
    nature: jax.Array    # i8[N,6]
    gender: jax.Array
    moves: jax.Array     # i16[N,6,4]
    sp_cand: jax.Array   # i8[N,6,K,6] candidate SP spreads
    sp_logit: jax.Array  # f32[N,6,K] prior weights (log)
    valid: jax.Array     # bool[N]

    @property
    def n(self):
        return self.species.shape[0]


def from_jsonl(path) -> TeamBank:
    """Build from JSON lines {packed, sets[6]} (exact SP: one candidate, the rest masked with -inf)."""
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l]
    teams = [team_from_sets([r["sets"], r["sets"]]) for r in rows]
    n = len(rows)
    g = lambda name: np.stack([np.asarray(getattr(t, name))[0] for t in teams])  # noqa: E731
    sp = g("sp")
    sp_cand = np.zeros((n, 6, K_SP, 6), np.int8)
    sp_cand[:, :, 0] = sp
    logit = np.full((n, 6, K_SP), -np.inf, np.float32)
    logit[:, :, 0] = 0.0
    return TeamBank(species=g("species"), item=g("item"), ability=g("ability"), nature=g("nature"), gender=g("gender"),
                    moves=g("moves"), sp_cand=sp_cand, sp_logit=logit, valid=np.ones(n, bool))


def save(bank: TeamBank, path):
    np.savez_compressed(path, **{k: np.asarray(getattr(bank, k)) for k in bank.__dataclass_fields__})


def load(path) -> TeamBank:
    with np.load(path) as z:
        return TeamBank(**{k: jnp.asarray(z[k]) for k in z.files})


def sample_team_pair(key, bank: TeamBank, no_mirror=True):
    """(Team, ids i32[2]): two bank teams (uniform over valid), SP spreads sampled from the candidates."""
    k1, k2, k3 = jax.random.split(key, 3)
    logits = jnp.where(bank.valid, 0.0, -jnp.inf)
    a = jax.random.categorical(k1, logits)
    lb = jnp.where(no_mirror, logits.at[a].set(-jnp.inf), logits)
    b = jax.random.categorical(k2, lb)
    ids = jnp.stack([a, b]).astype(jnp.int32)
    spk = jax.random.categorical(k3, bank.sp_logit[ids], axis=-1)            # [2,6]
    sp = jnp.take_along_axis(bank.sp_cand[ids], spk[..., None, None], axis=2)[:, :, 0]
    team = Team(species=bank.species[ids], item=bank.item[ids], ability=bank.ability[ids], nature=bank.nature[ids],
                gender=bank.gender[ids], moves=bank.moves[ids], sp=sp)
    return team, ids
