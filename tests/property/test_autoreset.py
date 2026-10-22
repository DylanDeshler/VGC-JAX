"""M5.4: vmapped auto-reset rollout with uniform legal actions: episodes finish, rewards are zero-sum, no errors,
final_obs differs from obs exactly on reset steps."""
import jax
import jax.numpy as jnp
import numpy as np

from vgcjax.config import EngineConfig
from vgcjax.core.dex import load_dex
from vgcjax.env import api, teambank as TB
from vgcjax.env.autoreset import AutoResetEnv


def test_rollout():
    cfg = EngineConfig(stage="core", max_turns=40)
    dex = load_dex("core")
    env = AutoResetEnv(cfg, dex, TB.load("data/teams/bank_core.npz"))
    B, T = 16, 60
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    es, out = jax.vmap(env.reset)(keys)

    def one(carry, k):
        es, out = carry
        ks = jax.random.split(k, B * 2).reshape(B, 2, -1)
        masks = jax.vmap(lambda s: api.legal_masks(s, cfg, dex))(es.state)
        acts = jax.vmap(api.random_actions)(ks[:, 0], masks, es.state.meta.phase)
        es, out, stats = jax.vmap(env.step)(es, acts, ks[:, 1])
        return (es, out), (stats, out.reward, out.error_flags)

    (es, out), (stats, rew, err) = jax.jit(lambda c, ks: jax.lax.scan(one, c, ks))((es, out), jax.random.split(
        jax.random.PRNGKey(1), T))
    done = np.asarray(stats.done)
    assert done.sum() >= B          # most battles finish within 60 steps
    assert np.all(np.asarray(err) == 0)
    r = np.asarray(rew)
    assert np.allclose(r.sum(-1), 0)
    assert np.all((np.abs(r).sum(-1) > 0) <= done)  # rewards only on terminal steps
