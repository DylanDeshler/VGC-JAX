"""08 §4 leak tests: hidden opponent information never changes observe(state, 0); perspective symmetry."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from harness.corpus import iter_battles, shard_files
from vgcjax.config import EngineConfig
from vgcjax.core.dex import load_dex
from vgcjax.env import api
from vgcjax.env.symmetry import swap_sides
from vgcjax.interop.team import team_from_sets

CFG = EngineConfig(stage="core", max_turns=40)


def states(n_games=6, every=3):
    import subprocess, pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    out = root / "scratch" / "leak_src"
    if not out.exists():
        subprocess.run(["node", str(root / "oracle/bin/oracle.js"), "gen", "--n", "20", "--seed", "leak", "--workers", "4",
                        "--teams", "stage:core", "--out", str(out)], check=True, capture_output=True)
    dex = load_dex("core")
    stepf = jax.jit(lambda st, a, k: api.step(st, a, k, CFG, dex))
    maskf = jax.jit(lambda st: api.legal_masks(st, CFG, dex))
    res = []
    for g, b in enumerate(iter_battles(shard_files(out)[0])):
        if g >= n_games:
            break
        st = api.init(None, team_from_sets(b[0]["team_sets"]), CFG, dex)
        key = jax.random.PRNGKey(g)
        for i in range(30):
            key, k1, k2 = jax.random.split(key, 3)
            st, out_ = stepf(st, api.random_actions(k1, maskf(st), st.meta.phase), k2)
            if i % every == 1:
                res.append(st)
            if bool(out_.terminated):
                break
    return dex, res


@pytest.fixture(scope="module")
def corpus():
    return states()


def obs0(st, dex):
    return jax.tree_util.tree_map(np.asarray, api.observe(st, 0, CFG, dex))


def eq(a, b):
    return all(np.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def test_hidden_perturbations_do_not_leak(corpus):
    dex, sts = corpus
    rng = np.random.default_rng(0)
    for st in sts:
        base = obs0(st, dex)
        m = st.mons
        # 1. opponent SP spread and stored stats
        st1 = st.replace(mons=m.replace(sp=m.sp.at[1].set(jnp.asarray(rng.integers(0, 33, (4, 6)), m.sp.dtype)),
                                        stats=m.stats.at[1].add(1)))
        assert eq(obs0(st1, dex), base)
        # 2. opponent exact HP within the same percentage/colour bucket (when possible)
        hp = np.asarray(m.hp[1]); mx = np.asarray(m.maxhp[1])
        for b in range(4):
            for d in (-1, 1):
                h2 = hp[b] + d
                if 0 < h2 <= mx[b] and hp[b] > 0:
                    from vgcjax.env.obs import hp_public
                    if tuple(map(int, hp_public(h2, mx[b]))) == tuple(map(int, hp_public(hp[b], mx[b]))):
                        st2 = st.replace(mons=m.replace(hp=m.hp.at[1, b].set(h2)))
                        assert eq(obs0(st2, dex), base)
        # 3. opponent PP
        st3 = st.replace(mons=m.replace(pp=m.pp.at[1].set(0)))
        assert eq(obs0(st3, dex), base)
        # the same perturbations on side 0 change the observation (own info is visible)
        st4 = st.replace(mons=m.replace(stats=m.stats.at[0].add(1)))
        assert not eq(obs0(st4, dex), base)


def test_perspective_symmetry(corpus):
    dex, sts = corpus
    for st in sts:
        a = jax.tree_util.tree_map(np.asarray, api.observe(st, 1, CFG, dex))
        b = obs0(swap_sides(st), dex)
        assert eq(a, b)
