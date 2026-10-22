"""M1.6: counter-based RNG — hash reference, distributions (chi-square), determinism under vmap, forced and log modes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as sstats

from vgcjax.battle import errors as E
from vgcjax.core import drawkey as dk
from vgcjax.core import rng as R
from vgcjax.data import ids


def key(**kw):
    base = dict(kind=ids.DK_DAMAGE_ROLL, turn=3, step_tag=ids.STEP_B_MOVE, step_actor=1, step_subject=57, actor=1,
                target=5, subject=57, hit=1, ord=0, sub=0, action_ord=0)
    base.update(kw)
    return dk.DrawKey(**base)


def test_pack_roundtrip_and_minus_one():
    k = key(actor=-1, target=-1, step_actor=-1)
    c = dk.pack(k)
    u = dk.unpack(*c)
    assert u["actor"] == -1 and u["target"] == -1 and u["step_actor"] == -1
    assert u["kind"] == ids.DK_DAMAGE_ROLL and u["subject"] == 57 and u["turn"] == 3
    cj = dk.pack(k, jnp)
    assert tuple(int(x) for x in cj) == c


def test_hash_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = [int(x) for x in rng.integers(0, 2**32, 2, dtype=np.uint64)]
        c = [int(x) for x in rng.integers(0, 2**30, 3, dtype=np.uint64)]
        got = int(R.hash32(jnp.array(w, jnp.uint32), [jnp.uint32(x) for x in c]))
        assert got == dk.hash32_int(w[0], w[1], *c)


def _draws(fn, n, seed=0):
    """n natural draws with distinct keys (vary turn/sub/subject) from one step."""
    r = R.make_rng(jax.random.PRNGKey(seed), 0)
    i = jnp.arange(n)
    ks = key(turn=i % 2048, sub=(i // 2048) % 256, subject=(i // (2048 * 256)) % 4096)
    ks = dk.DrawKey(*(jnp.broadcast_to(jnp.asarray(f), (n,)) for f in ks))
    return np.asarray(jax.vmap(lambda k: fn(r, k)[0])(ks))


@pytest.mark.parametrize("d", [2, 3, 8, 16, 100])
def test_randint_uniform(d):
    x = _draws(lambda r, k: R.randint(r, k, 0, d), 400_000)
    counts = np.bincount(x, minlength=d)
    assert counts.sum() == 400_000 and x.min() >= 0 and x.max() < d
    assert sstats.chisquare(counts).pvalue > 1e-3


@pytest.mark.parametrize("n,d", [(1, 8), (1, 4), (3, 10), (33, 100), (1, 24)])
def test_chance_rate(n, d):
    x = _draws(lambda r, k: R.chance(r, k, n, d), 400_000)
    p = n / d
    k = int(x.sum())
    assert sstats.binomtest(k, len(x), p).pvalue > 1e-3


@pytest.mark.slow
def test_hash_uniform_1e7():
    x = _draws(lambda r, k: R.randint(r, k, 0, 256), 10_000_000)
    assert sstats.chisquare(np.bincount(x, minlength=256)).pvalue > 1e-3


def test_vmap_determinism_and_key_sensitivity():
    keys = jax.random.split(jax.random.PRNGKey(1), 64)
    f = lambda k: R.randint(R.make_rng(k, 5), key(), 0, 16)[0]  # noqa: E731
    a = np.asarray(jax.vmap(f)(keys))
    b = np.asarray([int(f(k)) for k in keys])
    np.testing.assert_array_equal(a, b)
    assert len(set(a.tolist())) > 8
    # same key, different step counter -> different words
    r0 = R.make_rng(keys[0], 0); r1 = R.make_rng(keys[0], 1)
    assert not np.array_equal(np.asarray(r0.w), np.asarray(r1.w))


def test_degenerate_draws():
    r = R.make_rng(jax.random.PRNGKey(0), 0, log=R.DrawLog.empty(8))
    v, r = R.chance(r, key(), 100, 100)
    assert bool(v)
    v, r = R.randint(r, key(kind=ids.DK_PARTIAL_TRAP_DURATION), 5, 6)
    assert int(v) == 5
    v, r = R.sample(r, key(kind=ids.DK_SLEEP_DURATION), 1)
    assert int(v) == 0
    assert int(r.log.count) == 0


def test_forced_mode_and_log():
    k1, k2 = key(kind=ids.DK_CRIT, hit=1), key(kind=ids.DK_DAMAGE_ROLL)
    f = R.ForcedOutcomes.empty(4)
    f = R.ForcedOutcomes(keys=f.keys.at[0].set(jnp.stack(dk.pack(k1, jnp))).at[1].set(jnp.stack(dk.pack(k2, jnp))),
                         value=f.value.at[0].set(1).at[1].set(15), valid=f.valid.at[0].set(True).at[1].set(True))
    r = R.make_rng(jax.random.PRNGKey(0), 0, forced=f, log=R.DrawLog.empty(8))
    crit, r = R.chance(r, k1, 1, 24)
    roll, r = R.randint(r, k2, 0, 16)
    assert bool(crit) and int(roll) == 15 and int(r.err) == 0
    _, r = R.randint(r, key(kind=ids.DK_ACCURACY), 0, 16, active=False)  # inactive: no error, no log
    assert int(r.err) == 0 and int(r.log.count) == 2
    _, r = R.chance(r, key(kind=ids.DK_ACCURACY), 90, 100)  # missing forced key
    assert int(r.err) & E.ERR_FORCED_MISSING and int(r.log.count) == 3
    assert int(r.log.flags[0]) & R.LOG_USED_FORCED and not int(r.log.flags[2]) & R.LOG_USED_FORCED
    assert dk.unpack(*map(int, r.log.keys[1]))["kind"] == ids.DK_DAMAGE_ROLL
    np.testing.assert_array_equal(np.asarray(r.log.args[0]), [1, 24])


def test_sample_ref_and_tie_order():
    r = R.make_rng(jax.random.PRNGKey(3), 0)
    refs = jnp.array([4, 5, 6, 7]); valid = jnp.array([True, False, True, True])
    picks = set()
    for t in range(60):
        v, _ = R.sample_ref(r, key(kind=ids.DK_RANDOM_TARGET, turn=t), refs, valid)
        picks.add(int(v))
    assert picks == {4, 6, 7}
    # tie_order: forced permutation rank round-trips; natural orders cover all permutations of 3
    members = jnp.array([0, 1, 5, 6]); valid = jnp.array([True, True, True, False])
    seen = set()
    for t in range(200):
        order, _ = R.tie_order(r, key(kind=ids.DK_SPEED_TIE_FIELD_EVENT, turn=t), members, valid)
        o = tuple(int(x) for x in order[:3])
        seen.add(o)
        assert int(order[3]) == 3
    assert len(seen) == 6
    for rank in range(6):
        kk = key(kind=ids.DK_SPEED_TIE_SWITCHIN)
        f = R.ForcedOutcomes(keys=jnp.zeros((1, 3), jnp.uint32).at[0].set(jnp.stack(dk.pack(kk, jnp))),
                             value=jnp.array([rank]), valid=jnp.array([True]))
        rr = R.make_rng(jax.random.PRNGKey(0), 0, forced=f, log=R.DrawLog.empty(2))
        order, rr = R.tie_order(rr, kk, members, valid)
        assert int(rr.log.value[0]) == rank and int(rr.err) == 0
    # select: forced winner
    kk = key(kind=ids.DK_SPEED_TIE_SELECT)
    f = R.ForcedOutcomes(keys=jnp.zeros((1, 3), jnp.uint32).at[0].set(jnp.stack(dk.pack(kk, jnp))),
                         value=jnp.array([5]), valid=jnp.array([True]))
    rr = R.make_rng(jax.random.PRNGKey(0), 0, forced=f, log=R.DrawLog.empty(2))
    order, rr = R.tie_order(rr, kk, members, valid, select=True)
    assert int(members[order[0]]) == 5 and int(rr.log.value[0]) == 5
