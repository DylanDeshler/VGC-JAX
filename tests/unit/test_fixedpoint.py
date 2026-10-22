"""M1.2: fixedpoint.py equals the PS `fixedpoint` RPC on random cases (expectations from PS only)."""
import numpy as np
import pytest

from harness.oracle_client import chunks, default_client
from vgcjax.core import fixedpoint as fp

N = 250_000  # x4 ops = 1e6 cases (10 M1.2)


def _cases(rng):
    vals = rng.integers(0, 65536, N)
    # domain of real mechanics: modifiers <= x4, so value * m4096 < 2**31 (see vgcjax/core/fixedpoint.py)
    dens = rng.choice([1, 2, 3, 4, 10, 100, 4096], N)
    nums = rng.integers(1, 4 * dens + 1)
    big = rng.integers(0, 2**20, N)
    m1 = rng.integers(1, 16384, N)
    m2 = rng.integers(1, 16384, N)
    return vals, nums, dens, big, m1, m2


@pytest.fixture(scope="module")
def ps():
    rng = np.random.default_rng(12345)
    vals, nums, dens, big, m1, m2 = _cases(rng)
    ops = []
    for i in range(N):
        ops.append(["modify", int(vals[i]), int(nums[i]), int(dens[i])])
        ops.append(["trunc", int(big[i]) * 7, 16])
        ops.append(["chain", [[int(m1[i]), 4096], [int(m2[i]), 4096]]])
        ops.append(["chainFinal", int(vals[i]) % 16384, [[int(m1[i]), 4096], [int(nums[i]), int(dens[i])]]])
    oc = default_client()
    out = []
    for part in oc.map("fixedpoint", [{"ops": c} for c in chunks(ops, 50_000)]):
        out += part
    res = np.array(out, dtype=np.int64).reshape(N, 4)
    return (vals, nums, dens, big, m1, m2), res


def test_modify(ps):
    (vals, nums, dens, *_), res = ps
    np.testing.assert_array_equal(np.asarray(fp.modify(vals, nums, dens)), res[:, 0])


def test_trunc16(ps):
    (_, _, _, big, _, _), res = ps
    np.testing.assert_array_equal(np.asarray(fp.trunc16(big * 7)), res[:, 1])


def test_chain(ps):
    (_, _, _, _, m1, m2), res = ps
    got = fp.chain(fp.chain(4096, fp.mod4096(m1, 4096)), fp.mod4096(m2, 4096))
    np.testing.assert_array_equal(np.asarray(got), res[:, 2])


def test_chain_final(ps):
    (vals, nums, dens, _, m1, _), res = ps
    m = fp.chain_modify(fp.chain_modify(4096, m1, 4096), nums, dens)
    np.testing.assert_array_equal(np.asarray(fp.final_modify(vals % 16384, m)), res[:, 3])
