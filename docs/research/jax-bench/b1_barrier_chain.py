"""k chained N-way batched switches, with/without lax.optimization_barrier between them."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp
from jax import lax
from common import bench
from handlers import make_branches, random_ctx
B=16384
ctx = random_ctx(jax.random.PRNGKey(0), B)
def chain(n, k, barrier):
    brs = [make_branches(n, seed=s) for s in range(k)]
    def one(c, ids):
        for j in range(k):
            c = lax.switch(ids[j], brs[j], c)
            if barrier: c = lax.optimization_barrier(c)
        return c
    return lambda ctx, ids: jax.vmap(one)(ctx, ids)
for n in (32, 65):
    for k in (1, 2, 3, 4):
        ids = jax.random.randint(jax.random.PRNGKey(1), (B, k), 0, n)
        for bar in (False, True):
            r = bench(chain(n, k, bar), ctx, ids, B=B)
            print(f"n={n} k={k} barrier={bar} compile={r['compile_s']:.2f}s {r['ns_per_elem']:.1f} ns/elem", flush=True)
