"""Shared timing helpers for the JAX VGC engine micro-benchmarks.

Every benchmark measures:
  * trace+lower time (Python tracing -> StableHLO),
  * XLA compile time (HLO -> executable),
  * steady-state runtime (median of R runs after warmup), reported per batch element.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import sys
import time

import jax
import jax.numpy as jnp

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def env_info():
    return {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }


def compile_fn(fn, *args, jit_kwargs=None):
    """Return (compiled, t_lower_s, t_compile_s). fn must be a fresh closure so nothing is cached."""
    jit_kwargs = jit_kwargs or {}
    t0 = time.perf_counter()
    lowered = jax.jit(fn, **jit_kwargs).lower(*args)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    return compiled, t1 - t0, t2 - t1


def time_compiled(compiled, *args, reps=None, min_time=0.5, warmup=2):
    """Median wall time (s) of compiled(*args), blocking on the result."""
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    # estimate reps
    if reps is None:
        t = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        one = max(time.perf_counter() - t, 1e-6)
        reps = int(min(50, max(3, min_time / one)))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def bench(fn, *args, B=None, jit_kwargs=None, **kw):
    compiled, tl, tc = compile_fn(fn, *args, jit_kwargs=jit_kwargs)
    rt = time_compiled(compiled, *args, **kw)
    out = {"lower_s": tl, "compile_s": tc, "run_s": rt}
    if B:
        out["ns_per_elem"] = rt / B * 1e9
    try:
        ma = compiled.memory_analysis()
        if ma is not None:
            out["temp_bytes"] = int(ma.temp_size_in_bytes)
            out["arg_bytes"] = int(ma.argument_size_in_bytes)
            out["out_bytes"] = int(ma.output_size_in_bytes)
    except Exception:  # noqa: BLE001
        pass
    return out


def peak_rss_mb():
    # macOS reports bytes, Linux reports KiB
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 20) if sys.platform == "darwin" else r / 1024


def save(name, payload):
    payload = {"env": env_info(), **payload}
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[saved {path}]")


def md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            f = fmt.get(c)
            if f and isinstance(v, (int, float)):
                cells.append(format(v, f))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def i32(x):
    return jnp.asarray(x, jnp.int32)
