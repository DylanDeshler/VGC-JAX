"""`Dex`: the traced pytree of dense integer tables (ADR-006, 03 §2), loaded from generated `vgcjax/data/tables.npz`.

Stage support masks are selected at load time (`supported_<stage>_*` -> `supported_*`). Shapes are static per build.
"""
from __future__ import annotations

import functools
import pathlib

import jax.numpy as jnp
import numpy as np
from flax import struct

from vgcjax.data import ids

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
STAGES = ("core", "v0", "v1", "v2")


@struct.dataclass
class Dex:
    tables: dict  # name -> jnp array (see tools/datagen/codegen.py for the schema)

    def __getattr__(self, name):
        try:
            return self.tables[name]
        except KeyError:
            raise AttributeError(name) from None


@functools.lru_cache(maxsize=None)
def load_tables_np() -> dict[str, np.ndarray]:
    """All generated tables as numpy arrays (host)."""
    with np.load(DATA_DIR / "tables.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_dex(stage: str = "core") -> Dex:
    """Dex for `stage`; checks the generated data matches `ids.DATA_VERSION` (06 §1)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage}")
    t = load_tables_np()
    dv = bytes(t["data_version"]).decode()
    if dv != ids.DATA_VERSION:
        raise RuntimeError(f"tables.npz DATA_VERSION {dv} != ids.py {ids.DATA_VERSION}; re-run tools/datagen/codegen.py")
    out = {}
    for k, v in t.items():
        if k == "data_version":
            continue
        if k.startswith("supported_"):
            st, _, kind = k[len("supported_"):].partition("_")
            if st == stage:
                out[f"supported_{kind}"] = jnp.asarray(v)
            continue
        out[k] = jnp.asarray(v)
    return Dex(tables=out)


def data_version() -> str:
    return ids.DATA_VERSION
