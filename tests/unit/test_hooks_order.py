"""M3.1: engine handler keys + comparePriority order equal PS resolvePriority/speedSort on synthetic handler sets."""
import json
import pathlib
import random
import re

import jax.numpy as jnp
import numpy as np

from harness.oracle_client import default_client
from vgcjax.core.dex import load_dex
from vgcjax.data import ids
from vgcjax.effects import hooks as H

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPACES = {"move": (ids.MOVES, ids.MOVE_BASE), "ability": (ids.ABILITIES, ids.ABILITY_BASE),
          "item": (ids.ITEMS, ids.ITEM_BASE), "condition": (ids.CONDITIONS, ids.COND_BASE)}
RX = re.compile(r"^on(" + "|".join(p for p in ids.PREFIXES if p) + r")?([A-Z]\w*)$")
HOLDER = {"STATUS": "pokemon", "VOL": "pokemon", "SLOT": "slot", "SIDE": "side", "PW": "field", "WEATHER": "field",
          "TERRAIN": "field"}


def pool():
    ho = json.loads((ROOT / "data/export/handler_order.json").read_text())
    out = []
    for kind, (names, base) in SPACES.items():
        for eid, cbs in ho[kind].items():
            if eid not in names:
                continue
            for cb in cbs:
                m = RX.match(cb)
                if not m or m.group(2) not in ids.HOOK_NAMES or "SwitchIn" in cb:
                    continue
                holder = "pokemon"
                if kind == "condition":
                    ci = names.index(eid)
                    k = next(k for k in ids.COND_KINDS if getattr(ids, f"{k}_BASE") <= ci < getattr(ids, f"{k}_END"))
                    holder = HOLDER[k]
                out.append(dict(kind=kind, id=eid, callback=cb, holder=holder, eff=base + names.index(eid) - 1,
                                hook=ids.HOOK_NAMES.index(m.group(2)), pfx=ids.PREFIXES.index(m.group(1) or "")))
    return out


def test_keys_and_order_match_ps():
    rng = random.Random(4)
    P = pool()
    dex = load_dex("v2")
    sets = []
    for _ in range(300):
        n = rng.randint(2, 8)
        s = []
        for _ in range(n):
            h = dict(rng.choice(P))
            h["speed"] = rng.choice([50, 100, 100, 150]) if h["holder"] in ("pokemon", "slot") else 0
            s.append(h)
        sets.append(s)
    res = default_client().call("resolve_priority", {"sets": [[{k: h[k] for k in ("kind", "id", "callback", "holder",
                                                                                    "speed")} for h in s] for s in sets]})
    for s, r in zip(sets, res):
        eff = jnp.asarray([h["eff"] for h in s]); hook = jnp.asarray([h["hook"] for h in s])
        pfx = jnp.asarray([h["pfx"] for h in s]); spd = jnp.asarray([h["speed"] for h in s])
        keys = H.handler_key(dex, eff, hook, pfx, spd, 0, holder_stored_spe=r["storedSpe"])
        o, p, sp, sub, eo = (np.asarray(x) for x in keys)
        for i, k in enumerate(r["keys"]):
            assert (o[i] if o[i] != ids.ORDER_LAST else None) == k["order"], (s[i], k)
            assert (p[i], sub[i], sp[i]) == (k["priority"], k["subOrder"], k["speed"]), (s[i], k)
        perm, group = H.sort_order(keys, jnp.ones(len(s), bool))
        perm = np.asarray(perm).tolist(); group = np.asarray(group).tolist()
        # same order up to permutations inside PS tie groups (which PS shuffles)
        tied = {i for g in r["groups"] for i in g}
        assert [i for i in perm if i not in tied] == [i for i in r["order"] if i not in tied], (s, r)
        eng_groups = {}
        for pos, g in enumerate(group):
            eng_groups.setdefault(g, []).append(perm[pos])
        eng = sorted(sorted(v) for v in eng_groups.values() if len(v) > 1)
        assert eng == sorted(sorted(g) for g in r["groups"]), json.dumps([s, r])
