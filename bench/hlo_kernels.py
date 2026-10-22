"""Kernel-like ops per env step (09 §1.1a CPU proxy budget, ≤ 10,000), ported from research/jax-bench/hlo_kernels.py.

Compiles the same vmapped auto-reset rollout step as bench/throughput.py and counts top-level (non-fused) HLO
instructions in the optimised module (fusion, gather, scatter, dynamic-update-slice, sort, copy, reduce, ...),
multiplying while-loop bodies by their known trip count and recursing into call / conditional computations. On GPU
every fusion is one kernel, so this approximates launches per step; it does not depend on machine load.

    uv run python bench/hlo_kernels.py --stage core [--batch 256] [--dump out.hlo.txt]
"""
import argparse
import collections
import re

import jax

from vgcjax.config import EngineConfig
from vgcjax.core.dex import load_dex
from vgcjax.env import api, teambank as TB
from vgcjax.env.autoreset import AutoResetEnv

FREE = {"parameter", "constant", "get-tuple-element", "tuple", "bitcast", "after-all", "partition-id", "replica-id",
        "opt-barrier", "iota"}


def parse(text):
    comps, cur, entry = {}, None, None
    for line in text.splitlines():
        if not line.startswith(" ") and "{" in line and ("%" in line.split("(")[0] or line.startswith("ENTRY")):
            name = re.search(r"%?([\w.\-]+)\s*\(", line.replace("ENTRY ", ""))
            cur = name.group(1)
            comps[cur] = []
            if line.startswith("ENTRY"):
                entry = cur
            continue
        if line.startswith("}"):
            cur = None
            continue
        if cur is not None and "=" in line:
            m = re.match(r"\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*.*?\s([a-z][\w\-]*)\(", line)
            if m:
                comps[cur].append((m.group(1), line))
    return comps, entry


def count(comps, name, hist, mult=1):
    total = 0
    for op, line in comps[name]:
        if op == "while":
            body = re.search(r"body=%?([\w.\-]+)", line).group(1)
            cond = re.search(r"condition=%?([\w.\-]+)", line).group(1)
            n = re.search(r'"known_trip_count":\{"n":"(\d+)"', line)
            n = int(n.group(1)) if n else 1
            total += n * (count(comps, body, hist, mult * n) + count(comps, cond, hist, mult * n))
        elif op == "conditional":
            brs = re.findall(r"%?([\w.\-]+)", re.search(r"branch_computations=\{([^}]*)\}", line).group(1))
            total += 1 + max(count(comps, b, hist, mult) for b in brs)
        elif op == "call":
            total += count(comps, re.search(r"to_apply=%?([\w.\-]+)", line).group(1), hist, mult)
        elif op not in FREE:
            total += 1
            hist[op] += mult
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="core")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bank", default="data/teams/bank_core.npz")
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    cfg = EngineConfig(stage=a.stage, max_turns=40, events=not a.no_events)
    dex = load_dex(a.stage)
    env = AutoResetEnv(cfg, dex, TB.load(a.bank))
    B = a.batch
    es, out = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), B))

    def one(es, k):
        ks = jax.random.split(k, B * 2).reshape(B, 2, -1)
        masks = jax.vmap(lambda s: api.legal_masks(s, cfg, dex))(es.state)
        acts = jax.vmap(api.random_actions)(ks[:, 0], masks, es.state.meta.phase)
        es, _, stats = jax.vmap(env.step)(es, acts, ks[:, 1])
        return es, stats.done

    text = jax.jit(one).lower(es, jax.random.PRNGKey(1)).compile().as_text()
    if a.dump:
        open(a.dump, "w").write(text)
    comps, entry = parse(text)
    hist = collections.Counter()
    tot = count(comps, entry, hist)
    print(f"kernel-like ops per env step: {tot} (budget 10,000); top: {hist.most_common(8)}")


if __name__ == "__main__":
    main()
