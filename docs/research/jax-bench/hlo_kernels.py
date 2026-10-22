"""Estimate 'kernels executed per env step' from an optimized HLO dump (compiled.as_text()).

Counts top-level (non-fused) HLO instructions that would each be a separate kernel/thunk
(fusion, gather, scatter, dynamic-update-slice, sort, copy, reduce, custom-call, ...), multiplying
while-loop bodies by their known trip count and recursing into call/conditional computations.
On GPU every fusion is exactly one kernel (openxla.org/xla/gpu_architecture), so this is a rough
proxy for launches per step; XLA:GPU's fusion decisions differ somewhat from XLA:CPU's.

usage: python hlo_kernels.py results/rollout_B8192.hlo.txt [scan_trip_count=128]
"""
import collections
import re
import sys

FREE = {"parameter", "constant", "get-tuple-element", "tuple", "bitcast", "after-all", "partition-id",
        "replica-id", "opt-barrier", "iota"}


def parse(path):
    comps, cur, entry = {}, None, None
    for line in open(path):
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


def count(comps, name, memo, hist, mult=1):
    total = 0
    for op, line in comps[name]:
        if op == "while":
            body = re.search(r"body=%?([\w.\-]+)", line).group(1)
            cond = re.search(r"condition=%?([\w.\-]+)", line).group(1)
            n = re.search(r'"known_trip_count":\{"n":"(\d+)"', line)
            n = int(n.group(1)) if n else 1
            if n <= SMALL_LOOP:  # threefry's rolled 5-trip loops (CPU-only lowering) ~ one fused kernel on GPU
                total += 1
                hist["small-loop(as 1 kernel)"] += mult
                continue
            total += n * (count(comps, body, memo, hist, mult * n) + count(comps, cond, memo, hist, mult * n))
        elif op == "conditional":
            brs = re.findall(r"%?([\w.\-]+)", re.search(r"branch_computations=\{([^}]*)\}", line).group(1))
            total += 1 + max(count(comps, b, memo, hist, mult) for b in brs)
        elif op == "call":
            total += count(comps, re.search(r"to_apply=%?([\w.\-]+)", line).group(1), memo, hist, mult)
        elif op not in FREE:
            total += 1
            hist[op] += mult
    return total


SMALL_LOOP = 0


def main():
    global SMALL_LOOP
    path = sys.argv[1]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    comps, entry = parse(path)
    print(f"computations={len(comps)} entry={entry}")
    for small in (0, 5):
        SMALL_LOOP = small
        hist = collections.Counter()
        tot = count(comps, entry, {}, hist)
        label = "as compiled for CPU" if small == 0 else "treating trip<=5 loops (CPU-only rolled threefry) as 1 kernel"
        print(f"[{label}] kernel-like ops executed per rollout: {tot} -> per env step (T={T}): {tot / T:.0f}")
        print("   top ops per env step:", [(k, round(v / T)) for k, v in hist.most_common(8)])


if __name__ == "__main__":
    main()
