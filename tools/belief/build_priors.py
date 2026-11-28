"""Opponent set priors for the Bo1 (closed team sheet) belief filter: per species, K full-set "skeletons" (item,
ability, 4 moves, nature) with prior weights, and per-stat SP marginals (6 stats x 33 SP values) from Smogon chaos.

Why two factors: PS open team sheets (|showteam|) carry item / ability / moves / nature but leave the SP spread blank,
and chaos 'Spreads' is very diffuse (Incineroar: 12k distinct spreads, the top one ~8%), so a list of whole spreads
would rarely contain the true one. Skeletons come from real team sheets (joint co-occurrence), padded with chaos-modal
synthetic skeletons for species with few sheets; SP comes from the chaos spread table, marginalised per stat (the
filter updates those marginals exactly on the stat a damage / speed observation depends on: vgcjax/agent/belief.py).

Inputs (raw downloads: run with python -I, never commit them):
  --sheets   jsonl files with a 'packed' team string per line (data/teams/real_sheets_mc_v1.jsonl, data/raw/ps-replays/
             <month>/*_teams.jsonl); later files can be up-weighted with --sheet_weights
  --chaos    a Smogon chaos json (data/raw/smogon-chaos/<month>/gen9championsvgc2026regmc-1760.json for Bo1)
Output: an npz in the engine's id spaces (vgcjax/data/ids.py), shapes [NS, K, ...] / [NS, 6, 33].

  uv run python -I tools/belief/build_priors.py \\
      --sheets data/teams/real_sheets_mc_v1.jsonl data/raw/ps-replays/2026-11/gen9championsvgc2026regmc_teams.jsonl \\
               data/raw/ps-replays/2026-11/gen9championsvgc2026regmcbo3_teams.jsonl \\
      --sheet_weights 1 3 2 --chaos data/raw/smogon-chaos/2026-10/gen9championsvgc2026regmc-1760.json \\
      --out data/priors/set_priors_bo1.npz
"""
import argparse
import collections
import json
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vgcjax.data import ids  # noqa: E402

K = 16           # skeletons per species
NSP = 33         # SP values 0..32 (Champions: <= 32 per stat, 66 total)
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
tid = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())  # noqa: E731
IDX = {name: {x: i for i, x in enumerate(getattr(ids, name))} for name in ("SPECIES", "MOVES", "ABILITIES", "ITEMS",
                                                                             "NATURES")}


def parse_packed(packed):
    """PS packed team -> [(species, item, ability, moves(sorted tuple), nature)]; unknown ids dropped by the caller."""
    out = []
    for mon in packed.split("]"):
        f = mon.split("|")
        if len(f) < 6:
            continue
        sp = tid(f[1] or f[0])
        moves = tuple(sorted(tid(m) for m in f[4].split(",") if m))
        out.append((sp, tid(f[2]), tid(f[3]), moves, tid(f[5])))
    return out


def chaos_tables(path):
    d = json.load(open(path))["data"]
    out = {}
    for name, v in d.items():
        sp = tid(name)
        def norm(tab):
            tot = sum(tab.values()) or 1.0
            return {tid(k): w / tot for k, w in tab.items() if tid(k) not in ("", "nothing")}
        spreads = collections.Counter()
        natures = collections.Counter()
        for key, w in v.get("Spreads", {}).items():
            if ":" not in key:
                continue
            nat, vals = key.split(":", 1)
            natures[tid(nat)] += w
            spreads[(tid(nat), tuple(int(x) for x in vals.split("/")))] += w
        out[sp] = dict(items=norm(v.get("Items", {})), abilities=norm(v.get("Abilities", {})),
                       moves=norm(v.get("Moves", {})), natures=norm(natures), spreads=spreads,
                       usage=float(v.get("usage", 0.0)))
    return out


def synthetic_skeletons(ch, n):
    """Chaos-modal skeletons: the top items x top abilities x top nature with the top-4 moves, scored by the product
    of marginals (a coarse stand-in where sheets are scarce)."""
    items = sorted(ch["items"].items(), key=lambda x: -x[1])[:4] or [("", 1.0)]
    abil = sorted(ch["abilities"].items(), key=lambda x: -x[1])[:2] or [("", 1.0)]
    nats = sorted(ch["natures"].items(), key=lambda x: -x[1])[:3] or [("serious", 1.0)]
    moves = [m for m, _ in sorted(ch["moves"].items(), key=lambda x: -x[1])[:6]]
    out = []
    for it, wi in items:
        for ab, wa in abil:
            for na, wn in nats:
                out.append(((it, ab, tuple(sorted(moves[:4])), na), wi * wa * wn))
    return sorted(out, key=lambda x: -x[1])[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--sheet_weights", nargs="*", type=float, default=None)
    ap.add_argument("--chaos", required=True)
    ap.add_argument("--synthetic_mass", type=float, default=0.2,
                    help="prior mass given to chaos-synthetic skeletons when a species also has sheet skeletons")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    weights = a.sheet_weights or [1.0] * len(a.sheets)
    counts = collections.defaultdict(collections.Counter)
    for path, w in zip(a.sheets, weights):
        for line in open(path):
            for sp, it, ab, mv, na in parse_packed(json.loads(line)["packed"]):
                counts[sp][(it, ab, mv, na)] += w
    ch = chaos_tables(a.chaos)
    NS = len(ids.SPECIES)
    sk_item = np.zeros((NS, K), np.int16); sk_ab = np.zeros((NS, K), np.int16)
    sk_mv = np.zeros((NS, K, 4), np.int16); sk_na = np.zeros((NS, K), np.int16)
    sk_w = np.zeros((NS, K), np.float32); sk_n = np.zeros(NS, np.int16)
    sp_marg = np.zeros((NS, 6, NSP), np.float32)
    covered = 0
    for sp, si in IDX["SPECIES"].items():
        cands = collections.Counter()
        tot_sheet = sum(counts[sp].values())
        for skel, c in counts[sp].items():
            cands[skel] += (1 - a.synthetic_mass) * c / tot_sheet if tot_sheet else 0.0
        if sp in ch:
            syn = synthetic_skeletons(ch[sp], K)
            tot_syn = sum(w for _, w in syn) or 1.0
            mass = a.synthetic_mass if tot_sheet else 1.0
            for skel, w in syn:
                cands[skel] += mass * w / tot_syn
        kept = []
        for (it, ab, mv, na), w in cands.most_common():
            if ab not in IDX["ABILITIES"] or na not in IDX["NATURES"] or (it and it not in IDX["ITEMS"]) or \
                    any(m not in IDX["MOVES"] for m in mv) or not mv:
                continue
            kept.append(((it, ab, mv, na), w))
            if len(kept) == K:
                break
        if kept:
            covered += 1
            tot = sum(w for _, w in kept)
            for k, ((it, ab, mv, na), w) in enumerate(kept):
                sk_item[si, k] = IDX["ITEMS"][it] if it else 0
                sk_ab[si, k] = IDX["ABILITIES"][ab]
                sk_na[si, k] = IDX["NATURES"][na]
                for j, m in enumerate(mv[:4]):
                    sk_mv[si, k, j] = IDX["MOVES"][m]
                sk_w[si, k] = w / tot
            sk_n[si] = len(kept)
        # per-stat SP marginals from the chaos spread table (uniform where no data)
        if sp in ch and ch[sp]["spreads"]:
            for (_, vals), w in ch[sp]["spreads"].items():
                for s in range(6):
                    sp_marg[si, s, min(max(vals[s], 0), NSP - 1)] += w
            sp_marg[si] /= sp_marg[si].sum(axis=1, keepdims=True)
        else:
            sp_marg[si] = 1.0 / NSP
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(sheets=[str(p) for p in a.sheets], sheet_weights=weights, chaos=str(a.chaos), K=K, NSP=NSP,
                synthetic_mass=a.synthetic_mass, data_version=getattr(ids, "DATA_VERSION", ""))
    np.savez_compressed(out, skel_item=sk_item, skel_ability=sk_ab, skel_moves=sk_mv, skel_nature=sk_na,
                        skel_w=sk_w, skel_n=sk_n, sp_marg=sp_marg, meta=json.dumps(meta))
    print(f"priors: {covered} species with skeletons, K={K} -> {out}")
    top = sorted(((ch[s]["usage"], s) for s in ch if s in IDX["SPECIES"]), reverse=True)[:5]
    for _, s in top:
        si = IDX["SPECIES"][s]
        k0 = 0
        print(f"  {s}: {sk_n[si]} skeletons, top w={sk_w[si, k0]:.2f} item={ids.ITEMS[sk_item[si, k0]]} "
              f"ability={ids.ABILITIES[sk_ab[si, k0]]} nature={ids.NATURES[sk_na[si, k0]]} "
              f"moves={[ids.MOVES[m] for m in sk_mv[si, k0]]}; spe SP mode={int(np.argmax(sp_marg[si, 5]))}")


if __name__ == "__main__":
    main()
