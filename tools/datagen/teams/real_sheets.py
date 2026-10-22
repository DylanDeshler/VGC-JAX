#!/usr/bin/env python3
"""Build the frozen real-sheet corpus from scraped PS Bo3 |showteam| rows (06 §4.0, §4.2).

Usage (run with `python -I`; the replay/chaos inputs are untrusted and parsed as JSON only):
  uv run python -I tools/datagen/teams/real_sheets.py --format gen9championsvgc2026regmc --mod champions \
      --replays data/raw/ps-replays/2026-10/gen9championsvgc2026regmcbo3_teams.jsonl \
      --chaos data/raw/smogon-chaos/2026-09/gen9championsvgc2026regmcbo3-1760.json \
      --chaos data/raw/smogon-chaos/2026-09/gen9championsvgc2026regmc-0.json \
      --legal data/export/legal_pool.json --out data/teams/real_sheets_mc_v1.jsonl

Rules:
  - One row per unique packed team (the raw packed string is the canonical sheet); usernames, exact
    ratings, replay ids and upload times are dropped. `rating_bucket` = max player rating over the
    team's appearances floored to 100 (game rating if no player rating; null if neither).
  - Blank gender -> species' fixed gender (M/F/N) if any, else 'M' if allowed, else 'F'.
  - SP (stored in the packed EV field) are sampled from chaos `Spreads` for the species (Mega forme
    first when the sheet holds its own Mega Stone; other formes fall back to their base species), conditioned on the shown nature, trying the
    --chaos files in order; if no spread has that nature, the top spreads of the first source that
    lists the species are relabelled with the nature. Draws use a counter-mode sha256 stream seeded
    by sha256('spfill|' + canonical_sheet).
  - Every filled team is validated with PS TeamValidator (tools/datagen/teams/validate.js); invalid
    teams are dropped and counted in the meta file.
Outputs: <out> (JSONL, sorted by sheet hash), <out stem>.sha256, <out stem>.meta.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.dirname(os.path.abspath(__file__))
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
TOP_RELABEL = 10  # top-N spreads used by the relabel fallback


def toid(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), REPO)


class HashStream:
    """Deterministic uniform draws: u_i = sha256(seed || i)[:8] / 2^64."""

    def __init__(self, seed: bytes) -> None:
        self.seed = seed
        self.i = 0

    def uniform(self) -> float:
        d = hashlib.sha256(self.seed + self.i.to_bytes(4, "big")).digest()
        self.i += 1
        return int.from_bytes(d[:8], "big") / 2.0**64


def weighted_pick(entries: list[tuple[str, float]], u: float) -> str:
    tot = sum(w for _, w in entries)
    r = u * tot
    acc = 0.0
    for k, w in entries:
        acc += w
        if r < acc:
            return k
    return entries[-1][0]


def load_spreads(path: str) -> dict[str, list[tuple[str, float]]]:
    """species id -> [(\"Nature:a/b/c/d/e/f\", weight)] sorted by (-weight, key)."""
    with open(path) as f:
        data = json.load(f)["data"]
    out: dict[str, list[tuple[str, float]]] = {}
    for name, e in data.items():
        sp = [(k, float(v)) for k, v in e.get("Spreads", {}).items() if v > 0 and ":" in k]
        if sp:
            out[toid(name)] = sorted(sp, key=lambda x: (-x[1], x[0]))
    return out


def pack_evs(sps: list[int]) -> str:
    """PS Teams.pack EV field: comma-separated hp..spe with '' for 0; '' if all zero."""
    if not any(sps):
        return ""
    return ",".join(str(v) if v else "" for v in sps)


def fill_sp(keys: list[str], nature: str, sources: list[dict[str, list[tuple[str, float]]]],
            u: float) -> tuple[list[int], str]:
    for si, src in enumerate(sources):
        for k in keys:
            cand = [(s, w) for s, w in src.get(k, []) if s.split(":", 1)[0] == nature]
            if cand:
                pick = weighted_pick(cand, u)
                return [int(x) for x in pick.split(":", 1)[1].split("/")], f"nature:{si}:{k}"
    for si, src in enumerate(sources):
        for k in keys:
            if src.get(k):
                relab: dict[str, float] = {}
                for s, w in src[k][:TOP_RELABEL]:
                    sp = s.split(":", 1)[1]
                    relab[sp] = relab.get(sp, 0.0) + w
                cand = sorted(relab.items(), key=lambda x: (-x[1], x[0]))
                pick = weighted_pick(cand, u)
                return [int(x) for x in pick.split("/")], f"relabel:{si}:{k}"
    raise SystemExit(f"no chaos spreads for any of {keys} in any source")


def fill_gender(species: str, genders: dict[str, Any]) -> str:
    g = genders[species]
    if g["gender"]:
        return g["gender"]
    return "M" if g["ratioM"] > 0 else "F"


def rating_bucket(rows: list[dict[str, Any]]) -> int | None:
    pr = [r["player_rating"] for r in rows if isinstance(r.get("player_rating"), int)]
    gr = [r["game_rating"] for r in rows if isinstance(r.get("game_rating"), int)]
    best = max(pr) if pr else (max(gr) if gr else None)
    return None if best is None else (best // 100) * 100


def node_json(args: list[str], stdin: str | None = None) -> str:
    res = subprocess.run(["node", *args], input=stdin, capture_output=True, text=True, check=True, cwd=REPO)
    return res.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True)
    ap.add_argument("--mod", default="champions")
    ap.add_argument("--replays", required=True)
    ap.add_argument("--chaos", action="append", required=True, help="chaos JSON, in fallback order")
    ap.add_argument("--legal", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.legal) as f:
        legal = json.load(f)
    if legal["formatId"] != a.format:
        raise SystemExit(f"legal pool is for {legal['formatId']}, not {a.format}")
    stone_forme = {k: v["megaFormes"][0] for k, v in legal["items"].items() if v["isMegaStone"] and v["megaFormes"]}
    forme_from = {k: v["from"][0] for k, v in legal["battleFormes"].items()}
    # cosmetic/event formes (e.g. Vivillon-Fancy) share the base species' entry in chaos stats
    base_of = {k: toid(v["baseSpecies"]) for k, v in legal["species"].items()}
    base_of.update({c: toid(v["baseSpecies"]) for v in legal["species"].values() for c in v.get("cosmeticFormes", [])})
    genders = json.loads(node_json([os.path.join(HERE, "species_genders.js"), "--mod", a.mod]))
    sources = [load_spreads(p) for p in a.chaos]

    # group raw rows by canonical sheet (raw packed string)
    groups: dict[str, list[dict[str, Any]]] = {}
    n_rows = 0
    with open(a.replays) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_rows += 1
            groups.setdefault(r["packed"], []).append(r)

    built: list[dict[str, Any]] = []
    n_gender_fill = 0
    sp_kinds: dict[str, int] = {}
    for canon, rows in groups.items():
        sheet_sha = hashlib.sha256(canon.encode()).hexdigest()
        rng = HashStream(hashlib.sha256(("spfill|" + canon).encode()).digest())
        mons = []
        fills_g: list[int] = []
        fills_sp: list[str] = []
        for slot, m in enumerate(canon.split("]")):
            fld = m.split("|")
            if len(fld) < 12:
                raise SystemExit(f"malformed packed set: {m!r}")
            species = toid(fld[1] or fld[0])
            if species not in genders:
                raise SystemExit(f"unknown species {species!r}")
            fld[0] = fld[1] or fld[0]  # nickname := species name (OTS sheets carry no nicknames)
            fld[1] = ""
            item = toid(fld[2])
            mega = stone_forme.get(item)
            if mega and forme_from.get(mega) != species:
                mega = None
            u = rng.uniform()
            if not fld[7]:
                fld[7] = fill_gender(species, genders)
                fills_g.append(slot)
                n_gender_fill += 1
            if fld[6]:
                raise SystemExit("sheet already has SP; refusing to overwrite")
            keys = [k for k in (mega, species, base_of.get(species)) if k]
            sps, how = fill_sp(keys, fld[5], sources, u)
            fld[6] = pack_evs(sps)
            fills_sp.append(how)
            kind = how.split(":", 1)[0]
            sp_kinds[kind] = sp_kinds.get(kind, 0) + 1
            mons.append("|".join(fld))
        built.append({"sheet_sha256": sheet_sha, "packed": "]".join(mons), "rating_bucket": rating_bucket(rows),
                      "n_appearances": len(rows), "fills": {"gender_slots": fills_g, "sp": fills_sp}})
    built.sort(key=lambda r: r["sheet_sha256"])

    res = node_json([os.path.join(HERE, "validate.js"), "--format", a.format],
                    stdin="\n".join(r["packed"] for r in built) + "\n").splitlines()
    if len(res) != len(built):
        raise SystemExit("validator output length mismatch")
    valid, invalid = [], []
    for r, v in zip(built, res):
        vr = json.loads(v)
        if vr["ok"]:
            valid.append(r)
        else:
            invalid.append({"sheet_sha256": r["sheet_sha256"], "problems": vr["problems"]})
    print(f"rows={n_rows} unique={len(built)} valid={len(valid)} invalid={len(invalid)} "
          f"gender_fills={n_gender_fill} sp={sp_kinds}", file=sys.stderr)
    for x in invalid:
        print("INVALID", x["sheet_sha256"][:12], x["problems"], file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in valid:
            f.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    digest = sha256_file(a.out)
    stem = a.out[: -len(".jsonl")] if a.out.endswith(".jsonl") else a.out
    with open(stem + ".sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(a.out)}\n")
    meta = {
        "format": a.format, "mod": a.mod, "sha256": digest,
        "inputs": {rel(p): sha256_file(p) for p in [a.replays, *a.chaos, a.legal]},
        "counts": {"raw_rows": n_rows, "unique_sheets": len(built), "valid": len(valid), "invalid": len(invalid),
                   "gender_fills": n_gender_fill, "sp_fill_kinds": sp_kinds},
        "invalid": invalid,
        "rules": {
            "canonical_sheet": "raw packed team string as scraped (before fills)",
            "rating_bucket": "floor(max player_rating, 100); game_rating if no player rating",
            "gender": "species fixed gender (M/F/N), else M if ratioM > 0, else F",
            "sp": "chaos Spreads weighted by count, conditioned on nature, sources in order; "
                  f"fallback top-{TOP_RELABEL} spreads relabelled; u_slot = sha256(sha256('spfill|'+canonical) || slot)[:8]/2^64",
        },
    }
    with open(stem + ".meta.json", "w") as f:
        f.write(json.dumps(meta, sort_keys=True, indent=1) + "\n")


if __name__ == "__main__":
    main()
