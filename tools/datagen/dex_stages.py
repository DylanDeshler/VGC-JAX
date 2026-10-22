#!/usr/bin/env python3
"""Authoritative dex stage lists core / v0 / v1 / v2 (05 §1, §1.1; 06 §1).

Port of docs/research/dex/scripts/usage_coverage.py + dex_select.py, reduced to the stage
selection and its coverage numbers:
  - v0 / v1: cost_dex({'bo3_1760': tau, 'bo1_0': tau}) with FIXED tau (0.01 / 0.001 by default);
    the research adaptive meets() choice is not used.
  - v2: the whole legal pool.
  - core: DATA_ONLY moves with census data_refs == {} and core_refs == {}; no items; the abilities
    listed in tools/datagen/core_inert.json; the set-level species that can legally have one of them.
Every stage lists species, moves, abilities, items and the condition closure (engine conditions +
conditions reachable from the stage entities), plus meta (inputs with sha256, tau, PS commit, format).

Usage (run with `python -I`: the chaos inputs are untrusted and parsed as JSON only):
  uv run python -I tools/datagen/dex_stages.py --format gen9championsvgc2026regmc \
      --legal data/export/legal_pool.json --census data/export/handler_census.json \
      --sheets data/teams/real_sheets_mc_v1.jsonl --core-inert tools/datagen/core_inert.json \
      --chaos bo3_1760=data/raw/smogon-chaos/2026-09/gen9championsvgc2026regmcbo3-1760.json \
      --chaos bo1_0=data/raw/smogon-chaos/2026-09/gen9championsvgc2026regmc-0.json \
      --out-dir data/export/stages
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNREACHABLE_CONDITIONS = {"dynamax", "maxguard", "commanded", "commanding"}  # no Dynamax/Commander in Reg M-C
KINDS = ("move", "ability", "item")
COND_KEYS = ("volatile", "sideCondition", "slotCondition", "pseudoWeather", "weather", "terrain", "status")
USAGE_LABELS = ("bo3_1760", "bo1_0")  # the two chaos inputs the selection thresholds on
N_SYNTHETIC = 4000
SYNTHETIC_SEED = 1

Dex = dict[str, Any]


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


# ---------------------------------------------------------------------------- usage_coverage.py
def load_chaos(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(path) as f:
        d = json.load(f)
    info = d["info"]
    data: dict[str, Any] = {}
    for name, e in d["data"].items():
        w = sum(e["Abilities"].values())
        if w <= 0:
            continue
        ab = {k: v / w for k, v in e["Abilities"].items() if v > 0}
        it: dict[str, float] = {}
        for k, v in e["Items"].items():
            if v <= 0:
                continue
            k2 = "" if k in ("nothing", "") else k
            it[k2] = it.get(k2, 0) + v / w
        mv = {k: min(1.0, v / w) for k, v in e["Moves"].items() if k and v > 0}
        tm = {k: v / w for k, v in e["Teammates"].items() if v > 0}
        data[name] = {"id": toid(name), "name": name, "usage": e["usage"], "w": w, "raw": e["Raw count"],
                      "abilities": ab, "items": it, "moves": mv, "teammates": tm}
    return info, data


def entity_tables(data: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """Per-entity P(team contains) (1 - prod over species of (1 - usage * f)) and slot mass."""
    tot_usage = sum(s["usage"] for s in data.values())
    out: dict[str, dict[str, dict[str, float]]] = {"species": {}, "moves": {}, "abilities": {}, "items": {}}
    for name, s in data.items():
        out["species"][name] = {"p_team": s["usage"], "mass": s["usage"] / tot_usage}
    for kind in ("moves", "abilities", "items"):
        acc: dict[str, dict[str, float]] = {}
        tot_mass = 0.0
        for s in data.values():
            for e, f in s[kind].items():
                if kind == "items" and e == "":
                    continue
                a = acc.setdefault(e, {"log_not": 0.0, "mass": 0.0})
                p = min(0.999999, s["usage"] * f)
                a["log_not"] += math.log1p(-p)
                a["mass"] += s["usage"] * f
                tot_mass += s["usage"] * f
        for e, a in acc.items():
            out[kind][e] = {"p_team": 1 - math.exp(a["log_not"]), "mass": a["mass"] / tot_mass}
    return out


def mon_cov(s: dict[str, Any], dex: Dex) -> float:
    """P(a set of this species is fully expressible | dex). Moves treated as independent."""
    if s["name"] not in dex["species"]:
        return 0.0
    pa = sum(f for a, f in s["abilities"].items() if a in dex["abilities"])
    pi = sum(f for i, f in s["items"].items() if i == "" or i in dex["items"])
    pm = 1.0
    for m, f in s["moves"].items():
        if m not in dex["moves"]:
            pm *= max(0.0, 1.0 - f)
    return min(1.0, pa) * min(1.0, pi) * pm


def sample_teams(data: dict[str, Any], n: int, seed: int = SYNTHETIC_SEED, size: int = 6) -> list[list[str]]:
    """Teammate-correlated synthetic teams (naive-Bayes pairwise model on chaos teammate stats)."""
    rng = random.Random(seed)
    names = list(data)

    def clause_key(nm: str) -> str:
        return re.split(r"-(Mega|Alola|Galar|Hisui|Paldea|F$|M$)", nm)[0]

    prior = {nm: max(1e-6, data[nm]["usage"]) for nm in names}
    teams = []
    for _ in range(n):
        team: list[str] = []
        keys: set[str] = set()
        logscore = {nm: math.log(prior[nm]) for nm in names}
        for _slot in range(size):
            cand = [nm for nm in names if clause_key(nm) not in keys]
            mx = max(logscore[nm] for nm in cand)
            weights = [math.exp(logscore[nm] - mx) for nm in cand]
            r = rng.random() * sum(weights)
            acc = 0.0
            pick = cand[-1]
            for nm, w in zip(cand, weights):
                acc += w
                if acc >= r:
                    pick = nm
                    break
            team.append(pick)
            keys.add(clause_key(pick))
            tm = data[pick]["teammates"]
            for nm in names:
                cond = tm.get(nm, 0.0)
                ratio = max(1e-3, min(1e3, (cond + 1e-4) / prior[nm]))
                logscore[nm] += math.log(ratio)
        teams.append(team)
    return teams


def team_cov_mc(data: dict[str, Any], dex: Dex, teams: list[list[str]]) -> float:
    covs = {nm: mon_cov(s, dex) for nm, s in data.items()}
    acc = 0.0
    for t in teams:
        p = 1.0
        for nm in t:
            p *= covs[nm]
        acc += p
    return acc / len(teams)


# ---------------------------------------------------------------------------- dex_select.py
def load_sheets(path: str, legal: dict[str, Any]) -> list[dict[str, Any]]:
    """Real sheets from the frozen corpus (packed teams), parsed as in dex_select.load_replay_teams."""
    stone_to_forme = {k: v["megaFormes"][0] for k, v in legal["items"].items() if v["isMegaStone"] and v["megaFormes"]}
    forme_info = legal["battleFormes"]
    cosmetic = {c: b for b, v in legal["species"].items() for c in v.get("cosmeticFormes", [])}
    teams = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            mons = []
            for m in r["packed"].split("]"):
                fl = m.split("|")
                if len(fl) < 5:
                    raise SystemExit(f"malformed sheet {r.get('sheet_sha256')}")
                sp0 = toid(fl[1] or fl[0])
                sp = sp0 if sp0 in legal["species"] else cosmetic.get(sp0, sp0)
                item, ab, moves = toid(fl[2]), toid(fl[3]), [toid(x) for x in fl[4].split(",") if x]
                mega = None
                if item in stone_to_forme and forme_info.get(stone_to_forme[item], {}).get("from", [None])[0] == sp:
                    mega = stone_to_forme[item]
                mons.append({"species": sp, "mega": mega, "item": item, "ability": ab,
                             "mega_ability": forme_info[mega]["abilities"][0] if mega else None, "moves": moves})
            if len(mons) != 6:
                raise SystemExit(f"sheet {r.get('sheet_sha256')} has {len(mons)} mons")
            teams.append({"mons": mons})
    return teams


def real_team_cov(teams: list[dict[str, Any]], dex: Dex) -> float:
    """Fraction of real teams fully expressible (strict: Mega holders' base ability must be in the dex)."""
    sp, mv, ab, it = dex["species_ids"], dex["move"], dex["ability"], dex["item"]
    n_ok = 0
    for t in teams:
        ok = True
        for m in t["mons"]:
            if m["species"] not in sp or (m["mega"] and m["mega"] not in sp):
                ok = False
            if m["item"] and m["item"] not in it:
                ok = False
            if m["mega"] and m["mega_ability"] not in ab:
                ok = False
            if m["ability"] not in ab:
                ok = False
            if any(x not in mv for x in m["moves"]):
                ok = False
        n_ok += ok
    return n_ok / max(1, len(teams))


class Census:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.rec = raw["records"]
        self.engine = list(raw["engine_conditions"])

    def cond_refs(self, kind: str, eid: str) -> set[str]:
        """Conditions an entity can create/inspect (data refs + code refs + its own condition)."""
        r = self.rec[kind].get(eid)
        if not r:
            return set()
        out = set()
        for src in (r.get("data_refs") or {}, r.get("code_refs") or {}):
            for k in COND_KEYS:
                for c in src.get(k, []):
                    out.add(toid(c))
        if r.get("owned_condition") and r["owned_condition"]["fns"]:
            out.add(eid)
        return out

    def closure(self, dex: Dex) -> set[str]:
        """Conditions reachable from the dex entities (plus engine conditions)."""
        conds = set(self.engine)
        frontier = set(self.engine)
        for kind in KINDS:
            for eid in dex[kind]:
                frontier |= self.cond_refs(kind, eid)
        while frontier:
            c = frontier.pop()
            if c in UNREACHABLE_CONDITIONS or c not in self.rec["condition"]:
                continue
            cr = self.rec["condition"][c]
            owner = cr.get("owner")
            if owner and c not in dex.get(owner, set()) and c not in self.engine:
                # owned condition of an entity not in the dex: reachable only via that entity
                # (exceptions: conditions also created generically, e.g. by data refs of dex entities)
                if not any(c in self.cond_refs(k, e) for k in KINDS for e in dex[k]):
                    continue
            if c in conds and c not in self.engine:
                continue
            conds.add(c)
            for k in COND_KEYS:
                for x in cr.get("code_refs", {}).get(k, []):
                    if x not in conds:
                        frontier.add(x)
        return {c for c in conds if c in self.rec["condition"] and c not in UNREACHABLE_CONDITIONS}

    def entity_templates(self, kind: str, eid: str) -> set[int]:
        """Templates needed by an entity incl. its owned/referenced conditions (first level)."""
        ts = {f["template"] for f in self.rec[kind][eid]["fns"]}
        for c in self.cond_refs(kind, eid):
            if c in self.rec["condition"] and c not in UNREACHABLE_CONDITIONS:
                ts |= {f["template"] for f in self.rec["condition"][c]["fns"]}
        return ts

    def cost_summary(self, dex: Dex, conds: set[str]) -> dict[str, Any]:
        fns = [f for k in KINDS for e in dex[k] for f in self.rec[k][e]["fns"]]
        fns += [f for c in conds for f in self.rec["condition"][c]["fns"]]
        by_class = {k: dict(sorted(collections.Counter(self.rec[k][e]["class"] for e in dex[k]).items()))
                    for k in KINDS}
        by_class["condition"] = dict(sorted(collections.Counter(self.rec["condition"][c]["class"] for c in conds).items()))
        return {
            "n_conditions": len(conds), "n_handler_fns": len(fns),
            "n_distinct_templates": len({f["template"] for f in fns}), "loc": sum(f["loc"] for f in fns),
            "classes": by_class, "n_custom_entities": sum(v.get("CUSTOM", 0) for v in by_class.values()),
        }


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True)
    ap.add_argument("--legal", required=True)
    ap.add_argument("--census", required=True)
    ap.add_argument("--sheets", required=True, help="frozen real-sheet corpus (JSONL)")
    ap.add_argument("--core-inert", required=True)
    ap.add_argument("--chaos", action="append", required=True, help="label=path; needs bo3_1760 and bo1_0")
    ap.add_argument("--tau-v0", type=float, default=0.01)
    ap.add_argument("--tau-v1", type=float, default=0.001)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    with open(a.legal) as f:
        legal = json.load(f)
    with open(a.census) as f:
        census_raw = json.load(f)
    with open(a.core_inert) as f:
        core_inert = json.load(f)
    if legal["formatId"] != a.format or census_raw.get("format") != a.format:
        raise SystemExit(f"legal pool / census are for {legal['formatId']} / {census_raw.get('format')}, not {a.format}")
    census = Census(census_raw)
    chaos_paths = dict(s.split("=", 1) for s in a.chaos)
    if set(USAGE_LABELS) - set(chaos_paths):
        raise SystemExit(f"--chaos needs labels {USAGE_LABELS}")
    fmts: dict[str, Any] = {}
    for label in sorted(chaos_paths):
        info, data = load_chaos(chaos_paths[label])
        fmts[label] = {"info": info, "data": data, "tables": entity_tables(data)}
    rteams = load_sheets(a.sheets, legal)

    forme_from = {k: v["from"][0] for k, v in legal["battleFormes"].items()}
    chaos_name_by_id: dict[str, str] = {}
    for label in sorted(fmts):
        for nm in fmts[label]["data"]:
            chaos_name_by_id[toid(nm)] = nm
    legal_species_ids = set(legal["species"]) | {k for k in legal["battleFormes"] if not k.endswith("gmax")}

    # replay-derived P(team contains ability) for base abilities of Mega Stone holders (05 §1 "Base abilities")
    rep_ab: collections.Counter[str] = collections.Counter()
    mega_base_ab: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for t in rteams:
        seen: set[str] = set()
        for m in t["mons"]:
            seen.add(m["ability"])
            if m["mega"]:
                seen.add(m["mega_ability"])
                mega_base_ab[m["mega"]][m["ability"]] += 1
        for x in seen:
            rep_ab[x] += 1
    nrt = max(1, len(rteams))
    mega_base_ids = {ab for c in mega_base_ab.values() for ab in c}

    def p_team(kind: str, eid: str, label: str) -> float:
        tk = {"move": "moves", "ability": "abilities", "item": "items"}[kind]
        v = fmts[label]["tables"][tk].get(eid)
        return v["p_team"] if v else 0.0

    def max_p(kind: str, eid: str, labels: list[str]) -> float:
        ps = [p_team(kind, eid, lb) for lb in labels]
        if kind == "ability":
            ps.append(rep_ab.get(eid, 0) / nrt if eid in mega_base_ids else 0.0)
        return max(ps)

    def make_dex(species_ids: set[str], moves: set[str], abilities: set[str], items: set[str]) -> Dex:
        sp = set(species_ids)
        sp |= {forme_from[s] for s in sp if s in forme_from}  # megas imply their base forme
        return {"species_ids": sp, "species": {chaos_name_by_id[s] for s in sp if s in chaos_name_by_id},
                "move": set(moves), "ability": set(abilities), "item": set(items),
                "moves": set(moves), "abilities": set(abilities), "items": set(items)}

    # cost-aware selection: zero-cost entities + usage >= tau + free riders whose templates are paid for
    engine_templates: set[int] = set()
    for c in census.closure({"move": set(), "ability": set(), "item": set()}):
        engine_templates |= {f["template"] for f in census.rec["condition"][c]["fns"]}
    all_ents = [(k, e) for k in KINDS for e in sorted(census.rec[k])]
    ent_tmpl = {(k, e): census.entity_templates(k, e) for k, e in all_ents}

    def cost_dex(tau_by_label: dict[str, float]) -> tuple[Dex, dict[tuple[str, str], str]]:
        reason: dict[tuple[str, str], str] = {}
        tmpl = set(engine_templates)
        for k, e in all_ents:
            if ent_tmpl[(k, e)] <= engine_templates:
                reason[(k, e)] = "zero_cost"
        for k, e in all_ents:
            if any(max_p(k, e, [lb]) >= tau for lb, tau in tau_by_label.items()):
                reason.setdefault((k, e), "usage")
                tmpl |= ent_tmpl[(k, e)]
        changed = True
        while changed:
            changed = False
            for k, e in all_ents:
                if (k, e) not in reason and ent_tmpl[(k, e)] <= tmpl:
                    reason[(k, e)] = "free_rider_shared_template"
                    changed = True
        d = make_dex(legal_species_ids, {e for k, e in reason if k == "move"},
                     {e for k, e in reason if k == "ability"}, {e for k, e in reason if k == "item"})
        return d, reason

    # synthetic teams per usage label (usage_coverage.py), for the coverage numbers only
    syn = {lb: sample_teams(fmts[lb]["data"], N_SYNTHETIC) for lb in USAGE_LABELS}

    ps_commit = legal.get("ps_commit") or census_raw.get("ps_commit")
    inputs = {rel(p): sha256_file(p) for p in [a.legal, a.census, a.sheets, a.core_inert,
                                                *[chaos_paths[lb] for lb in sorted(chaos_paths)]]}
    base_meta = {"format": a.format, "ps_commit": ps_commit, "inputs": inputs,
                 "chaos_labels": {lb: rel(chaos_paths[lb]) for lb in sorted(chaos_paths)},
                 "real_sheets": len(rteams)}

    def stage_doc(name: str, dex: Dex, species: set[str], selection: dict[str, Any],
                  reason: dict[tuple[str, str], str] | None) -> dict[str, Any]:
        conds = census.closure(dex)
        cov = {lb: round(team_cov_mc(fmts[lb]["data"], dex, syn[lb]), 4) for lb in USAGE_LABELS}
        cov["real_sheets_strict"] = round(real_team_cov(rteams, dex), 4)
        doc: dict[str, Any] = {
            "stage": name,
            "meta": {**base_meta, "selection": selection},
            "species": sorted(species),
            "moves": sorted(dex["move"]), "abilities": sorted(dex["ability"]), "items": sorted(dex["item"]),
            "conditions": sorted(conds),
            "summary": {"counts": {"species": len(species), "moves": len(dex["move"]),
                                   "abilities": len(dex["ability"]), "items": len(dex["item"]),
                                   "conditions": len(conds)},
                        "cost": census.cost_summary(dex, conds), "team_coverage": cov},
        }
        if reason is not None:
            doc["inclusion"] = {k: {e: reason[(k, e)] for e in sorted(dex[k])} for k in KINDS}
        return doc

    docs: dict[str, dict[str, Any]] = {}
    for name, tau in (("v0", a.tau_v0), ("v1", a.tau_v1)):
        taus = {lb: tau for lb in USAGE_LABELS}
        d, reason = cost_dex(taus)
        docs[name] = stage_doc(name, d, d["species_ids"], {
            "rule": "cost_dex: zero-cost + max(P(team) in each label, replay P(team) of Mega holders' base abilities) >= tau + free riders",
            "tau": taus}, reason)
    full = make_dex(legal_species_ids, set(census.rec["move"]), set(census.rec["ability"]), set(census.rec["item"]))
    docs["v2"] = stage_doc("v2", full, full["species_ids"], {"rule": "whole legal pool"}, None)

    # core (05 §1.1)
    core_abilities = set(core_inert["abilities"])
    for ab in core_abilities:
        if census.rec["ability"].get(ab, {}).get("class") != "DATA_ONLY":
            raise SystemExit(f"core_inert ability {ab} is not a legal DATA_ONLY ability")
    core_moves = {e for e, r in census.rec["move"].items()
                  if r["class"] == "DATA_ONLY" and not r["data_refs"] and not r["core_refs"]}
    core_species = {s for s, v in legal["species"].items() if any(x["id"] in core_abilities for x in v["abilities"])}
    core = make_dex(core_species, core_moves, core_abilities, set())
    docs["core"] = stage_doc("core", core, core_species, {
        "rule": "moves: DATA_ONLY with data_refs == {} and core_refs == {}; items: none; abilities: core_inert.json; "
                "species: set-level species that can legally have a core ability; no Mega Stones"}, None)

    os.makedirs(a.out_dir, exist_ok=True)
    for name in ("core", "v0", "v1", "v2"):
        with open(os.path.join(a.out_dir, f"{name}.json"), "w") as f:
            f.write(json.dumps(docs[name], sort_keys=True, indent=1) + "\n")
        s = docs[name]["summary"]
        print(f"{name}: {s['counts']} custom={s['cost']['n_custom_entities']} "
              f"templates={s['cost']['n_distinct_templates']} coverage={s['team_coverage']}")


if __name__ == "__main__":
    main()
