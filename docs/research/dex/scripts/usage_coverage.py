#!/usr/bin/env python3
"""Usage coverage + team-coverage analysis from Smogon chaos JSON.

Usage:
  python3 -I usage_coverage.py <legal_pool.json> <out_dir> <label=path.json> [<label=path.json> ...]

Outputs (in out_dir):
  coverage_summary.json   per-cutoff mass-coverage curves and team-coverage sweep
  usage_tables.json       per-entity usage numbers per cutoff (P(team contains), slot mass)
"""
import json
import math
import os
import re
import sys

import random


def toid(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_chaos(path):
    d = json.load(open(path))
    info = d["info"]
    data = {}
    for name, e in d["data"].items():
        w = sum(e["Abilities"].values())
        if w <= 0:
            continue
        ab = {k: v / w for k, v in e["Abilities"].items() if v > 0}
        it = {}
        for k, v in e["Items"].items():
            if v <= 0:
                continue
            k2 = "" if k in ("nothing", "") else k
            it[k2] = it.get(k2, 0) + v / w
        mv = {k: min(1.0, v / w) for k, v in e["Moves"].items() if k and v > 0}
        tm = {k: v / w for k, v in e["Teammates"].items() if v > 0}
        data[name] = {"id": toid(name), "usage": e["usage"], "w": w, "raw": e["Raw count"],
                      "abilities": ab, "items": it, "moves": mv, "teammates": tm}
    return info, data


def entity_tables(data):
    """Return per-entity stats: team inclusion probability (approx) and slot mass."""
    tot_usage = sum(s["usage"] for s in data.values())  # ~6
    out = {"species": {}, "moves": {}, "abilities": {}, "items": {}}
    # species
    for name, s in data.items():
        out["species"][name] = {"p_team": s["usage"], "mass": s["usage"] / tot_usage}
    # moves / abilities / items
    for kind, key in (("moves", "moves"), ("abilities", "abilities"), ("items", "items")):
        acc = {}
        tot_mass = 0.0
        for name, s in data.items():
            for e, f in s[key].items():
                if kind == "items" and e == "":
                    continue
                a = acc.setdefault(e, {"log_not": 0.0, "mass": 0.0, "users": {}})
                p = min(0.999999, s["usage"] * f)
                a["log_not"] += math.log1p(-p)
                a["mass"] += s["usage"] * f
                a["users"][name] = f
                tot_mass += s["usage"] * f
        for e, a in acc.items():
            top = sorted(a["users"].items(), key=lambda x: -x[1] * data[x[0]]["usage"])[:5]
            out[kind][e] = {"p_team": 1 - math.exp(a["log_not"]), "mass": a["mass"] / tot_mass,
                            "n_users": len(a["users"]),
                            "top_users": [[u, round(f, 4)] for u, f in top]}
    return out


def mass_curve(tab, thresholds=(0.9, 0.95, 0.99, 0.995, 0.999, 1.0)):
    vals = sorted((v["mass"] for v in tab.values()), reverse=True)
    res = {}
    for t in thresholds:
        c = 0.0
        for i, v in enumerate(vals):
            c += v
            if c >= t - 1e-9:
                res[str(t)] = i + 1
                break
        else:
            res[str(t)] = len(vals)
    res["total"] = len(vals)
    return res


def mon_cov(s, dex):
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


def mon_cov_union(s, dex):
    """Union-bound (pessimistic) version: 1 - sum of missing frequencies, per component."""
    if s["name"] not in dex["species"]:
        return 0.0
    pa = sum(f for a, f in s["abilities"].items() if a in dex["abilities"])
    pi = sum(f for i, f in s["items"].items() if i == "" or i in dex["items"])
    miss = sum(f for m, f in s["moves"].items() if m not in dex["moves"])
    return max(0.0, min(1.0, pa) * min(1.0, pi) * (1 - miss))


def sample_teams(data, n, seed=0, size=6):
    """Teammate-correlated synthetic teams (naive-Bayes pairwise model on chaos teammate stats)."""
    rng = random.Random(seed)
    names = list(data)
    base = {nm: toid(nm.split("-Mega")[0]) for nm in names}
    # species clause key: base species id (strip forme suffixes known to share dex number)
    def clause_key(nm):
        return re.split(r"-(Mega|Alola|Galar|Hisui|Paldea|F$|M$)", nm)[0]
    prior = {nm: max(1e-6, data[nm]["usage"]) for nm in names}
    teams = []
    for _ in range(n):
        team = []
        keys = set()
        logscore = {nm: math.log(prior[nm]) for nm in names}
        for slot in range(size):
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


def team_cov(data, dex, teams=None):
    covs = {nm: mon_cov(s, dex) for nm, s in data.items()}
    covu = {nm: mon_cov_union(s, dex) for nm, s in data.items()}
    tot = sum(s["usage"] for s in data.values())
    mon = sum(s["usage"] * covs[nm] for nm, s in data.items()) / tot
    monu = sum(s["usage"] * covu[nm] for nm, s in data.items()) / tot
    res = {"mon_cov": mon, "team_cov_indep": mon ** 6, "mon_cov_unionbound": monu, "team_cov_indep_unionbound": monu ** 6}
    if teams:
        acc = 0.0
        for t in teams:
            p = 1.0
            for nm in t:
                p *= covs[nm]
            acc += p
        res["team_cov_mc"] = acc / len(teams)
    return res


def build_dex_by_threshold(tables_list, taus):
    """Include every entity whose P(team contains) >= tau in any of the given formats."""
    dex = {"species": set(), "moves": set(), "abilities": set(), "items": set()}
    for tables, tau in zip(tables_list, taus):
        for kind in dex:
            for e, v in tables[kind].items():
                if v["p_team"] >= tau:
                    dex[kind].add(e)
    return dex


def main():
    legal_path, out_dir = sys.argv[1], sys.argv[2]
    specs = [a.split("=", 1) for a in sys.argv[3:]]
    legal = json.load(open(legal_path))
    megastones = {k for k, v in legal["items"].items() if v["isMegaStone"]}
    os.makedirs(out_dir, exist_ok=True)
    fmts = {}
    for label, path in specs:
        info, data = load_chaos(path)
        for nm, s in data.items():
            s["name"] = nm
        fmts[label] = {"info": info, "data": data, "tables": entity_tables(data)}
        print(label, info["number of battles"], len(data), file=sys.stderr)

    summary = {"formats": {}, "sweeps": {}}
    for label, f in fmts.items():
        t = f["tables"]
        ms = {k: v for k, v in t["items"].items() if k in megastones}
        tot_ms = sum(v["mass"] for v in ms.values())
        ms_norm = {k: {"mass": v["mass"] / tot_ms} for k, v in ms.items()} if tot_ms else {}
        nonms = {k: v for k, v in t["items"].items() if k not in megastones}
        tot_nm = sum(v["mass"] for v in nonms.values())
        nonms_norm = {k: {"mass": v["mass"] / tot_nm} for k, v in nonms.items()}
        mega_species = {k: v for k, v in t["species"].items() if "-Mega" in k}
        summary["formats"][label] = {
            "battles": f["info"]["number of battles"], "cutoff": f["info"]["cutoff"],
            "metagame": f["info"]["metagame"],
            "mass_curves": {
                "species": mass_curve(t["species"]), "moves": mass_curve(t["moves"]),
                "abilities": mass_curve(t["abilities"]), "items": mass_curve(t["items"]),
                "items_excl_megastones": mass_curve(nonms_norm), "megastones": mass_curve(ms_norm),
            },
            "mega_slot_share": sum(v["mass"] for v in mega_species.values()),
            "mega_species_count": len(mega_species),
            "item_slot_share_megastones": tot_ms,
            "item_slot_share_none": sum(s["usage"] * s["items"].get("", 0) for s in f["data"].values()) / sum(s["usage"] for s in f["data"].values()),
        }
    # synthetic teams per format
    teams = {label: sample_teams(f["data"], 4000, seed=1) for label, f in fmts.items()}

    # threshold sweep: single-format thresholds
    taus = [1e-2, 5e-3, 3e-3, 2e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 0.0]
    labels = list(fmts)
    for src in labels:
        rows = []
        for tau in taus:
            dex = build_dex_by_threshold([fmts[src]["tables"]], [tau])
            row = {"tau": tau, "sizes": {k: len(v) for k, v in dex.items()},
                   "megastones": len(dex["items"] & megastones), "team_cov": {}}
            for tgt in labels:
                row["team_cov"][tgt] = team_cov(fmts[tgt]["data"], dex, teams[tgt])
            rows.append(row)
        summary["sweeps"][f"threshold_on_{src}"] = rows
    json.dump(summary, open(os.path.join(out_dir, "coverage_summary.json"), "w"), indent=1)
    tabs = {label: f["tables"] for label, f in fmts.items()}
    json.dump(tabs, open(os.path.join(out_dir, "usage_tables.json"), "w"))
    # persist synthetic teams for reuse
    json.dump(teams, open(os.path.join(out_dir, "synthetic_teams.json"), "w"))


if __name__ == "__main__":
    main()
