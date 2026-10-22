#!/usr/bin/env python3
"""Trend / drift check: v1 dex team coverage on previous-regulation stats; top-species overlap.
Usage: python3 -I trend_check.py <dex_dir> <label=chaos.json> ...
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usage_coverage import load_chaos, team_cov, sample_teams, toid  # noqa: E402

dex_dir = sys.argv[1]
v1 = {"species": set(), "moves": set(), "abilities": set(), "items": set()}
sp_ids = {s["id"] for s in json.load(open(os.path.join(dex_dir, "v1_species.json")))["species"]}
for k, f in (("moves", "v1_moves.json"), ("abilities", "v1_abilities.json"), ("items", "v1_items.json")):
    v1[k] = {e["id"] for e in json.load(open(os.path.join(dex_dir, f)))["entries"]}
out = {}
tops = {}
for spec in sys.argv[2:]:
    label, path = spec.split("=", 1)
    info, data = load_chaos(path)
    for nm, s in data.items():
        s["name"] = nm
    v1["species"] = {nm for nm in data if toid(nm) in sp_ids}
    teams = sample_teams(data, 3000, seed=3)
    tc = team_cov(data, v1, teams)
    missing_sp = sorted((nm for nm in data if toid(nm) not in sp_ids), key=lambda n: -data[n]["usage"])[:8]
    tops[label] = [nm for nm, _ in sorted(data.items(), key=lambda x: -x[1]["usage"])[:30]]
    out[label] = {"battles": info["number of battles"], "team_cov_mc": round(tc["team_cov_mc"], 4),
                  "team_cov_indep": round(tc["team_cov_indep"], 4),
                  "species_not_in_v1": [(n, round(data[n]["usage"], 4)) for n in missing_sp]}
labels = list(tops)
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b = set(tops[labels[i]]), set(tops[labels[j]])
        out[f"top30_overlap {labels[i]} vs {labels[j]}"] = len(a & b)
json.dump(out, open(os.path.join(dex_dir, "out", "trend_check.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
