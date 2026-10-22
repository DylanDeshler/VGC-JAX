#!/usr/bin/env python3
"""Dex selection + handler-cost accounting + team coverage (synthetic and real replay teams).

Usage:
  python3 -I dex_select.py <dex_dir> <legal_pool.json> <census_raw_legal.json> <replay_teams.jsonl> \
      <label=chaos.json> [...]

Reads <dex_dir>/out/synthetic_teams.json (from usage_coverage.py) and writes into <dex_dir>:
  v1_species.json v1_moves.json v1_abilities.json v1_items.json
  handler_census_v1.json handler_census_legal.json
  out/dex_options.json (sizes / costs / coverage for every candidate dex)
"""
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usage_coverage import load_chaos, entity_tables, team_cov  # noqa: E402


def toid(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


UNREACHABLE_CONDITIONS = {"dynamax", "maxguard", "commanded", "commanding"}  # no Dynamax/Commander in Reg M-C
KINDS = ("move", "ability", "item")
TARGETS = ("bo3_1760", "bo1_1760", "bo3_0", "bo1_0")


# ----------------------------------------------------------------------------------------------
def load_replay_teams(path, legal):
    stone_to_forme = {k: v["megaFormes"][0] for k, v in legal["items"].items() if v["isMegaStone"] and v["megaFormes"]}
    forme_info = legal["battleFormes"]
    cosmetic = {c: b for b, v in legal["species"].items() for c in v.get("cosmeticFormes", [])}
    teams = []
    seen = set()
    for line in open(path):
        r = json.loads(line)
        if r["packed"] in seen:
            continue
        seen.add(r["packed"])
        mons = []
        ok = True
        for m in r["packed"].split("]"):
            f = m.split("|")
            if len(f) < 5:
                ok = False
                break
            sp0 = toid(f[1] or f[0])
            sp = sp0 if sp0 in legal["species"] else cosmetic.get(sp0, sp0)
            item, ab, moves =  toid(f[2]), toid(f[3]), [toid(x) for x in f[4].split(",") if x]
            mega = None
            if item in stone_to_forme and forme_info.get(stone_to_forme[item], {}).get("from", [None])[0] == sp:
                mega = stone_to_forme[item]
            mons.append({"species": sp, "mega": mega, "item": item, "ability": ab,
                         "mega_ability": forme_info[mega]["abilities"][0] if mega else None, "moves": moves})
        if ok and len(mons) == 6:
            teams.append({"rating": r.get("game_rating"), "mons": mons})
    return teams


def real_team_cov(teams, dex, strict=True):
    """Fraction of real teams fully expressible. strict: base-forme ability of mega holders must be in dex."""
    sp, mv, ab, it = dex["species_ids"], dex["move"], dex["ability"], dex["item"]
    n_ok = 0
    fail = collections.Counter()
    for t in teams:
        miss = set()
        for m in t["mons"]:
            if m["species"] not in sp:
                miss.add(("species", m["species"]))
            if m["mega"] and m["mega"] not in sp:
                miss.add(("species", m["mega"]))
            if m["item"] and m["item"] not in it:
                miss.add(("item", m["item"]))
            if m["mega"]:
                if m["mega_ability"] not in ab:
                    miss.add(("ability", m["mega_ability"]))
                if strict and m["ability"] not in ab:
                    miss.add(("ability", m["ability"]))
            elif m["ability"] not in ab:
                miss.add(("ability", m["ability"]))
            for x in m["moves"]:
                if x not in mv:
                    miss.add(("move", x))
        if not miss:
            n_ok += 1
        for x in miss:
            fail[x] += 1
    return n_ok / max(1, len(teams)), fail


# ----------------------------------------------------------------------------------------------
class Census:
    def __init__(self, raw):
        self.raw = raw
        self.rec = raw["records"]
        self.engine = list(raw["engine_conditions"])
        self.tmin = raw["template_min"]

    def own_fns(self, kind, eid):
        r = self.rec[kind][eid]
        return r["fns"]

    def cond_refs(self, kind, eid):
        """Conditions an entity can create/inspect (data refs + code refs)."""
        r = self.rec[kind].get(eid)
        if not r:
            return set()
        out = set()
        for src in (r.get("data_refs") or {}, r.get("code_refs") or {}):
            for k in ("volatile", "sideCondition", "slotCondition", "pseudoWeather", "weather", "terrain", "status"):
                for c in src.get(k, []):
                    out.add(toid(c))
        if r.get("owned_condition") and r["owned_condition"]["fns"]:
            out.add(eid)
        return out

    def closure(self, dex):
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
            for k in ("volatile", "sideCondition", "slotCondition", "pseudoWeather", "weather", "terrain", "status"):
                for x in cr.get("code_refs", {}).get(k, []):
                    if x not in conds:
                        frontier.add(x)
        return {c for c in conds if c in self.rec["condition"] and c not in UNREACHABLE_CONDITIONS}

    def fn_list(self, dex, conds):
        """All handler functions of the dex (entity fns + condition fns, owned conditions counted once)."""
        out = []
        for kind in KINDS:
            for eid in dex[kind]:
                for f in self.rec[kind][eid]["fns"]:
                    out.append((kind, eid, f))
        for c in conds:
            for f in self.rec["condition"][c]["fns"]:
                out.append(("condition", c, f))
        return out

    def entity_templates(self, kind, eid):
        """Templates needed by an entity incl. its owned/referenced conditions (first level)."""
        ts = {f["template"] for f in self.rec[kind][eid]["fns"]}
        for c in self.cond_refs(kind, eid):
            if c in self.rec["condition"] and c not in UNREACHABLE_CONDITIONS:
                ts |= {f["template"] for f in self.rec["condition"][c]["fns"]}
        return ts

    def cost_summary(self, dex):
        conds = self.closure(dex)
        fns = self.fn_list(dex, conds)
        templates = {f["template"] for _, _, f in fns}
        by_class = {}
        for kind in KINDS:
            c = collections.Counter(self.rec[kind][e]["class"] for e in dex[kind])
            by_class[kind] = dict(c)
        by_class["condition"] = dict(collections.Counter(self.rec["condition"][c]["class"] for c in conds))
        hooks = collections.defaultdict(set)
        hook_ents = collections.defaultdict(set)
        for kind, eid, f in fns:
            hooks[f["hook"]].add(f["template"])
            hook_ents[f["hook"]].add(f"{kind}:{eid}")
        return {
            "n_conditions": len(conds), "n_handler_fns": len(fns), "n_distinct_templates": len(templates),
            "loc": sum(f["loc"] for _, _, f in fns), "classes": by_class,
            "n_custom_entities": sum(by_class[k].get("CUSTOM", 0) for k in by_class),
            "n_hook_names": len(hooks), "max_branches_per_hook": max((len(v) for v in hooks.values()), default=0),
            "top_hooks_by_branches": sorted(((h, len(v), len(hook_ents[h])) for h, v in hooks.items()), key=lambda x: -x[1])[:12],
        }, conds


# ----------------------------------------------------------------------------------------------
def main():
    dex_dir, legal_path, census_path, replay_path = sys.argv[1:5]
    specs = [a.split("=", 1) for a in sys.argv[5:]]
    legal = json.load(open(legal_path))
    census = Census(json.load(open(census_path)))
    fmts = {}
    for label, p in specs:
        info, data = load_chaos(p)
        for nm, s in data.items():
            s["name"] = nm
        fmts[label] = {"info": info, "data": data, "tables": entity_tables(data)}
    syn = json.load(open(os.path.join(dex_dir, "out", "synthetic_teams.json")))
    rteams = load_replay_teams(replay_path, legal)
    print("replay teams", len(rteams), file=sys.stderr)

    megastones = {k for k, v in legal["items"].items() if v["isMegaStone"]}
    forme_from = {k: v["from"][0] for k, v in legal["battleFormes"].items()}
    chaos_name_by_id = {}
    for f in fmts.values():
        for nm in f["data"]:
            chaos_name_by_id[toid(nm)] = nm
    legal_species_ids = set(legal["species"]) | {k for k in legal["battleFormes"] if not k.endswith("gmax")}

    # replay-derived frequencies (fraction of real teams containing X); base abilities of mega holders
    rep_freq = {"move": collections.Counter(), "ability": collections.Counter(), "item": collections.Counter(), "species": collections.Counter()}
    mega_base_ab = collections.defaultdict(collections.Counter)
    for t in rteams:
        seen = collections.defaultdict(set)
        for m in t["mons"]:
            seen["species"].add(m["mega"] or m["species"])
            seen["item"].add(m["item"])
            seen["ability"].add(m["ability"])
            if m["mega"]:
                seen["ability"].add(m["mega_ability"])
                mega_base_ab[m["mega"]][m["ability"]] += 1
            for x in m["moves"]:
                seen["move"].add(x)
        for k, v in seen.items():
            for x in v:
                rep_freq[k][x] += 1
    nrt = max(1, len(rteams))
    rep_p = {k: {x: c / nrt for x, c in v.items()} for k, v in rep_freq.items()}
    n_two_megas = sum(1 for t in rteams if sum(1 for m in t["mons"] if m["mega"]) >= 2)

    def p_team(kind, eid, label):
        tk = {"move": "moves", "ability": "abilities", "item": "items", "species": "species"}[kind]
        tab = fmts[label]["tables"][tk]
        key = chaos_name_by_id.get(eid, eid) if kind == "species" else eid
        v = tab.get(key)
        return v["p_team"] if v else 0.0

    def mass(kind, eid, label):
        tk = {"move": "moves", "ability": "abilities", "item": "items", "species": "species"}[kind]
        tab = fmts[label]["tables"][tk]
        key = chaos_name_by_id.get(eid, eid) if kind == "species" else eid
        v = tab.get(key)
        return v["mass"] if v else 0.0

    def make_dex(species_ids, moves, abilities, items):
        sp = set(species_ids)
        sp |= {forme_from[s] for s in sp if s in forme_from}  # megas imply their base forme
        return {"species_ids": sp, "species": {chaos_name_by_id[s] for s in sp if s in chaos_name_by_id},
                "move": set(moves), "ability": set(abilities), "item": set(items),
                "moves": set(moves), "abilities": set(abilities), "items": set(items)}

    def evaluate(name, dex, note=""):
        cost, conds = census.cost_summary(dex)
        cov = {}
        for label in TARGETS:
            if label in fmts:
                tc = team_cov(fmts[label]["data"], dex, syn.get(label))
                cov[label] = {k: round(v, 4) for k, v in tc.items()}
        rs, fail = real_team_cov(rteams, dex, strict=True)
        rl, _ = real_team_cov(rteams, dex, strict=False)
        return {"name": name, "note": note,
                "sizes": {"species_set_level": len([s for s in dex["species_ids"] if s in legal["species"]]),
                          "species_incl_battle_formes": len(dex["species_ids"]),
                          "mega_formes": len([s for s in dex["species_ids"] if legal["battleFormes"].get(s, {}).get("isMega")]),
                          "moves": len(dex["move"]), "abilities": len(dex["ability"]), "items": len(dex["item"]),
                          "megastones": len(dex["item"] & megastones)},
                "cost": cost, "coverage_chaos": cov,
                "coverage_replays_bo3": {"strict": round(rs, 4), "lenient_mega_base_ability": round(rl, 4), "n_teams": len(rteams)},
                "top_replay_failures": [[f"{k}:{v}", c] for (k, v), c in fail.most_common(15)]}, conds

    options = []
    # (1) mass-based, 99% per category at bo3_1760 (the background doc's dex)
    def mass_top(label, kind, thr):
        tk = {"move": "moves", "ability": "abilities", "item": "items", "species": "species"}[kind]
        items = sorted(fmts[label]["tables"][tk].items(), key=lambda x: -x[1]["mass"])
        out, c = [], 0.0
        for k, v in items:
            if c >= thr - 1e-12:
                break
            out.append(toid(k) if kind == "species" else k)
            c += v["mass"]
        return out
    for thr in (0.95, 0.99):
        d = make_dex(mass_top("bo3_1760", "species", thr), mass_top("bo3_1760", "move", thr),
                     mass_top("bo3_1760", "ability", thr), mass_top("bo3_1760", "item", thr))
        options.append(evaluate(f"mass{int(thr*100)}_bo3_1760", d, "top entities to reach this share of slot mass per category")[0])

    # (2) usage thresholds on P(team contains)
    def tau_dex(taus):
        sp, mv, ab, it = set(), set(), set(), set()
        for label, tau in taus.items():
            t = fmts[label]["tables"]
            sp |= {toid(k) for k, v in t["species"].items() if v["p_team"] >= tau}
            mv |= {k for k, v in t["moves"].items() if v["p_team"] >= tau}
            ab |= {k for k, v in t["abilities"].items() if v["p_team"] >= tau}
            it |= {k for k, v in t["items"].items() if v["p_team"] >= tau}
        return sp, mv, ab, it

    for taus in ({"bo3_1760": 1e-3}, {"bo3_1760": 1e-4}, {"bo3_1760": 1e-4, "bo1_0": 5e-4}):
        sp, mv, ab, it = tau_dex(taus)
        # add base abilities of mega holders seen on >= tau of replay teams
        tau_min = min(taus.values())
        for mf in [s for s in sp if s in legal["battleFormes"]]:
            for a, c in mega_base_ab.get(mf, {}).items():
                if c / nrt >= tau_min:
                    ab.add(a)
        d = make_dex(sp, mv, ab, it)
        options.append(evaluate("usage_" + "+".join(f"{k}>={v:g}" for k, v in taus.items()), d,
                                "every entity on >= tau of teams (P(team contains)) in the listed formats")[0])

    # (3) cost-aware: all legal species/formes; free entities (no new bespoke templates beyond engine core);
    #     paid entities selected by usage; then free riders whose templates are already paid for.
    engine_templates = set()
    for c in census.closure({"move": set(), "ability": set(), "item": set()}):
        engine_templates |= {f["template"] for f in census.rec["condition"][c]["fns"]}
    all_ents = [(k, e) for k in KINDS for e in census.rec[k]]
    ent_tmpl = {(k, e): census.entity_templates(k, e) for k, e in all_ents}

    def max_p(kind, eid, labels):
        ps = [p_team(kind, eid, l) for l in labels]
        if kind == "ability":
            ps.append(rep_p["ability"].get(eid, 0.0) if eid in {a for c in mega_base_ab.values() for a in c} else 0.0)
        return max(ps)

    def cost_dex(tau_by_label):
        chosen = set()
        tmpl = set(engine_templates)
        for k, e in all_ents:
            if ent_tmpl[(k, e)] <= engine_templates:
                chosen.add((k, e))
        for k, e in all_ents:
            if any(max_p(k, e, [l]) >= tau for l, tau in tau_by_label.items()):
                chosen.add((k, e))
                tmpl |= ent_tmpl[(k, e)]
        changed = True
        while changed:
            changed = False
            for k, e in all_ents:
                if (k, e) not in chosen and ent_tmpl[(k, e)] <= tmpl:
                    chosen.add((k, e))
                    changed = True
        sp = legal_species_ids
        return make_dex(sp, [e for k, e in chosen if k == "move"], [e for k, e in chosen if k == "ability"],
                        [e for k, e in chosen if k == "item"])

    sweep = []
    for tau in (1e-2, 3e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1e-4, 5e-5, 2e-5, 0.0):
        d = cost_dex({"bo3_1760": tau, "bo1_0": tau} if tau > 0 else {"bo3_1760": -1})
        r, _ = evaluate(f"cost_aware_tau={tau:g}", d, "all legal species; zero-cost entities + paid entities with P(team)>=tau in bo3_1760 or bo1_0 + free riders")
        sweep.append(r)
    options += sweep
    for tau in (3e-3, 1e-3, 3e-4, 1e-4):
        d = cost_dex({"bo3_1760": tau})
        options.append(evaluate(f"cost_aware_bo3_1760_only_tau={tau:g}", d, "as cost_aware but usage threshold only on bo3_1760")[0])

    # (4) full legal pool
    full = make_dex(legal_species_ids, census.rec["move"], census.rec["ability"], census.rec["item"])
    full_eval, full_conds = evaluate("full_legal_pool (v2)", full)
    options.append(full_eval)

    # ---- choose v1: smallest cost-aware tau meeting targets ------------------------------
    def meets(o):
        c = o["coverage_chaos"]
        return c["bo3_1760"]["team_cov_mc"] >= 0.99 and c["bo1_0"]["team_cov_mc"] >= 0.95 and o["coverage_replays_bo3"]["strict"] >= 0.95
    chosen = None
    for o in sweep:  # sweep is ordered from high tau (small dex) to low tau
        if meets(o):
            chosen = o
            break
    if chosen is None:
        chosen = sweep[-2]
    v1_tau = float(chosen["name"].split("=")[1])
    v1 = cost_dex({"bo3_1760": v1_tau, "bo1_0": v1_tau})
    v1_eval, v1_conds = evaluate("v1", v1)

    # ---- write v1 lists -------------------------------------------------------------------
    labels_out = [l for l in ("bo3_1760", "bo3_0", "bo1_1760", "bo1_0") if l in fmts]

    def usage_block(kind, eid):
        u = {l: {"p_team": round(p_team(kind, eid, l), 6), "slot_mass": round(mass(kind, eid, l), 6)} for l in labels_out}
        rk = {"species": "species"}.get(kind, kind)
        u["replays_bo3_p_team"] = round(rep_p[rk].get(eid, 0.0), 5)
        return u

    def reason(kind, eid):
        if kind != "species" and ent_tmpl[(kind, eid)] <= engine_templates:
            return "zero_cost"
        if kind != "species" and max_p(kind, eid, ["bo3_1760", "bo1_0"]) >= v1_tau:
            return "usage"
        return "free_rider_shared_template" if kind != "species" else "all_legal_species"

    def ent_row(kind, eid):
        r = census.rec[kind][eid]
        return {"id": eid, "name": r["name"], "class": r["class"], "inclusion": reason(kind, eid),
                "n_hooks": r["n_hooks"], "hooks": r["hooks"], "condition_hooks": r["condition_hooks"],
                "usage": usage_block(kind, eid)}

    meta = {"format": legal["format"], "selection": f"cost-aware, tau={v1_tau:g} on P(team contains) in bo3_1760 or bo1_0",
            "stats_month": "2026-09", "ps_commit": "d601443"}
    sp_rows = []
    for s in sorted(v1["species_ids"]):
        info = legal["species"].get(s) or legal["battleFormes"].get(s)
        row = {"id": s, "name": info["name"], "battle_forme": s in legal["battleFormes"],
               "is_mega": bool(legal["battleFormes"].get(s, {}).get("isMega")),
               "base": forme_from.get(s), "types": info["types"], "baseStats": info["baseStats"],
               "abilities": [a["id"] for a in info["abilities"]] if isinstance(info["abilities"], list) and info["abilities"] and isinstance(info["abilities"][0], dict) else info["abilities"],
               "required_item": info.get("requiredItem"), "usage": usage_block("species", s),
               "observed_in_chaos": s in chaos_name_by_id}
        if row["is_mega"]:
            row["base_ability_dist_replays"] = dict(mega_base_ab.get(s, {}))
        sp_rows.append(row)
    sp_rows.sort(key=lambda r: -r["usage"]["bo3_1760"]["p_team"])
    json.dump({"meta": meta, "count": len(sp_rows), "species": sp_rows}, open(os.path.join(dex_dir, "v1_species.json"), "w"), indent=1)
    for kind, fname in (("move", "v1_moves.json"), ("ability", "v1_abilities.json"), ("item", "v1_items.json")):
        rows = sorted((ent_row(kind, e) for e in v1[kind]), key=lambda r: -r["usage"]["bo3_1760"]["p_team"])
        excluded = sorted(({"id": e, "class": census.rec[kind][e]["class"],
                            "max_p_team_bo3_1760_bo1_0": round(max_p(kind, e, ["bo3_1760", "bo1_0"]), 6)}
                           for e in census.rec[kind] if e not in v1[kind]), key=lambda r: -r["max_p_team_bo3_1760_bo1_0"])
        json.dump({"meta": meta, "count": len(rows), "entries": rows, "excluded_legal": excluded},
                  open(os.path.join(dex_dir, fname), "w"), indent=1)

    # ---- handler census outputs -------------------------------------------------------------
    def census_out(dex, conds, label, ev):
        ents = {k: {e: census.rec[k][e] for e in sorted(dex[k])} for k in KINDS}
        ents["condition"] = {c: census.rec["condition"][c] for c in sorted(conds)}
        hook_table = {}
        for k, group in ents.items():
            ht = collections.defaultdict(lambda: {"entities": [], "templates": set()})
            for eid, r in group.items():
                fns = list(r["fns"])
                for f in fns:
                    h = ht[f["path"] if not f["path"].startswith("condition.") else f["path"]]
                    h["entities"].append(eid)
                    h["templates"].add(f["template"])
            hook_table[k] = {h: {"n_entities": len(v["entities"]), "n_distinct_templates": len(v["templates"]),
                                 "entities": sorted(set(v["entities"]))}
                             for h, v in sorted(ht.items(), key=lambda x: -len(x[1]["entities"]))}
        # merged by bare hook name across entity types (what a dispatcher would switch on)
        merged = collections.defaultdict(lambda: {"by_kind": collections.Counter(), "templates": set()})
        for k, group in ents.items():
            for eid, r in group.items():
                for f in r["fns"]:
                    merged[f["hook"]]["by_kind"][k] += 1
                    merged[f["hook"]]["templates"].add(f["template"])
        merged_out = {h: {"by_kind": dict(v["by_kind"]), "n_entities": sum(v["by_kind"].values()),
                          "n_distinct_templates": len(v["templates"])}
                      for h, v in sorted(merged.items(), key=lambda x: -sum(x[1]["by_kind"].values()))}
        # data-only field frequency
        field_freq = {}
        for k in KINDS:
            c = collections.Counter()
            for r in ents[k].values():
                for f in r["data"]:
                    c[f.split(".")[0].split("[")[0]] += 1
            field_freq[k] = dict(c.most_common())
        cls = {k: dict(collections.Counter(r["class"] for r in g.values())) for k, g in ents.items()}
        return {"label": label, "meta": meta if label == "v1" else {"format": legal["format"], "ps_commit": "d601443"},
                "summary": {"counts": {k: len(g) for k, g in ents.items()}, "classes": cls, "cost": ev["cost"]},
                "hook_frequency_by_kind": hook_table, "hook_frequency_merged": merged_out,
                "data_field_frequency": field_freq,
                "classification_rules": {
                    "DATA_ONLY": "no function-valued keys anywhere (incl. secondary/self/owned condition); references only core conditions brn/par/psn/tox/slp/frz/confusion/flinch",
                    "DATA_PLUS_COMMON": f"every function is a parametric template shared by >= {census.tmin} legal entities (e.g. type-resist berries, mega stones, Surge setters, Protect wrapper), and/or it sets a named volatile/side/slot condition, pseudo-weather, weather or terrain whose handlers are counted once under 'condition'",
                    "CUSTOM": "at least one bespoke function (template shared by < 3 legal entities)"},
                "entities": ents}
    json.dump(census_out(v1, v1_conds, "v1", v1_eval), open(os.path.join(dex_dir, "handler_census_v1.json"), "w"))
    json.dump(census_out(full, full_conds, "legal", full_eval), open(os.path.join(dex_dir, "handler_census_legal.json"), "w"))
    json.dump({"options": options, "v1": v1_eval, "v1_tau": v1_tau, "replay_teams": len(rteams),
               "replay_teams_with_2plus_megastones": n_two_megas,
               "mega_base_abilities": {k: dict(v) for k, v in mega_base_ab.items()}},
              open(os.path.join(dex_dir, "out", "dex_options.json"), "w"), indent=1)
    for o in options + [v1_eval]:
        c = o["coverage_chaos"]
        print(f"{o['name']:<45} sizes={o['sizes']} custom={o['cost']['n_custom_entities']} tmpl={o['cost']['n_distinct_templates']} "
              f"fns={o['cost']['n_handler_fns']} conds={o['cost']['n_conditions']} | "
              + " ".join(f"{l}:{c[l]['team_cov_mc']:.3f}/{c[l]['team_cov_indep']:.3f}" for l in TARGETS if l in c)
              + f" | replays strict {o['coverage_replays_bo3']['strict']:.3f} lenient {o['coverage_replays_bo3']['lenient_mega_base_ability']:.3f}")


if __name__ == "__main__":
    main()
