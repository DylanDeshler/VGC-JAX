#!/usr/bin/env python3
"""Render markdown tables (and a complexity ranking) from the dex outputs.

Usage: python3 -I report_tables.py <dex_dir> <out_md>
Also writes <dex_dir>/out/complexity_v1.json
"""
import collections
import json
import os
import re
import sys

NOTORIOUS = {
    "commander": "Commander (Tatsugiri/Dondozo)", "allyswitch": "Ally Switch", "protosynthesis": "Paradox",
    "quarkdrive": "Paradox", "illusion": "Illusion", "transform": "Transform", "imposter": "Imposter",
    "dancer": "Dancer", "instruct": "Instruct", "encore": "Encore", "disguise": "Disguise (form change)",
    "stancechange": "Stance Change (form change)", "zerotohero": "Zero to Hero (form change)",
    "hungerswitch": "Hunger Switch (form change)", "forecast": "Forecast (form change)", "trace": "Trace",
    "skillswap": "Skill Swap", "roleplay": "Role Play", "entrainment": "Entrainment", "doodle": "Doodle",
    "copycat": "Copycat", "metronome": "Metronome", "mefirst": "Me First", "mirrormove": "Mirror Move",
    "snatch": "Snatch", "magicbounce": "Magic Bounce", "magiccoat": "Magic Coat", "afteryou": "After You",
    "quash": "Quash", "pursuit": "Pursuit", "substitute": "Substitute", "batonpass": "Baton Pass",
    "shedtail": "Shed Tail", "revivalblessing": "Revival Blessing", "neutralizinggas": "Neutralizing Gas",
    "receiver": "Receiver", "powerofalchemy": "Power of Alchemy", "costar": "Costar", "terashell": "Tera Shell",
    "followme": "redirection", "ragepowder": "redirection", "lightningrod": "redirection", "stormdrain": "redirection",
    "emergencyexit": "mid-turn switch", "wimpout": "mid-turn switch", "ejectbutton": "mid-turn switch",
    "ejectpack": "mid-turn switch", "redcard": "forced switch", "uturn": "mid-turn switch",
    "voltswitch": "mid-turn switch", "partingshot": "mid-turn switch", "flipturn": "mid-turn switch",
    "teleport": "mid-turn switch", "chillyreception": "mid-turn switch", "focuspunch": "priority charge",
    "beakblast": "priority charge", "shelltrap": "priority charge", "futuresight": "delayed slot effect",
    "wish": "delayed slot effect", "healingwish": "slot condition", "perishsong": "perish counter",
    "gravity": "field rule changes", "magicroom": "item suppression", "wonderroom": "stat swap",
    "trickroom": "turn-order inversion", "fling": "item consumption", "trick": "item swap",
    "switcheroo": "item swap", "knockoff": "item removal", "covet": "item theft", "thief": "item theft",
    "unburden": "item-loss trigger", "symbiosis": "item passing", "pickpocket": "item theft",
    "magician": "item theft", "harvest": "berry regen", "cudchew": "berry re-eat", "ripen": "berry modifier",
    "moldbreaker": "ability suppression", "teravolt": "ability suppression", "turboblaze": "ability suppression",
    "gastroacid": "ability suppression", "simplebeam": "ability change", "worryseed": "ability change",
    "mimicry": "type change", "protean": "type change", "libero": "type change", "colorchange": "type change",
    "soak": "type change", "magicpowder": "type change", "forestscurse": "type change", "trickortreat": "type change",
    "burnup": "type change", "roost": "type change", "electrify": "move type change", "iondeluge": "move type change",
    "skydrop": "semi-invuln two-target", "bide": "multi-turn", "uproar": "multi-turn", "rollout": "multi-turn",
    "iceball": "multi-turn", "outrage": "locked move", "petaldance": "locked move", "thrash": "locked move",
    "ragingfury": "locked move", "disable": "move lock", "taunt": "move lock", "torment": "move lock",
    "imprison": "move lock", "healblock": "move lock", "throatchop": "move lock", "spite": "pp", "grudge": "pp",
    "pressure": "pp", "lastrespects": "faint counter", "ragefist": "hit counter", "supremeoverlord": "faint counter",
    "stall": "turn order", "quickclaw": "fractional priority", "custapberry": "fractional priority",
    "mycelium might": "turn order", "myceliummight": "fractional priority", "prankster": "priority + immunity",
    "galewings": "priority", "triage": "priority", "psychicterrain": "priority block", "queenlymajesty": "priority block",
    "dazzling": "priority block", "armortail": "priority block", "quickguard": "priority block",
    "wideguard": "spread block", "intimidate": "switch-in boost chain", "defiant": "boost reaction",
    "competitive": "boost reaction", "mirrorarmor": "boost reflection", "contrary": "boost inversion",
    "simple": "boost scaling", "unaware": "boost ignoring", "megalauncher": "mega", "dragonize": "Champions mega ability",
    "megasol": "Champions mega ability", "piercingdrill": "Champions mega ability", "spicyspray": "Champions mega ability",
    "firemane": "Champions mega ability", "eelevate": "Champions mega ability", "auraguard": "Champions mega ability",
    "fairyaura": "aura", "darkaura": "aura", "aurabreak": "aura", "parentalbond": "multi-hit ability",
    "skilllink": "multi-hit", "sheerforce": "secondary suppression", "serenegrace": "secondary scaling",
    "shielddust": "secondary immunity", "covertcloak": "secondary immunity", "analytic": "turn-order dependent",
    "stakeout": "switch dependent", "comatose": "status", "commandeer": "", "zenmode": "form change",
    "schooling": "form change", "powerconstruct": "form change", "shieldsdown": "form change",
    "gulpmissile": "form change", "iceface": "form change", "battlebond": "form change", "flowergift": "form change",
}


def esc(x):
    return str(x).replace("|", "\\|")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(out)


def main():
    dex_dir, out_md = sys.argv[1], sys.argv[2]
    raw = json.load(open(os.path.join(dex_dir, "out", "census_raw_legal.json")))["records"]
    v1c = json.load(open(os.path.join(dex_dir, "handler_census_v1.json")))
    lgc = json.load(open(os.path.join(dex_dir, "handler_census_legal.json")))
    opts = json.load(open(os.path.join(dex_dir, "out", "dex_options.json")))
    cov = json.load(open(os.path.join(dex_dir, "out", "coverage_summary.json")))
    md = []

    # 1. mass curves
    rows = []
    for label, f in cov["formats"].items():
        mc = f["mass_curves"]
        def c(k):
            v = mc[k]
            return f"{v['0.9']}/{v['0.95']}/{v['0.99']}/{v['0.999']} ({v['total']})"
        rows.append([label, f["battles"], c("species"), c("moves"), c("abilities"), c("items_excl_megastones"), c("megastones"),
                     f"{f['mega_slot_share']:.3f}"])
    md.append("### T1. Slot-mass coverage curves (entities needed for 90/95/99/99.9% of slot mass; total observed)\n")
    md.append(table(["format", "battles", "species (incl. megas)", "moves", "abilities", "items (non-stone)", "mega stones", "mega slot share"], rows))

    # 2. dex options
    rows = []
    for o in opts["options"]:
        s, c, cc, r = o["sizes"], o["cost"], o["coverage_chaos"], o["coverage_replays_bo3"]
        rows.append([o["name"], f"{s['species_set_level']}+{s['mega_formes']}M", s["moves"], s["abilities"], f"{s['items']} ({s['megastones']} stones)",
                     c["n_custom_entities"], c["n_conditions"], c["n_distinct_templates"], c["n_handler_fns"],
                     f"{cc['bo3_1760']['team_cov_mc']:.3f}", f"{cc['bo1_1760']['team_cov_mc']:.3f}", f"{cc['bo3_0']['team_cov_mc']:.3f}",
                     f"{cc['bo1_0']['team_cov_mc']:.3f}", f"{r['strict']:.3f}"])
    md.append("\n### T2. Candidate dexes: size, handler cost, team coverage\n")
    md.append(table(["dex", "species", "moves", "abilities", "items", "CUSTOM ents (incl. conds)", "conditions", "distinct handler templates", "handler fns",
                     "team cov 1760 Bo3", "1760 Bo1", "all Bo3", "all Bo1", "real Bo3 replay teams (strict)"], rows))

    # 3. hook tables (v1) by kind
    for label, cen in (("v1", v1c), ("legal", lgc)):
        s = cen["summary"]
        rows = []
        for k in ("move", "ability", "item", "condition"):
            cl = s["classes"].get(k, {})
            rows.append([k, s["counts"][k], cl.get("DATA_ONLY", 0), cl.get("DATA_PLUS_COMMON", 0), cl.get("CUSTOM", 0)])
        md.append(f"\n### T3-{label}. Classification counts ({label})\n")
        md.append(table(["kind", "n", "DATA_ONLY", "DATA_PLUS_COMMON", "CUSTOM"], rows))

    def event_of(h):
        m = re.match(r"^on(Foe|Ally|Source|Any)?(.*)$", h)
        return m.group(2) if m else h

    for label, cen in (("v1", v1c), ("legal", lgc)):
        merged = cen["hook_frequency_merged"]
        rows = []
        for h, v in list(merged.items())[:45]:
            bk = v["by_kind"]
            rows.append([h, v["n_entities"], v["n_distinct_templates"], bk.get("move", 0), bk.get("ability", 0), bk.get("item", 0), bk.get("condition", 0)])
        md.append(f"\n### T4-{label}. Hook frequency (top 45 of {len(merged)} hook names; templates = distinct branches after parametric dedup)\n")
        md.append(table(["hook", "entities", "distinct templates", "moves", "abilities", "items", "conditions"], rows))
        # event-level aggregation
        ev = collections.defaultdict(lambda: [0, 0, set()])
        for h, v in merged.items():
            e = event_of(h)
            ev[e][0] += v["n_entities"]
            ev[e][1] += v["n_distinct_templates"]
            ev[e][2].add(h)
        md.append(f"\n{label}: {len(merged)} hook names collapse to {len(ev)} events after stripping Foe/Ally/Source/Any prefixes.\n")

    # 4. per-kind hook tables with entity names (v1)
    md.append("\n### T5. v1 hook -> entity lists, by entity type (paths; `condition.*` = handlers of the entity's own condition)\n")
    for k in ("move", "ability", "item", "condition"):
        ht = v1c["hook_frequency_by_kind"][k]
        rows = []
        for h, v in ht.items():
            names = v["entities"]
            shown = ", ".join(names[:40]) + (f", ... (+{len(names)-40})" if len(names) > 40 else "")
            rows.append([h, v["n_entities"], v["n_distinct_templates"], shown])
        md.append(f"\n#### {k} ({len(ht)} hook paths)\n")
        md.append(table(["hook", "n", "templates", "entities"], rows))

    # 5. data field frequency (v1)
    md.append("\n### T6. v1 data-only field frequency\n")
    for k in ("move", "ability", "item"):
        ff = v1c["data_field_frequency"][k]
        md.append(f"\n**{k}**: " + ", ".join(f"`{f}` {n}" for f, n in ff.items()))

    # 6. complexity ranking (v1 CUSTOM)
    mentioned_by = collections.Counter()
    for kind in ("move", "ability", "item", "condition"):
        for eid, r in raw[kind].items():
            for mk in ("mentions_move", "mentions_ability", "mentions_item", "mentions_condition"):
                for x in (r.get("code_refs") or {}).get(mk, []):
                    if x != eid:
                        mentioned_by[x] += 1
    ents = v1c["entities"]
    rank = []
    for kind in ("move", "ability", "item", "condition"):
        for eid, r in ents[kind].items():
            if r["class"] != "CUSTOM":
                continue
            own = r.get("owned_condition") or {}
            fns = r["fns"] + (own.get("fns") or [] if kind != "condition" else [])
            loc = sum(f["loc"] for f in fns)
            core = sum((r.get("core_refs") or {}).values())
            mb = mentioned_by.get(eid, 0)
            hooks = sorted({f["hook"] for f in fns})
            score = loc + 6 * len(fns) + 4 * core + 3 * mb
            flag = NOTORIOUS.get(eid, "")
            rank.append({"kind": kind, "id": eid, "score": score, "fns": len(fns), "loc": loc, "core_refs": core,
                         "mentioned_by": mb, "hooks": hooks, "flag": flag})
    rank.sort(key=lambda x: -x["score"])
    json.dump(rank, open(os.path.join(dex_dir, "out", "complexity_v1.json"), "w"), indent=1)
    rows = [[i + 1, x["kind"], x["id"], x["score"], x["fns"], x["loc"], x["core_refs"], x["mentioned_by"], x["flag"], ", ".join(x["hooks"][:8])]
            for i, x in enumerate(rank[:60])]
    md.append("\n### T7. v1 CUSTOM entities ranked by complexity score (= LoC + 6*fns + 4*core-code refs + 3*mentions by other entities)\n")
    md.append(table(["#", "kind", "id", "score", "fns", "LoC", "core refs", "mentioned by", "flag", "hooks"], rows))
    # notorious presence
    legal_ids = {k: set(lgc["entities"][k]) for k in ("move", "ability", "item", "condition")}
    v1_ids = {k: set(ents[k]) for k in ("move", "ability", "item", "condition")}
    rows = []
    for nid in ["commander", "allyswitch", "protosynthesis", "quarkdrive", "illusion", "transform", "imposter", "dancer",
                "instruct", "encore", "disguise", "stancechange", "zerotohero", "hungerswitch", "forecast", "trace",
                "skillswap", "roleplay", "magicbounce", "neutralizinggas", "substitute", "batonpass", "shedtail",
                "revivalblessing", "afteryou", "quash", "followme", "ragepowder", "lightningrod", "stormdrain",
                "wideguard", "quickguard", "fakeout", "trickroom", "tailwind", "gravity", "perishsong", "unburden",
                "dragonize", "megasol", "piercingdrill", "spicyspray", "firemane", "eelevate", "auraguard"]:
        where = [k for k in legal_ids if nid in legal_ids[k]]
        inv1 = [k for k in v1_ids if nid in v1_ids[k]]
        rows.append([nid, ",".join(where) or "not legal", "yes" if inv1 else ("no" if where else "-"), NOTORIOUS.get(nid, "")])
    md.append("\n### T8. Notorious mechanics: legal in Reg M-C? in v1?\n")
    md.append(table(["id", "legal as", "in v1", "note"], rows))
    open(out_md, "w").write("\n".join(md) + "\n")
    print("wrote", out_md, len(rank), "custom ranked")


if __name__ == "__main__":
    main()
