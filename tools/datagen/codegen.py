"""Codegen: data/export (+ stages) -> vgcjax/data/{ids.py, tables.npz}  (06 §1, 03 §1-2, M1.1).

ID spaces are built from the full legal pool (stage v2), so ids are identical in every stage; each stage only flips
`supported_<stage>` masks (03 §2). Everything is deterministic: same inputs -> byte-identical outputs.

    uv run python tools/datagen/codegen.py [--export data/export] [--out vgcjax/data]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import pathlib
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGES = ("core", "v0", "v1", "v2", "impl")

TYPES = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic",
         "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]  # TypeId 1..18; 19 = '???'; 0 = none
TYPELESS = 19
NTYPES = 20
STATS6 = ["hp", "atk", "def", "spa", "spd", "spe"]
BOOSTS7 = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
CATEGORIES = ["Physical", "Special", "Status"]
TARGETS = ["normal", "self", "adjacentAlly", "adjacentAllyOrSelf", "adjacentFoe", "allAdjacentFoes", "allAdjacent",
           "all", "allySide", "foeSide", "allies", "any", "randomNormal", "scripted", "allyTeam"]
MOVE_FLAGS = ["allyanim", "bite", "bullet", "bypasssub", "cantusetwice", "charge", "contact", "dance", "defrost",
              "distance", "failcopycat", "failencore", "failinstruct", "failmefirst", "failmimic", "futuremove",
              "gravity", "heal", "metronome", "mirror", "mustpressure", "noassist", "nonsky", "noparentalbond",
              "nosketch", "nosleeptalk", "pledgecombo", "powder", "protect", "pulse", "punch", "recharge",
              "reflectable", "slicing", "snatch", "sound", "wind"]
ABILITY_FLAGS = ["breakable", "cantsuppress", "failroleplay", "failskillswap", "noentrain", "noreceiver", "notrace",
                 "notransform"]
SELF_SWITCH = {None: 0, True: 1, "copyvolatile": 2, "shedtail": 3}
STATUSES = ["brn", "par", "psn", "tox", "slp", "frz", "fnt"]
COND_KINDS = ["STATUS", "VOL", "SIDE", "SLOT", "PW", "WEATHER", "TERRAIN"]
SLOT_CONDITIONS = {"wish", "healingwish", "revivalblessing", "futuremove", "healreplacement"}  # PS addSlotCondition ids
IMMUNITY_KEYS = ["brn", "par", "psn", "tox", "frz", "sandstorm", "hail", "powder", "prankster", "trapped"]
# step_subject codes for non-move actions (contract §3: "the running action's move id or action-kind code")
ACTION_KINDS = ["switch", "instaswitch", "megaEvo", "beforeTurn", "beforeTurnMove", "revivalblessing",
                "priorityChargeMove", "start", "team", "residual", "runSwitch", "commit"]
# contract/drawkey.md §5 (forced) then §4 (tie kinds); order fixes the 8-bit DrawKind codes
DRAW_KINDS = ["ACCURACY", "CRIT", "DAMAGE_ROLL", "CONFUSION_SELFHIT_DAMAGE_ROLL", "SECONDARY", "SECONDARY_STATUS_PICK",
              "SELF_EFFECT_ROLL", "MULTIHIT_COUNT", "RANDOM_TARGET", "FRAC_PRIO", "PARALYSIS", "SLEEP_DURATION",
              "FREEZE_THAW", "CONFUSION_DURATION", "CONFUSION_SELFHIT", "PARTIAL_TRAP_DURATION", "LOCKEDMOVE_DURATION",
              "PROTECT_STALL", "ALLY_SWITCH_STALL", "INFATUATION", "FORCED_SWITCH_PICK", "ABILITY_ONHIT_PROC",
              "ABILITY_RESIDUAL_PROC", "ABILITY_TRACE_TARGET", "ABILITY_FOREWARN_PICK", "ABILITY_PICKUP_TARGET",
              "ABILITY_MOODY_STAT", "ITEM_FOCUS_BAND", "MOVE_FICKLE_BEAM", "MOVE_SHELLSIDEARM_TIE",
              "MOVE_ACUPRESSURE_STAT", "MOVE_CALL_PICK",
              "SPEED_TIE_SELECT", "SPEED_TIE_SWITCHIN", "SPEED_TIE_FIELD_EVENT", "SPEED_TIE_WEATHER_EACH",
              "SPEED_TIE_HANDLER_MUL", "SPEED_TIE_ABILITY", "RUNSWITCH_INSERT_TIE"]  # appended: contract v2
STEP_TAGS = ["COMMIT", "A_SWITCH", "A_MEGA", "A_OTHER", "B_MOVE", "RESIDUAL", "SWITCHIN"]
# handler prefixes (PS on<Prefix><Event>); bit k of handler_mask
PREFIXES = ["", "Ally", "Foe", "Source", "Any", "Side", "Field"]
# events the engine calls even if no pool entity handles them yet
ENGINE_EVENTS = ["BasePower", "ModifyAtk", "ModifyDef", "ModifySpA", "ModifySpD", "ModifySpe", "ModifyDamage",
                 "ModifyAccuracy", "ModifyCritRatio", "ModifySTAB", "ModifyPriority", "FractionalPriority",
                 "Residual", "SwitchIn", "Update", "BeforeTurn", "Weather", "DisableMove", "TrapPokemon",
                 "MaybeTrapPokemon", "CriticalHit", "Invulnerability", "TryHit", "DamagingHit", "EntryHazard",
                 "RedirectTarget", "WeatherModifyDamage", "LockMove", "Damage", "Effectiveness", "BeforeMove"]
ORDER_LAST = 32767
SPECIES_FLAGS = ["isMega", "battleOnly", "cosmetic", "requiredItem", "nfe"]
ITEM_FLAGS = ["isBerry", "megaStone", "ignoreKlutz", "isGem", "isChoice", "hasFling"]


def load(path: pathlib.Path):
    return json.loads(path.read_text())


def toid(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def classify_conditions(conds: list[str], cexp: dict, moves: dict) -> dict[str, str]:
    """ConditionId kind per condition (03 §1 sub-ranges), from PS effectType + how PS adds it."""
    side = {toid(m["sideCondition"]) for m in moves.values() if m.get("sideCondition")}
    pw = {toid(m["pseudoWeather"]) for m in moves.values() if m.get("pseudoWeather")}
    out = {}
    for c in conds:
        e = cexp.get(c, {})
        et = e.get("effectType")
        fns = e.get("fnKeys", [])
        if et == "Status" or c in STATUSES:
            out[c] = "STATUS"
        elif et == "Weather":
            out[c] = "WEATHER"
        elif et == "Terrain":
            out[c] = "TERRAIN"
        elif c in SLOT_CONDITIONS:
            out[c] = "SLOT"
        elif c in side or any(f.startswith("onSide") for f in fns):
            out[c] = "SIDE"
        elif c in pw or any(f.startswith("onField") for f in fns):
            out[c] = "PW"
        else:
            out[c] = "VOL"
    return out


class Builder:
    def __init__(self, export: pathlib.Path):
        self.export = export
        self.species = load(export / "species.json")
        self.moves = load(export / "moves.json")
        self.abilities = load(export / "abilities.json")
        self.items = load(export / "items.json")
        self.conds = load(export / "conditions.json")
        self.typechart = load(export / "typechart.json")
        self.natures = load(export / "natures.json")
        self.megas = load(export / "megas.json")
        self.meta = load(export / "meta.json")
        self.stages = {s: load(export / "stages" / f"{s}.json") for s in STAGES}
        self.arrays: dict[str, np.ndarray] = {}
        self.consts: list[str] = []

    # ------------------------------------------------------------------ id spaces
    def build_ids(self) -> None:
        v2 = self.stages["v2"]
        species = sorted(set(v2["species"]))
        abilities = set(v2["abilities"])
        items = set(v2["items"])
        for sid in species:
            for a in self.species[sid].get("abilities", {}).values():
                abilities.add(toid(a))
        for base, lst in self.megas.items():
            if base in species:
                for e in lst:
                    items.add(e["item"])
        moves = set(v2["moves"]) | {m for m in ("struggle", "recharge") if m in self.moves}
        self.species_ids = ["none"] + species
        self.move_ids = ["none"] + sorted(moves)
        self.ability_ids = ["none"] + sorted(abilities)
        self.item_ids = ["none"] + sorted(items)
        kinds = classify_conditions(sorted(set(v2["conditions"]) | set(STATUSES)), self.conds, self.moves)
        self.cond_kind = kinds
        conds: list[str] = ["none"]
        self.cond_ranges: dict[str, tuple[int, int]] = {}
        for k in COND_KINDS:
            members = STATUSES if k == "STATUS" else sorted(c for c, kk in kinds.items() if kk == k)
            start = len(conds)
            conds += members
            self.cond_ranges[k] = (start, len(conds))
        self.cond_ids = conds
        self.nature_ids = sorted(self.natures)
        self.sid = {s: i for i, s in enumerate(self.species_ids)}
        self.mid = {s: i for i, s in enumerate(self.move_ids)}
        self.aid = {s: i for i, s in enumerate(self.ability_ids)}
        self.iid = {s: i for i, s in enumerate(self.item_ids)}
        self.cid = {s: i for i, s in enumerate(self.cond_ids)}
        self.nid = {s: i for i, s in enumerate(self.nature_ids)}
        self.tid = {t: i + 1 for i, t in enumerate(TYPES)} | {"???": TYPELESS}
        # unified EffectId (03 §1): 0 none, then moves, abilities, items, conditions, action kinds
        self.move_base = 1
        self.ability_base = self.move_base + len(self.move_ids) - 1
        self.item_base = self.ability_base + len(self.ability_ids) - 1
        self.cond_base = self.item_base + len(self.item_ids) - 1
        self.action_base = self.cond_base + len(self.cond_ids) - 1
        self.neffect = self.action_base + len(ACTION_KINDS)
        assert self.neffect < 4096, "EffectId must fit the 12-bit DrawKey subject field"

    def cond(self, name) -> int:
        if not name:
            return 0
        c = toid(name)
        if c not in self.cid:
            raise KeyError(f"condition {name} not in the ConditionId space")
        return self.cid[c]

    # ------------------------------------------------------------------ tables
    def arr(self, name: str, a) -> None:
        self.arrays[name] = np.ascontiguousarray(a)

    def build_species(self) -> None:
        n = len(self.species_ids)
        base = np.zeros((n, 6), np.int16)
        types = np.full((n, 2), -1, np.int8)
        weight = np.zeros(n, np.int16)
        abil = np.zeros((n, 4), np.int16)
        mega = np.zeros((n, 2, 2), np.int16)
        flags = np.zeros(n, np.uint32)
        base_species = np.zeros(n, np.int16)
        num = np.zeros(n, np.int16)
        for i, sid in enumerate(self.species_ids[1:], 1):
            s = self.species[sid]
            base[i] = [s["baseStats"][k] for k in STATS6]
            for j, t in enumerate(s["types"]):
                types[i, j] = self.tid[t]
            weight[i] = s["weighthg"]
            for j, slot in enumerate(["0", "1", "H", "S"]):
                a = s.get("abilities", {}).get(slot)
                abil[i, j] = self.aid.get(toid(a), 0) if a else 0
            for j, e in enumerate(self.megas.get(sid, [])[:2]):
                mega[i, j] = [self.iid.get(e["item"], 0), self.sid.get(e["mega"], 0)]
            f = 0
            if s.get("isMega"):
                f |= 1 << 0
            if s.get("battleOnly"):
                f |= 1 << 1
            if s.get("isCosmeticForme"):
                f |= 1 << 2
            if s.get("requiredItem") or s.get("requiredItems"):
                f |= 1 << 3
            if s.get("nfe"):
                f |= 1 << 4
            flags[i] = f
            base_species[i] = self.sid.get(toid(s.get("baseSpecies", sid)), i)
            num[i] = s.get("num", 0)
        self.arr("species_base_stats", base)
        self.arr("species_types", types)
        self.arr("species_weight_hg", weight)
        self.arr("species_abilities", abil)
        self.arr("species_mega", mega)
        self.arr("species_flags", flags)
        self.arr("species_base_species", base_species)
        self.arr("species_num", num)

    def boosts7(self, d) -> list[int]:
        d = d or {}
        return [int(d.get(k, 0)) for k in BOOSTS7]

    @staticmethod
    def boost_order(keys) -> list[int]:
        """BOOSTS7 indices in PS key order (battle.boost iterates `for (k in boost)` and logs one -boost/-unboost per
        key in that order; export_dex.js boost_key_order), then the remaining stats in canonical order."""
        ks = [BOOSTS7.index(k) for k in keys]
        return ks + [j for j in range(7) if j not in ks]

    def build_moves(self) -> None:
        n = len(self.move_ids)
        z16 = lambda *shape: np.zeros((n, *shape) if shape else n, np.int16)  # noqa: E731
        bp, mtype, cat, acc, prio, pp, crit, tgt = (z16() for _ in range(8))
        multihit, drain, recoil, heal = z16(2), z16(2), z16(2), z16(2)
        ohko, self_switch, force_switch, status, vol, side, slot, pw, weather, terrain = (z16() for _ in range(10))
        boosts, self_boosts, self_boost2, boost_order = z16(7), z16(7), z16(7), z16(7)
        self_vol, self_chance, selfdestruct, fixed_damage = z16(), z16(), z16(), z16()
        sec_chance, sec_status, sec_vol = z16(2), z16(2), z16(2)
        sec_boosts, sec_self_boosts = z16(2, 7), z16(2, 7)
        sec_flags = np.zeros((n, 2), np.uint32)
        nsec = z16()
        flags = np.zeros((n, 2), np.uint32)  # 2 words: no uint64 without x64 (03 §4.3)
        misc = {k: z16() for k in ["override_off_stat", "override_def_stat", "override_off_pokemon", "ignore_defensive",
                                   "ignore_evasion", "ignore_immunity", "ignore_ability", "tracks_target", "will_crit",
                                   "breaks_protect", "sleep_usable", "smart_target", "multiaccuracy", "struggle_recoil",
                                   "has_crash_damage", "mind_blown_recoil", "thaws_target", "has_sheer_force",
                                   "stalling_move", "calls_move", "no_pp_boosts", "non_ghost_target",
                                   "ignore_positive_defensive", "ignore_negative_offensive", "ignore_offensive",
                                   "spread_hit", "force_stab", "has_fn"]}
        stat5 = {"atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}
        for i, mid in enumerate(self.move_ids[1:], 1):
            m = self.moves[mid]
            bp[i] = m.get("basePower", 0)
            mtype[i] = self.tid[m["type"]]
            cat[i] = CATEGORIES.index(m["category"])
            acc[i] = 0 if m["accuracy"] is True else m["accuracy"]
            prio[i] = m.get("priority", 0)
            pp[i] = m["champions_maxpp"]
            crit[i] = m.get("critRatio", 1) or 0
            tgt[i] = TARGETS.index(m["target"])
            mh = m.get("multihit")
            if isinstance(mh, list):
                multihit[i] = mh
            elif isinstance(mh, int):
                multihit[i] = [mh, mh]
            for arr_, key in ((drain, "drain"), (recoil, "recoil"), (heal, "heal")):
                if m.get(key):
                    arr_[i] = m[key]
            ohko[i] = 0 if not m.get("ohko") else (2 if m["ohko"] == "Ice" else 1)
            self_switch[i] = SELF_SWITCH[m.get("selfSwitch")]
            force_switch[i] = 1 if m.get("forceSwitch") else 0
            status[i] = self.cond(m.get("status"))
            vol[i] = self.cond(m.get("volatileStatus"))
            side[i] = self.cond(m.get("sideCondition"))
            slot[i] = self.cond(m.get("slotCondition"))
            pw[i] = self.cond(m.get("pseudoWeather"))
            weather[i] = self.cond(m.get("weather"))
            terrain[i] = self.cond(m.get("terrain"))
            boosts[i] = self.boosts7(m.get("boosts"))
            ko = m["boost_key_order"]          # PS key orders (data/export is key-sorted)
            boost_order[i] = self.boost_order(ko["boosts"])
            selfd = m.get("self") or {}
            # only primary boosts carry an order table: every other boosts dict must be in canonical order
            for other in [ko["self"], ko["selfBoost"]] + [x for sec in ko["secondaries"] for x in (sec["boosts"], sec["self"])]:
                ks = [BOOSTS7.index(k) for k in other]
                assert ks == sorted(ks), (mid, other)
            self_boosts[i] = self.boosts7(selfd.get("boosts"))
            self_boost2[i] = self.boosts7((m.get("selfBoost") or {}).get("boosts"))  # PS move.selfBoost
            self_vol[i] = self.cond(selfd.get("volatileStatus"))
            self_chance[i] = selfd.get("chance", -1) if selfd else -1
            selfdestruct[i] = {None: 0, "always": 1, "ifHit": 2}[m.get("selfdestruct")]
            dmg = m.get("damage")
            fixed_damage[i] = -1 if dmg == "level" else (dmg or 0)
            secs = m.get("secondaries") or []
            nsec[i] = len(secs)
            assert len(secs) <= 2, mid
            for j, sec in enumerate(secs):
                sec_chance[i, j] = sec.get("chance", -1)
                sec_status[i, j] = self.cond(sec.get("status"))
                sec_vol[i, j] = self.cond(sec.get("volatileStatus"))
                sec_boosts[i, j] = self.boosts7(sec.get("boosts"))
                sec_self_boosts[i, j] = self.boosts7((sec.get("self") or {}).get("boosts"))
                sec_flags[i, j] = (1 if sec.get("dustproof") else 0) | (2 if f"secondaries[{j}].onHit" in m["fnKeys"] else 0)
            for k, v in (m.get("flags") or {}).items():
                if v and k in MOVE_FLAGS:
                    j = MOVE_FLAGS.index(k)
                    flags[i, j // 32] |= np.uint32(1 << (j % 32))
            ooff = m.get("overrideOffensiveStat")
            misc["override_off_stat"][i] = stat5.get(ooff, 0)
            misc["override_def_stat"][i] = stat5.get(m.get("overrideDefensiveStat"), 0)
            misc["override_off_pokemon"][i] = 1 if m.get("overrideOffensivePokemon") == "target" else 0
            ii = m.get("ignoreImmunity")
            misc["ignore_immunity"][i] = 0 if not ii else (1 if ii is True else 2)  # 2 = per-type object (bespoke)
            for key, col in [("ignoreDefensive", "ignore_defensive"), ("ignoreEvasion", "ignore_evasion"),
                             ("ignoreAbility", "ignore_ability"), ("tracksTarget", "tracks_target"),
                             ("willCrit", "will_crit"), ("breaksProtect", "breaks_protect"),
                             ("sleepUsable", "sleep_usable"), ("smartTarget", "smart_target"),
                             ("multiaccuracy", "multiaccuracy"), ("struggleRecoil", "struggle_recoil"),
                             ("hasCrashDamage", "has_crash_damage"), ("mindBlownRecoil", "mind_blown_recoil"),
                             ("thawsTarget", "thaws_target"), ("hasSheerForce", "has_sheer_force"),
                             ("stallingMove", "stalling_move"), ("callsMove", "calls_move"),
                             ("noPPBoosts", "no_pp_boosts"), ("nonGhostTarget", "non_ghost_target"),
                             ("ignorePositiveDefensive", "ignore_positive_defensive"),
                             ("ignoreNegativeOffensive", "ignore_negative_offensive"),
                             ("ignoreOffensive", "ignore_offensive"), ("spreadHit", "spread_hit"),
                             ("forceSTAB", "force_stab")]:
                misc[col][i] = 1 if m.get(key) else 0
            misc["has_fn"][i] = 1 if [k for k in m["fnKeys"] if not k.startswith("condition.")] else 0
        for name, a in [("bp", bp), ("type", mtype), ("category", cat), ("accuracy", acc), ("priority", prio),
                        ("pp", pp), ("crit_ratio", crit), ("target", tgt), ("multihit", multihit), ("drain", drain),
                        ("recoil", recoil), ("heal", heal), ("ohko", ohko), ("self_switch", self_switch),
                        ("force_switch", force_switch), ("status", status), ("volatile", vol), ("side_cond", side),
                        ("slot_cond", slot), ("pseudo_weather", pw), ("weather", weather), ("terrain", terrain),
                        ("boosts", boosts), ("boost_order", boost_order), ("self_boosts", self_boosts), ("selfboost", self_boost2), ("self_volatile", self_vol),
                        ("self_chance", self_chance), ("selfdestruct", selfdestruct), ("fixed_damage", fixed_damage),
                        ("nsec", nsec), ("sec_chance", sec_chance), ("sec_status", sec_status),
                        ("sec_volatile", sec_vol), ("sec_boosts", sec_boosts), ("sec_self_boosts", sec_self_boosts),
                        ("sec_flags", sec_flags), ("flags", flags)]:
            self.arr(f"move_{name}", a)
        for k, a in misc.items():
            self.arr(f"move_{k}", a)

    def build_abilities_items(self) -> None:
        n = len(self.ability_ids)
        af = np.zeros(n, np.uint32)
        for i, aid in enumerate(self.ability_ids[1:], 1):
            for k, v in (self.abilities[aid].get("flags") or {}).items():
                if v and k in ABILITY_FLAGS:
                    af[i] |= np.uint32(1 << ABILITY_FLAGS.index(k))
        self.arr("ability_flags", af)
        n = len(self.item_ids)
        itf = np.zeros(n, np.uint32)
        fling = np.zeros(n, np.int16)
        ng = np.zeros((n, 2), np.int16)
        for i, iid in enumerate(self.item_ids[1:], 1):
            it = self.items[iid]
            f = 0
            if it.get("isBerry"):
                f |= 1 << 0
            if it.get("megaStone"):
                f |= 1 << 1
            if it.get("ignoreKlutz"):
                f |= 1 << 2
            if it.get("isGem"):
                f |= 1 << 3
            if it.get("isChoice"):
                f |= 1 << 4
            if it.get("fling"):
                f |= 1 << 5
                fling[i] = (it["fling"] or {}).get("basePower", 0)
            itf[i] = f
            g = it.get("naturalGift")
            if g:
                ng[i] = [g.get("basePower", 0), self.tid.get(g.get("type"), 0)]
        self.arr("item_flags", itf)
        self.arr("item_fling_bp", fling)
        self.arr("item_natural_gift", ng)

    def build_types_natures(self) -> None:
        dt = np.zeros((NTYPES, NTYPES), np.int8)  # [defending type, attacking type] = PS damageTaken code
        imm = np.zeros((NTYPES, len(IMMUNITY_KEYS)), np.int8)
        tc = self.typechart["types"]
        for t, i in self.tid.items():
            if t == "???":
                continue
            row = tc[toid(t)]["damageTaken"]
            for a, j in self.tid.items():
                if a in row:
                    dt[i, j] = row[a]
            for k, key in enumerate(IMMUNITY_KEYS):
                imm[i, k] = row.get(key, 0)
        self.arr("type_damage_taken", dt)
        self.arr("type_immunity", imm)
        plus = np.full(len(self.nature_ids), -1, np.int8)
        minus = np.full(len(self.nature_ids), -1, np.int8)
        for i, nid in enumerate(self.nature_ids):
            n = self.natures[nid]
            if n["plus"]:
                plus[i] = STATS6.index(n["plus"])
                minus[i] = STATS6.index(n["minus"])
        self.arr("nature_plus", plus)
        self.arr("nature_minus", minus)

    def build_conditions(self) -> None:
        n = len(self.cond_ids)
        kind = np.zeros(n, np.int8)
        dur = np.zeros(n, np.int16)
        flags = np.zeros(n, np.uint32)
        for i, c in enumerate(self.cond_ids[1:], 1):
            kind[i] = COND_KINDS.index(self.cond_kind.get(c, "STATUS"))
            e = self.conds.get(c, {})
            dur[i] = e.get("duration") or 0
            f = 0
            if "durationCallback" in e.get("fnKeys", []):
                f |= 1 << 0
            if e.get("noCopy"):
                f |= 1 << 1
            if e.get("affectsFainted"):
                f |= 1 << 2
            flags[i] = f
        self.arr("cond_kind", kind)
        self.arr("cond_duration", dur)
        self.arr("cond_flags", flags)

    def build_supported(self) -> None:
        for st in STAGES:
            s = self.stages[st]
            for name, ids, members in [("species", self.species_ids, s["species"]), ("move", self.move_ids, s["moves"]),
                                       ("ability", self.ability_ids, s["abilities"]),
                                       ("item", self.item_ids, s["items"]),
                                       ("cond", self.cond_ids, s["conditions"])]:
                mem = set(members)
                sup = np.array([i > 0 and x in mem for i, x in enumerate(ids)], np.bool_)
                if name == "move":
                    sup[self.mid["struggle"]] = True
                if name == "cond":
                    for c in STATUSES:
                        sup[self.cid[c]] = True
                if name == "species" and st != "core":
                    pass
                self.arr(f"supported_{st}_{name}", sup)

    def build_hooks(self) -> None:
        """HookIds (one per PS event), handler prefix masks and resolved (order, priority, subOrder) per
        (effect, hook, prefix) from data/export/handler_order.json (PS resolvePriority)."""
        import re
        ho = load(self.export / "handler_order.json")
        rx = re.compile(r"^on(" + "|".join(p for p in PREFIXES if p) + r")?([A-Z]\w*)$")
        events = set(ENGINE_EVENTS)
        entries = []  # (effect id, event, prefix index, order, priority, subOrder)
        spaces = [("move", self.move_ids, self.move_base), ("ability", self.ability_ids, self.ability_base),
                  ("item", self.item_ids, self.item_base), ("condition", self.cond_ids, self.cond_base)]
        holder_kind = {"STATUS": "pokemon", "VOL": "pokemon", "SLOT": "slot", "SIDE": "side", "PW": "field",
                       "WEATHER": "field", "TERRAIN": "field"}
        for kind, idl, base in spaces:
            table = ho.get(kind, {})
            for i, eid in enumerate(idl[1:], 1):
                cbs = table.get(eid, {})
                for cb, meta in cbs.items():
                    m = rx.match(cb)
                    if not m:
                        continue
                    pfx = PREFIXES.index(m.group(1) or "")
                    ev = m.group(2)
                    events.add(ev)
                    sub = meta["subOrder"]
                    if isinstance(sub, dict):
                        sub = sub[holder_kind.get(self.cond_kind.get(eid, "VOL"), "pokemon")]
                    order = ORDER_LAST if meta["order"] is None else meta["order"]
                    entries.append((base + i - 1, ev, pfx, order, meta["priority"], sub))
        self.events = sorted(events)
        nh = len(self.events) + 1
        hid = {e: i + 1 for i, e in enumerate(self.events)}
        mask = np.zeros((self.neffect, nh), np.uint8)
        order = np.full((self.neffect, nh, len(PREFIXES), 3), 0, np.int16)
        order[..., 0] = ORDER_LAST
        for eff, ev, pfx, o, pr, sub in entries:
            h = hid[ev]
            mask[eff, h] |= np.uint8(1 << pfx)
            order[eff, h, pfx] = [o, pr, sub]
        self.arr("handler_mask", mask)
        self.arr("handler_order", order)

    def build_preview(self) -> None:
        orders = np.array(list(itertools.permutations(range(6), 4)), np.int8)
        assert orders.shape == (360, 4)
        self.arr("preview_orders", orders)

    # ------------------------------------------------------------------ outputs
    def data_version(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.export.glob("*.json")) + sorted((self.export / "stages").glob("*.json")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
        h.update(pathlib.Path(__file__).read_bytes())
        return h.hexdigest()[:16]

    def write(self, out: pathlib.Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        dv = self.data_version()
        self.arrays["data_version"] = np.frombuffer(dv.encode(), np.uint8).copy()
        buf = io.BytesIO()
        # deterministic npz: fixed member order and timestamps
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name in sorted(self.arrays):
                b = io.BytesIO()
                np.lib.format.write_array(b, self.arrays[name], allow_pickle=False)
                zi = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, b.getvalue())
        (out / "tables.npz").write_bytes(buf.getvalue())
        (out / "ids.py").write_text(self.ids_py(dv))
        init = out / "__init__.py"
        if not init.exists():
            init.write_text('"""GENERATED by tools/datagen/codegen.py — never hand-edit (ids.py, tables.npz)."""\n')

    def ids_py(self, dv: str) -> str:
        L = ['"""GENERATED by tools/datagen/codegen.py from data/export — never hand-edit.',
             "", "ID spaces (03 §1) are stage-independent; stage support lives in tables.npz `supported_<stage>_*`.",
             '"""', "# ruff: noqa", "", f'DATA_VERSION = "{dv}"', f'PS_COMMIT = "{self.meta["ps_commit"]}"',
             f'FORMAT = "{self.meta["format"]}"', ""]

        def names(title: str, ids: list[str]) -> None:
            L.append(f"{title} = (")
            for i in range(0, len(ids), 8):
                L.append("    " + ", ".join(f'"{x}"' for x in ids[i:i + 8]) + ",")
            L.append(")")

        def enum(prefix: str, ids: list[str], start: int = 0) -> None:
            for i, x in enumerate(ids):
                L.append(f"{prefix}{x.upper()} = {i + start}")

        names("SPECIES", self.species_ids)
        names("MOVES", self.move_ids)
        names("ABILITIES", self.ability_ids)
        names("ITEMS", self.item_ids)
        names("CONDITIONS", self.cond_ids)
        names("NATURES", self.nature_ids)
        for prefix, idl in (("SPECIES_", self.species_ids), ("MOVE_", self.move_ids), ("ABILITY_", self.ability_ids),
                            ("ITEM_", self.item_ids), ("NATURE_", self.nature_ids)):
            enum(prefix, idl)
        L += [f"NS, NM, NA, NI, NC, NNAT = {len(self.species_ids)}, {len(self.move_ids)}, {len(self.ability_ids)}, "
              f"{len(self.item_ids)}, {len(self.cond_ids)}, {len(self.nature_ids)}", ""]
        L.append("# TypeId: 0 none, 1..18, 19 = ??? (typeless)")
        enum("TYPE_", ["none"] + [t.upper() for t in TYPES] + ["TYPELESS"])
        L.append(f"NTYPES = {NTYPES}")
        L.append(f"IMMUNITY_KEYS = {tuple(IMMUNITY_KEYS)!r}")
        L.append("")
        L.append("# ConditionId kind sub-ranges [start, end)")
        for k, (a, b) in self.cond_ranges.items():
            L.append(f"{k}_BASE, {k}_END = {a}, {b}")
        L.append(f"COND_KINDS = {tuple(COND_KINDS)!r}")
        enum("COND_", self.cond_ids)
        L.append("")
        L.append("# unified EffectId ranges (03 §1)")
        L += [f"MOVE_BASE = {self.move_base}", f"ABILITY_BASE = {self.ability_base}", f"ITEM_BASE = {self.item_base}",
              f"COND_BASE = {self.cond_base}", f"ACTION_BASE = {self.action_base}", f"NEFFECT = {self.neffect}",
              f"ACTION_KINDS = {tuple(ACTION_KINDS)!r}", ""]
        L.append("# move table enums")
        enum("CAT_", [c.upper() for c in CATEGORIES])
        enum("TARGET_", [t.upper() for t in TARGETS])
        L.append(f"TARGETS = {tuple(TARGETS)!r}")
        L.append("# move flags: (word, bit) into move_flags[:, 2]")
        for j, f in enumerate(MOVE_FLAGS):
            L.append(f"MF_{f.upper()} = ({j // 32}, {j % 32})")
        for j, f in enumerate(ABILITY_FLAGS):
            L.append(f"AF_{f.upper()} = {1 << j}")
        for j, f in enumerate(SPECIES_FLAGS):
            L.append(f"SF_{f.upper()} = {1 << j}")
        for j, f in enumerate(ITEM_FLAGS):
            L.append(f"IF_{f.upper()} = {1 << j}")
        L.append(f"STATS6 = {tuple(STATS6)!r}")
        L.append(f"BOOSTS7 = {tuple(BOOSTS7)!r}")
        L.append("")
        L.append("# HookIds: one per PS event name (0 = none); prefixes index handler_mask bits")
        for i, e in enumerate(self.events):
            L.append(f"HOOK_{e.upper()} = {i + 1}")
        L.append(f"HOOK_NAMES = {tuple(['none'] + self.events)!r}")
        L.append(f"NH = {len(self.events) + 1}")
        L.append(f"PREFIXES = {tuple(PREFIXES)!r}")
        for j, pfx in enumerate(PREFIXES):
            L.append(f"PFX_{(pfx or 'ON').upper()} = {j}")
        L.append(f"ORDER_LAST = {ORDER_LAST}")
        L.append("")
        L.append("# DrawKind (contract/drawkey.md §5 forced kinds, then §4 tie kinds) and step tags (§3.1)")
        enum("DK_", DRAW_KINDS)
        L.append(f"DRAW_KINDS = {tuple(DRAW_KINDS)!r}")
        enum("STEP_", STEP_TAGS)
        L.append("")
        return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(ROOT / "data" / "export"))
    ap.add_argument("--out", default=str(ROOT / "vgcjax" / "data"))
    a = ap.parse_args()
    b = Builder(pathlib.Path(a.export))
    b.build_ids()
    b.build_species()
    b.build_moves()
    b.build_abilities_items()
    b.build_types_natures()
    b.build_conditions()
    b.build_supported()
    b.build_hooks()
    b.build_preview()
    b.write(pathlib.Path(a.out))
    print(f"codegen: {len(b.arrays)} tables, NS={len(b.species_ids)} NM={len(b.move_ids)} NA={len(b.ability_ids)} "
          f"NI={len(b.item_ids)} NC={len(b.cond_ids)} NEFFECT={b.neffect}")


if __name__ == "__main__":
    main()
