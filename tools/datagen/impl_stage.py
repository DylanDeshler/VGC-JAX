"""The `impl` stage: what the engine implements so far (M3+ incremental conformance; 11 §3 targeted fuzz).

impl = core (05 §1.1) + every v0 entity whose mechanics the engine supports, so that conformance corpora drawn from
it (oracle.js gen --teams stage:impl, --require <entity>) only exercise implemented code. Moves join through
SUPPORTED_REFS: a DATA_ONLY move qualifies when its census data_refs only name supported conditions and it has no
core_refs; other entities are listed by id in EXTRA as their handlers land. Output: data/export/stages/impl.json.

    python -I tools/datagen/impl_stage.py
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPORT = ROOT / "data" / "export"

# census data_refs kind -> condition ids the engine implements (vgcjax/effects)
SUPPORTED_REFS = {
    "status": {"brn", "par", "psn", "tox", "slp", "frz"},
    "volatile": {"flinch", "confusion"},
    "weather": {"sunnyday", "raindance", "sandstorm", "snowscape"},
    "terrain": {"electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"},
}
# entities implemented beyond the data-only rule, by kind (filled in as handlers land)
EXTRA: dict[str, set[str]] = {
    "moves": {"protect", "detect", "wideguard", "fakeout", "feint",     # effects/protect.py (Quick Guard: no learner yet)
              "tailwind", "trickroom", "reflect", "lightscreen",         # effects/field.py
              "sunnyday", "raindance", "sandstorm", "snowscape",         # effects/weather.py
              "weatherball", "thunder", "hurricane", "blizzard",
              "electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"},   # effects/terrain.py
    "abilities": {"drought", "drizzle", "sandstream", "snowwarning", "electricsurge", "grassysurge", "psychicsurge",
                  "mistysurge", "cloudnine", "intimidate", "pressure", "unnerve", "frisk",    # effects/abilities.py
                  "technician", "toughclaws", "ironfist", "sharpness", "megalauncher", "strongjaw", "reckless",
                  "punkrock", "sandforce", "hugepower", "purepower", "hustle", "guts", "blaze", "torrent",
                  "overgrow", "swarm", "thickfat", "solarpower", "multiscale", "filter", "solidrock", "friendguard",
                  "sniper", "adaptability", "compoundeyes",                                   # effects/ability_mods.py
                  "limber", "insomnia", "vitalspirit", "immunity", "waterveil", "waterbubble", "magmaarmor",
                  "leafguard", "purifyingsalt", "innerfocus", "owntempo", "oblivious", "clearbody", "whitesmoke",
                  "hypercutter", "keeneye", "bigpecks", "scrappy", "guarddog", "sweetveil", "flowerveil",
                  "voltabsorb", "waterabsorb", "eartheater", "sapsipper", "motordrive", "flashfire", "soundproof",
                  "bulletproof", "telepathy", "goodasgold", "static", "flamebody", "poisonpoint", "roughskin",
                  "stamina", "weakarmor", "gooey", "effectspore", "poisontouch", "spicyspray", "justified", "rattled",
                  "defiant", "competitive", "speedboost", "regenerator", "naturalcure", "angerpoint", "steadfast"},
                                                                                              # effects/ability_blocks.py
    "items": {"blackbelt", "blackglasses", "charcoal", "dragonfang", "fairyfeather", "hardstone", "magnet", "metalcoat",
              "miracleseed", "mysticwater", "nevermeltice", "poisonbarb", "sharpbeak", "silkscarf", "silverpowder",
              "softsand", "spelltag", "twistedspoon", "lifeorb", "expertbelt", "leftovers", "rockyhelmet",
              "focussash", "lightclay", "heatrock", "damprock", "smoothrock", "icyrock", "terrainextender",
              "sitrusberry", "babiriberry", "chartiberry", "chopleberry", "cobaberry", "colburberry", "habanberry",
              "kasibberry", "kebiaberry", "occaberry", "passhoberry", "payapaberry", "rindoberry", "roseliberry",
              "shucaberry", "tangaberry", "wacanberry", "yacheberry"},                         # + effects/berries.py
                                                                                              # effects/items.py
    "conditions": {"protect", "stall", "wideguard", "quickguard", "tailwind", "trickroom", "reflect", "lightscreen",
                   "sunnyday", "raindance", "sandstorm", "snowscape",
                   "electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"},
}


def supported_move(rec: dict) -> bool:
    if rec.get("class") != "DATA_ONLY" or rec.get("core_refs"):
        return False
    for kind, conds in (rec.get("data_refs") or {}).items():
        if kind not in SUPPORTED_REFS or not set(conds) <= SUPPORTED_REFS[kind]:
            return False
    return True


def main() -> None:
    core = json.loads((EXPORT / "stages" / "core.json").read_text())
    v0 = json.loads((EXPORT / "stages" / "v0.json").read_text())
    census = json.loads((EXPORT / "handler_census.json").read_text())["records"]
    species_data = json.loads((EXPORT / "species.json").read_text())
    abilities = set(core["abilities"]) | EXTRA["abilities"]
    toid = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())  # noqa: E731
    species = set(core["species"]) | {s for s in v0["species"] if any(
        toid(a) in abilities for a in species_data[s]["abilities"].values())}
    moves = set(core["moves"]) | {m for m in v0["moves"] if supported_move(census["move"].get(m, {}))} | EXTRA["moves"]
    conds = set(core["conditions"]) | set().union(*SUPPORTED_REFS.values()) | EXTRA["conditions"]
    out = {
        "stage": "impl",
        "species": sorted(species),
        "moves": sorted(moves),
        "abilities": sorted(abilities),
        "items": sorted(set(core["items"]) | EXTRA["items"]),
        "conditions": sorted(conds),
        "implemented_conditions": sorted(set().union(*SUPPORTED_REFS.values()) | EXTRA["conditions"]),
        "meta": {"rule": "core + DATA_ONLY v0 moves whose data_refs are implemented (tools/datagen/impl_stage.py "
                         "SUPPORTED_REFS) + EXTRA", "supported_refs": {k: sorted(v) for k, v in SUPPORTED_REFS.items()}},
    }
    (EXPORT / "stages" / "impl.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"impl: {len(out['moves'])} moves ({len(out['moves']) - len(core['moves'])} beyond core), "
          f"{len(out['abilities'])} abilities, {len(out['items'])} items, {len(out['species'])} species")


if __name__ == "__main__":
    main()
