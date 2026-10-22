# Dex scope and handler census: JAX engine for Champions VGC 2026 Reg M-C

*Research date: 2026-10-21. Oracle: Pokémon Showdown commit `d601443`, `Dex.mod('champions')`, format `gen9championsvgc2026regmc`. Every number below was produced by the scripts in `research/dex/scripts/`. The commands to reproduce them are in §9.*

> **Note on reproducibility.** Partway through the run the machine rebooted and `/tmp` was wiped, taking `/tmp/psrepo` and `/tmp/stats_dl` with it. I re-cloned PS at `d601443` into `/tmp/psrepo/ps`, ran `npm install` and `node build`, and re-downloaded the three stats files into `research/dex/raw/stats_2026-09/` (byte sizes match the originals). The rebuilt legal pool matches the one built before the reboot exactly.

---

## 0. TL;DR

- **Mass coverage overstates what a dex supports; team coverage is the metric to use.** The dex from the background doc covers 99% of slot mass at 1760 Bo3: 92 species, 157 moves, 65 abilities, 60 items. That dex fully expresses only:
  - **70.7%** of 1760 Bo3 teams;
  - **21%** of all-ladder Bo1 teams;
  - **12.5%** of real open-team-sheet (OTS) teams scraped from replays.
- **Mega holders' base abilities are the hidden reason.**
  - **78.5%** of real Reg M-C teams carry at least 2 Mega Stones. Only one can Mega Evolve per battle, so the others play in base forme with their base ability.
  - Chaos stats never record that base ability; they report the Mega forme's ability.
  - Base abilities that matter: Charizard's Blaze is on about 20% of real teams, Floette's Flower Veil on 14%, Golisopod's Emergency Exit on 12%, and Garchomp's Rough Skin on 14%.
- **Legal pool (v2):**
  - **293 set-level species.** That is 269 if cosmetic formes are dropped, and 231 distinct base species.
  - **82 Mega formes** (81 Mega Stones) plus 7 other in-battle formes.
  - **510 moves.**
  - **215 abilities:** 202 can appear on a team sheet, and 13 exist only on Mega formes.
  - **166 items:** 81 Mega Stones and 85 others.
  - Not legal: Commander, Paradox/Booster Energy, the Ruin abilities, Covert Cloak, Clear Amulet, Safety Goggles, Mirror Herb, Assault Vest, Choice Band/Specs, Weakness Policy, Eject Pack, Spore, Tera Blast. Several of these were listed as v1 must-haves in `jax-vgc-engine-scope.md` §4; that list needs correcting.
- **Handler census, full legal pool:**
  - 483 entities need bespoke code (CUSTOM): 174 moves, 185 abilities, 35 items and 89 conditions.
  - 1,107 handler functions in total, about 7.1k LoC.
  - **747 distinct handler templates**, i.e. functions left after merging ones that differ only in literals. This is what `lax.switch` cost depends on.
  - These are spread over 135 hook names.
- **Recommended v1** ("cost-aware" selection, τ = 0.1%):
  - **Contents:** every legal species and forme; every entity that adds no handler code; and every CUSTOM entity that appears on at least 0.1% of teams at 1760 Bo3 or all-ladder Bo1. That gives **454 moves, 172 abilities and 160 items**.
  - **Handler cost:** **376 CUSTOM entities**, 83 conditions, 919 handler functions, **603 distinct templates**, 118 hook names.
  - **Team coverage:** 99.7% at 1760 Bo3, 99.1% at 1760 Bo1, 97.5% all-ladder Bo3, **95.7% all-ladder Bo1**, and **99.4% of 3,211 real Reg M-C Bo3 OTS teams**.
  - **Robustness:** it also covers 97.7% of VGC-Bench Reg M-B teams, and 99.5% of 1760 Bo3 teams from last month's Reg M-B stats.
  - **Cost:** v1 is still about **80% of the full pool's handler cost.** In this format the long tail is cheap per entity, but it is what keeps teams from being fully covered.
- **Staging:**
  - **v0** (bring-up): 220 CUSTOM / 369 templates, giving 90% of 1760 Bo3 teams.
  - **v1-lite** (1760 Bo3 only): 312 / 514, giving 99.6% at 1760 Bo3 but only 83% all-ladder.
  - **v1:** 376 / 603.
  - **v2:** 483 / 747.
- **Teams:**
  - **Real teams:** 3,211 real Reg M-C OTS teams were scraped from public Bo3 replays. With a placeholder Stat Point (SP) spread, 100% pass `TeamValidator`.
  - **VGC-Bench** publishes Champions Reg M-A and M-B teams and logs, but nothing for M-C yet. M-B teams are 100% legal under M-C rules and 97.7% fall inside v1.
  - **Not usable:** `metamon-teams` is singles-only.
  - **PS random doubles generator:** only 9% of its teams are VGC-legal, because it ignores Item Clause.
  - **Prototype usage-stats sampler:** 2,000 of 2,000 sampled teams were legal and inside v1.

---

## 1. Data and method

**Usage stats.** Smogon chaos JSON, 2026-09, `gen9championsvgc2026regmc{,bo3}-{0,1500,1630,1760}`: 1.63M Bo1 battles and 223k Bo3 battles.
- 2026-10 is not published yet (HTTP 404 on 2026-10-21).
- 2026-08 has only Reg M-B (`gen9championsvgc2026regmb*`), because Reg M-C started during September. The 2026-09 directory contains both M-B and M-C.
- I downloaded M-B 2026-08 (bo3-1760, bo1-1760, bo1-0) as a drift check (§4.4).

**Real teams.**
- **Reg M-C Bo3:** 1,927 public replays uploaded 2026-10-20 to 2026-10-21, fetched via `replay.pokemonshowdown.com/search.json`. They contain 3,850 `|showteam|` lines, which is **3,211 unique OTS teams**.
  - The sample is low-to-mid ladder: the median game rating is 1136 among the 1,149 rated teams, and only 20 games are rated ≥1500.
  - Bo3 forces OTS, so every game has full sheets: species, item, ability, moves and nature. Stat Points are hidden.
- **Reg M-C Bo1:** OTS is opt-in, so only 125 unique teams were found in 510 replays.
- **VGC-Bench Reg M-B Bo3** logs from HF: 3,843 games, 6,230 unique teams.

**Team coverage.** "Team coverage of dex D" is the fraction of teams in which every species/forme, move, ability and item is in D. I estimate it three ways:

| estimator | how | notes |
|---|---|---|
| `team_cov_mc` (headline) | 4,000 synthetic teams per format, sampled with a naive-Bayes teammate model built from chaos `Teammates` co-occurrence (P(t \| s) = Teammates_s[t] / weight_s), with species clause applied. Per mon: P(expressible) = P(ability ∈ D) · P(item ∈ D) · Π over moves m ∉ D of (1 − f_m). | Moves are treated as independent within a set. Mega holders use the Mega forme's ability (chaos limitation). |
| `team_cov_indep` | (usage-weighted mean per-mon expressibility)^6 | Ignores teammate correlation. Usually 0–3 points higher than MC. |
| real replay teams (strict) | Exact check on 3,211 real sheets. Mega holders need both the base ability and the Mega ability in D. | The ground truth, but drawn from the low-mid ladder. |

The **selection score** for an entity is P(team contains it) = 1 − Π over species s of (1 − usage_s · f_s(e)), computed per format. If an entity is left out of the dex, at most this fraction of teams is lost.

---

## 2. Usage coverage

Full curves are in Appendix T1, from `out/coverage_summary.json`. Number of entities needed to reach 90/95/99/99.9% of slot mass (total observed in brackets):

| format | species incl. Megas | moves | abilities | items (non-stone) | Mega Stones |
|---|---|---|---|---|---|
| 1760 Bo3 | 39/52/92/158 (191) | 77/100/157/242 (473) | 31/40/65/102 (167) | 14/18/28/48 (81) | 17/21/33/52 (65) |
| all Bo3 | 54/82/175/266 (302) | 93/128/224/371 (504) | 39/54/97/156 (209) | 15/20/43/66 (84) | 21/31/57/75 (81) |
| 1760 Bo1 | 58/84/167/247 (277) | 95/131/223/340 (502) | 40/55/94/144 (203) | 16/22/42/63 (84) | 22/29/53/74 (80) |
| all Bo1 | 92/145/236/304 (333) | 121/174/290/414 (510) | 54/77/128/178 (214) | 20/30/54/72 (85) | 32/48/68/79 (81) |

- Mega formes hold about **29.5%** of team slots at every cutoff, and almost every mon holds an item (0.1–0.2% "no item").
- At all-ladder Bo1, **every one of the 510 legal moves** is used by someone.

**Mass coverage is a poor proxy for team coverage.** A team has 6 species, 24 move slots, 6 (or more) abilities and 6 items, so each tail entity cuts coverage multiplicatively:

| dex | 1760 Bo3 | 1760 Bo1 | all Bo3 | all Bo1 | real Bo3 sheets (strict / lenient) |
|---|---|---|---|---|---|
| 95% mass @1760 Bo3 (50+20 Mega / 100 / 40 / 36) | 0.349 | 0.130 | 0.184 | 0.052 | 0.003 / 0.275 |
| 99% mass @1760 Bo3 (83+34 Mega / 157 / 65 / 60) | 0.707 | 0.395 | 0.464 | 0.213 | 0.125 / 0.558 |
| every entity on ≥0.1% of 1760 Bo3 teams | 0.893 | 0.655 | 0.700 | 0.457 | 0.763 |
| every entity on ≥0.01% of 1760 Bo3 teams | 0.992 | 0.845 | 0.839 | 0.691 | 0.900 |

On real sheets, the strict-vs-lenient gap for the 99%-mass dex (0.125 vs 0.558) comes **entirely from the base abilities of Mega Stone holders**. These are the most common base abilities among Mega Stone holders in the 3,211 real teams:

| Mega holder | base ability (number of teams) |
|---|---|
| Charizard (Y) | Blaze 635, Solar Power 64 |
| Raichu (Y) | Lightning Rod 558, Static 33 |
| Floette | Flower Veil 464 |
| Garchomp (Z) | Rough Skin 445, Sand Veil 84 |
| Golisopod | Emergency Exit 387 |
| Staraptor | Intimidate 289, Reckless 11 |
| Raichu (X) | Lightning Rod 271, Static 25 |
| Tyranitar | Sand Stream 232 |
| Gardevoir | Trace 84, Synchronize 59, Telepathy 34 |
| Camerupt | Solid Rock 123, Magma Armor 15 |

The design consequence: the engine must implement base-forme abilities for Mega holders, including Emergency Exit's mid-turn switch, even though the usage stats never show them. The v1 selection adds them using the frequencies observed in replays.

---

## 3. Legal pool (v2)

Enumeration: `scripts/legal_pool.js`. For each species it runs `TeamValidator.checkSpecies` and `validateSet`, using one set per ability slot (with the required item for Megas). For each legal species and each move it runs `checkCanLearn` and `checkMove`. Items go through `checkItem`; Mega Stones are kept only if their forme is reachable.

| | count | notes |
|---|---|---|
| set-level species/formes | **293** | 269 excluding cosmetic formes (Vivillon ×19, Alcremie ×8, Squawkabilly ×4, Gourgeist sizes, etc. are separate entries); 231 base species |
| Mega formes | **82** | includes the Champions Megas, e.g. Raichu-Mega-X/Y, Garchomp-Mega-Z, Absol-Mega-Z, Lucario-Mega-Z, Floette-Mega, Meowstic-M/F-Mega, Golisopod-, Glimmora-, Baxcalibur-, Scovillain-, Crabominable-Mega |
| other in-battle formes | 7 | Castform-Sunny/Rainy/Snowy, Aegislash-Blade, Mimikyu-Busted, Morpeko-Hangry, Palafin-Hero |
| moves (learnable by some legal species) | **510** | 260 DATA_ONLY, 76 DATA_PLUS_COMMON, 174 CUSTOM |
| abilities | **215** | 202 on team sheets + 13 Mega-only: Aerilate, Parental Bond, Shadow Tag, Innards Out, Unseen Fist, Fairy Aura, and the new Mega Sol, Dragonize, Aura Guard, Piercing Drill, Eelevate, Fire Mane, Spicy Spray |
| items | **166** | 81 Mega Stones + 85 others: 24 berries, type-boost items, terrain seeds, weather rocks, Choice Scarf, Focus Sash, Life Orb, Leftovers, Sitrus, White Herb, Rocky Helmet, Eject Button, Red Card, Quick Claw, etc. |
| conditions reachable | **102** | statuses, volatiles, side/slot conditions, weathers, terrains, pseudo-weathers |

**Absent from Reg M-C:**
- Tatsugiri/Dondozo (Commander);
- all Paradox Pokémon (Protosynthesis, Quark Drive, Booster Energy);
- the Ruin quartet, Koraidon/Miraidon, and all restricted and mythical Pokémon;
- Mega-specific restricted formes;
- Psychic/Misty Surge (Indeedee's Psychic Surge *is* legal);
- items: Covert Cloak, Clear Amulet, Safety Goggles, Mirror Herb, Assault Vest, Choice Band, Choice Specs, Weakness Policy, Throat Spray, Eject Pack, Loaded Dice, Eviolite, Room Service, Adrenaline Orb, Custap Berry;
- moves: Spore, Tera Blast, Ivy Cudgel, Blood Moon, Glacial Lance, Astral Barrage.

There is no Terastallization.

**Engine-level Champions differences (not per-entity).** `data/mods/champions/scripts.ts` overrides:
- `statModify` (Stat Point stats at level 50, no IVs) and `calculatePP` (all PP capped at 20);
- `pokemon.getActionSpeed` (no Trick Room underflow);
- `pokemon.formeChange` (a Mega stays Mega after fainting) and `pokemon.clearVolatile`;
- `actions.canTerastallize` (returns null), `canMegaEvo`, `modifyDamage`, `spreadMoveHit` and `hitStepMoveHitLoop`.

Entity tweaks in the mod (`out/champions_mod_diff.json`):
- **66 moves**, mostly basePower/PP/accuracy/flags. Behaviour changes to Curse, Disable, Encore, No Retreat and Salt Cure; and `onDisableMove` on Fake Out, First Impression and Belch.
- **9 abilities:** Anger Shell, Berserk, Emergency Exit, Wimp Out, Healer, Natural Cure, Regenerator, Run Away, Unseen Fist.
- **1 item:** Eject Button.
- **3 conditions:** paralysis 1/8, sleep 2–3 turns, freeze with a 3-turn cap and a 1/4 thaw chance.

---

## 4. Dex options and recommendation

### 4.1 Handler classification (used for cost)

The census (`scripts/handler_census.js`) reads every move, ability, item and condition through `Dex.mod('champions')`, which merges base gen9 data with the mod. It records:
- every function-valued key, including keys nested in `secondary`/`secondaries[]`/`self`/the entity's own `condition`;
- every non-default data field;
- conditions the entity *creates*, from data (`volatileStatus`, `sideCondition`, `slotCondition`, `pseudoWeather`, `weather`, `terrain`, `status`) and from code (`addVolatile('x')`, `addSideCondition`, `setWeather`, `setTerrain`, `addPseudoWeather`, `setStatus`, `addSlotCondition`);
- inspection and removal references (Brick Break → reflect, and so on);
- quoted-id mentions of other entities (interactions);
- quoted-id hits in core files (`sim/battle*.ts`, `pokemon.ts`, `side.ts`, `field.ts`, `battle-queue.ts`, `dex-*.ts`, and the mod's `scripts.ts`).

Each function is normalised by stripping comments and replacing string/number literals with placeholders, then hashed with its hook name. The result is a **template**. Two functions with the same template can share one `lax.switch` branch, with a per-entity parameter table.

- **DATA_ONLY:** no function anywhere. It references only core statuses or volatiles (brn/par/psn/tox/slp/frz/confusion/flinch). Examples: Flamethrower, Heat Wave, Rock Slide, Icy Wind, U-turn (`selfSwitch`), Roar (`forceSwitch`), Will-O-Wisp, the weather rocks.
- **DATA_PLUS_COMMON:** every function is a template shared by at least 3 legal entities, and/or the entity just sets a named condition whose handlers are counted once under "condition". Examples:
  - Protect, Detect, Spiky Shield and others share the `onPrepareHit` stall check and `onHit` add-stall wrapper; the per-move `protect`-like conditions are counted separately.
  - Tailwind, Trick Room, Reflect.
  - The 81 Mega Stones' `onTakeItem`, the 18 type-resist berries, the type-boost items, the Surges, the -ate abilities, Blaze/Torrent/Overgrow/Swarm, and Static/Flame Body/Poison Point.
- **CUSTOM:** at least one function shared by fewer than 3 legal entities.

| kind | legal: n / DATA_ONLY / DATA_PLUS_COMMON / CUSTOM | v1: n / DATA_ONLY / DATA_PLUS_COMMON / CUSTOM |
|---|---|---|
| move | 510 / 260 / 76 / 174 | 454 / 260 / 65 / 129 |
| ability | 215 / 6 / 24 / 185 | 172 / 6 / 23 / 143 |
| item | 166 / 7 / 124 / 35 | 160 / 7 / 124 / 29 |
| condition | 102 / 2 / 11 / 89 | 83 / 1 / 7 / 75 |
| **total CUSTOM** | **483** | **376** |

Abilities are almost all CUSTOM: 86% of legal abilities have bespoke code. Items are mostly parametric. Half of all moves need no handler at all.

### 4.2 Candidate dexes

Full table in Appendix T2. "Cost-aware" selection works in three steps:
1. Include every species and forme (they are pure data), plus every move/ability/item whose handler templates are already needed by engine-core conditions ("zero cost").
2. Add every entity with P(team contains) ≥ τ in 1760 Bo3 **or** all-ladder Bo1. The τ test also counts the replay-observed base abilities of Mega holders.
3. Add any remaining entity whose templates are all already paid for ("free riders").

| dex | moves / abilities / items | CUSTOM ents | conditions | templates | handler fns | 1760 Bo3 | 1760 Bo1 | all Bo3 | all Bo1 | real Bo3 sheets |
|---|---|---|---|---|---|---|---|---|---|---|
| 99% mass @1760 Bo3 (background doc) | 157 / 65 / 60 | 154 | 51 | 292 | 402 | 0.707 | 0.395 | 0.464 | 0.213 | 0.125 |
| **v0** cost-aware τ=1% | 377 / 110 / 142 | 220 | 57 | 369 | 622 | 0.899 | 0.778 | 0.787 | 0.620 | 0.820 |
| cost-aware τ=0.3% | 423 / 145 / 151 | 311 | 75 | 519 | 806 | 0.978 | 0.946 | 0.924 | 0.866 | 0.957 |
| **v1-lite** cost-aware, 1760 Bo3 only, τ=0.01% | 424 / 150 / 149 | 312 | 69 | 514 | 800 | 0.996 | 0.925 | 0.916 | 0.834 | 0.955 |
| **v1** cost-aware τ=0.1% | **454 / 172 / 160** | **376** | **83** | **603** | **919** | **0.997** | **0.991** | **0.975** | **0.957** | **0.994** |
| cost-aware τ=0.03% | 481 / 192 / 161 | 423 | 90 | 659 | 996 | 1.000 | 0.998 | 0.992 | 0.985 | 0.999 |
| **v2** full legal pool | 510 / 215 / 166 | 483 | 102 | 747 | 1107 | 1 | 1 | 1 | 1 | 1 |

All cost-aware rows include all 293 set-level species, the 82 Mega formes and the 7 other battle formes.

### 4.3 Recommendation

- **v1 = cost-aware τ = 0.1%.** Lists are in `research/dex/v1_{species,moves,abilities,items}.json`. Each entry carries P(team) and slot mass at 1760 Bo3, all Bo3, 1760 Bo1 and all Bo1, the replay P(team), its class, its hooks, and why it was included (`usage` / `zero_cost` / `free_rider_shared_template`). Each file also lists the excluded legal entities, ranked by usage.
  - It is the smallest dex in the sweep meeting all of: ≥99% team coverage at 1760 Bo3, ≥95% at all-ladder Bo1, and ≥95% of real sheets.
  - **Excluded:** 56 moves, 43 abilities and 6 items, each on fewer than 0.1% of teams. The largest are Grav Apple, Growth, Fell Stinger, Role Play, Court Change, Beak Blast, Safeguard, Thief, Reversal; Guard Dog, Rivalry, Supersweet Syrup, Cheek Pouch, Fluffy, Gluttony, Symbiosis, Plus/Minus, Guts; Focus Band, Persim, Shed Shell, Chilan, Oran, Pecha.
- **Staging (suggested milestones):**
  1. **v0** (220 CUSTOM, 369 templates). Covers 90% of 1760 Bo3 teams and 82% of real sheets. Enough to start PPO on a sampled team pool.
  2. **v1** (376 / 603). Adds about 156 CUSTOM entities, mostly one-to-three-handler abilities and moves.
  3. **v2** (483 / 747). Adds about 107 more CUSTOM entities, each on fewer than 0.1% of teams.
- **For a policy aimed only at high-ladder Bo3**, v1-lite (312 / 514, 99.6% at 1760 Bo3) is the cheaper alternative. Its all-ladder coverage drops to 83%.
- **Training needs differ from coverage needs.** The team sampler can be limited to any dex. Team coverage only matters for (a) playing arbitrary ladder opponents and (b) evaluating against real teams.

### 4.4 Drift check (does v1 survive a new regulation month?)

`out/trend_check.json` applies the v1 chosen on M-C 2026-09 to other data:

| applied to | team coverage (MC) |
|---|---|
| Reg M-B 2026-08, 1760 Bo3 | 0.995 |
| Reg M-B 2026-08, 1760 Bo1 | 0.984 |
| Reg M-B 2026-08, all Bo1 | 0.930 |
| VGC-Bench Reg M-B Bo3 teams (4,000 real sheets) | 0.977 |
| M-C Bo1 OTS replay teams (125 real sheets) | 1.000 |

Every species in the M-B stats is in v1. The top-30 species of M-B 2026-08 (1760 Bo3) and M-C 2026-09 (1760 Bo3) overlap in 22 of 30. **Takeaway:** the species meta shifts between regulations, but the entity dex barely moves, because v1 already contains every legal species and the zero-cost tail.

---

## 5. Handler census: hooks

Machine-readable outputs: `research/dex/handler_census_v1.json` and `handler_census_legal.json`. Each contains per-entity records (functions with path, hook, LoC, template id and template sharing count; data fields; created conditions; inspected conditions; mentions; core refs), per-kind hook tables with entity lists, a merged hook table, data-field frequencies, and the classification rules.

**Shape of the dispatch problem in v1** (counts include each entity's own `condition` handlers once):
- **Size:** 919 handler functions in 603 distinct templates, over **118 hook names**. That drops to **93 events** after stripping the `Foe/Ally/Source/Any` listener prefixes.
- **Branches per hook:** 46 hooks have a single template, 47 have 2–5, 14 have 6–15, and **11 have 16 or more**.
- **Largest branch counts (templates per hook):** onStart 56 (81 entities: 54 conditions, 19 abilities, 8 items), onHit 50, onTryHit 36, onBasePower 26 (52 entities), onModifyMove 22, basePowerCallback 20, onResidual 20, onDamagingHit 20, onTry 18, onEnd 17, onBeforeMove 16 (all conditions), onUpdate 12.
- **Hooks where shared templates collapse many entities into few branches:**
  - onTakeItem: 82 entities → 3 templates (Mega Stones);
  - onSourceModifyDamage: 24 → 7 (resist berries);
  - onEat: 24 → 5;
  - durationCallback: 16 → 6;
  - onFieldEnd: 10 → 1;
  - onImmunity: 9 → 3 (weather immunities);
  - onTryBoost: 10 → 5 (Intimidate blockers).
- **Data-only fields used by v1 moves:** basePower 290, secondary 176, `self` 64, boosts 45, volatileStatus 39, priority 38, multihit 14, sideCondition 10, recoil 9, selfSwitch 8, drain 8, selfBoost 7, stallingMove 6, selfdestruct 6, weather 5, heal 5, forceSwitch 4, terrain 4, ohko 4, plus long-tail flags (hasCrashDamage, willCrit, slotCondition, thawsTarget, callsMove, breaksProtect, pseudoWeather, damage, multiaccuracy, sleepUsable, override*Stat, smartTarget, tracksTarget, mindBlownRecoil).
- **Item data fields:** fling, itemUser, megaStone, isBerry, naturalGift, boosts, isChoice, isGem.
- **Ability data:** mostly event priorities, plus `flags` (breakable, etc.) and `suppressWeather`.

**Design implications.**
1. **Dispatch per event, not per entity type.**
   - Build one table per event: rows = entity ids (move, ability, item and condition id spaces concatenated), columns = template id plus parameters.
   - Most events have at most 5 branches. Only about 11 events need a large `lax.switch`.
   - For those, split by entity kind, or use a two-level switch: a "has handler" mask first, then a switch over only the templates present in the batch (sorted gather). Under `vmap` every branch runs, so 56 onStart branches per call is the main cost.
2. **Turn the modifier hooks into multiplier tables:**
   - **hooks:** onBasePower, onModifyAtk/SpA/Def/SpD/Spe, onSourceModifyDamage, onModifyDamage, onModifyCritRatio, onModifyAccuracy;
   - **parameters:** (type, category, flag, condition) → 4096-based multiplier.
   - This removes about 120 functions from the switch path.
3. **Condition lifecycle hooks** follow fixed patterns:
   - `durationCallback`, `onFieldStart/End/Restart`, `onSideStart/End`, `onStart/End/Restart`, `onResidual`;
   - the conditions are weathers, terrains, screens, Tailwind, Trick Room, Gravity, Magic Room/Wonder Room, the Protect family, Encore/Taunt/Disable/Torment/Heal Block/Throat Chop, Substitute, Yawn, Perish Song, Salt Cure, Leech Seed, Wish, Future Sight, Charge, Stockpile, and so on;
   - generic code can cover them: duration tables, extension items (Light Clay, the weather rocks and Terrain Extender are all DATA_ONLY), and per-turn countdown.

Per-kind hook tables with entity names are in Appendix T5. The top-45 merged hook tables for v1 and for the legal pool are in Appendix T4.

---

## 6. Complexity ranking of v1 CUSTOM entities

**Automatic score.** In `out/complexity_v1.json` and Appendix T7:

score = LoC + 6 × (number of functions) + 4 × (core-file references) + 3 × (mentions by other entities)

- **Top of the ranking:** the weathers (sunnyday, sandstorm, raindance, snowscape), Substitute (move and condition), Gravity, sleep, lockedmove, confusion, Heal Block, partiallytrapped, freeze, futuremove, twoturnmove, the four terrains, stall, Disable, Charge, Curse, Metronome (item), Aurora Veil, Throat Chop, Encore.
- **Bias:** the score measures data-file handlers. It **under-weights** mechanics whose logic lives in core engine code. Transform, for example, is one line in the move but about 80 lines in `pokemon.transformInto`.

The hand-curated table below corrects for that. P(team) is the chaos estimate of the fraction of teams containing the entity, at 1760 Bo3 / all-ladder Bo1.

| mechanic | legal / in v1 | P(team) 1760 Bo3 / all Bo1 | where the complexity lives | engine note |
|---|---|---|---|---|
| **Mega Evolution** (system) | yes / yes | 29.5% of slots; 78.5% of real teams carry ≥2 stones | `battle-actions` runMegaEvo, Champions `canMegaEvo` and `formeChange` (a Mega persists after fainting), speed recalculation after Mega (Gen 7+) | Mid-turn change of species, type, ability and stats. Mega abilities fire onStart on Mega (Drought, Intimidate-Megas). Pre-Mega base ability needed (§2). |
| **Emergency Exit / Eject Button / Parting Shot / U-turn / Flip Turn / Volt Switch / Red Card** | yes / yes | Emergency Exit on about 12% of real sheets (Golisopod base); Parting Shot 45%; U-turn 21%; Eject Button 4.9% | queue suspension; `switchFlag`; Champions overrides for Emergency Exit, Wimp Out and Eject Button | The MIDTURN_SWITCH phase is mandatory in v0. |
| **Revival Blessing** | yes / yes | 1.3% / 2.9% | `slotCondition`, special "choose a fainted ally" request; 13 core-file references | **Needs a new decision phase** (choose a fainted party member). |
| Fake Out / First Impression | yes / yes | 83% / 70% | onTry first-turn check; Champions adds `onDisableMove` (unselectable after turn 1) | Track turns active. Action-mask parity matters. |
| Protect family, Wide Guard, Quick Guard | yes / yes | Protect 98.9%; Wide Guard 13.9%; Quick Guard 4.4% | shared stall template; per-move conditions; Champions changes PP; Unseen Fist / Piercing Drill bypass via `onHitProtect` | Consecutive-use counter (stall). |
| Follow Me / Rage Powder / Lightning Rod | yes / yes | 14.7% / 14.1% / 0.5% (Raichu base Lightning Rod on 25% of real sheets) | `onFoeRedirectTarget`, `onAnyRedirectTarget` | Rage Powder is powder: Grass types are immune. Stalwart and Snipe Shot ignore it. No Storm Drain. |
| Encore | yes / yes | 14.9% / 15.7% | Encore condition (overridden by Champions). If the target has not moved yet, onStart rewrites its queued action (`queue.changeAction`). onDisableMove locks the other moves. | High usage. Needs the action queue in state and must gate the action mask. |
| Trick Room / Tailwind | yes / yes | 29.6% / 45.6% | pseudo-weather or side condition; Champions removes Trick Room speed underflow | Dynamic speed loop (Gen 8+). |
| Weather wars (Drought, Drizzle, Sand Stream, Snow Warning, Mega Sol) | yes / yes | Drought 23%, Drizzle 17%, Snow Warning 7.5%, Sand Stream 5.3% | weather conditions are the top of the complexity ranking (34 entities mention sunnyday) | Switch-in order by speed. Mega Sol (Meganium-Mega) makes its *own* moves act as if sun is up (`onWeatherModifyDamage` plus a reference in `pokemon.ts`). |
| Grassy / Psychic / Electric terrain, Seeds | yes / yes | Grassy Surge 54%, Psychic Surge 16% | terrain conditions; `onTerrainChange` seeds | Grassy Glide priority, Psychic Terrain priority block, Expanding Force. |
| Intimidate and reactions | yes / yes | Intimidate 42.5%, Defiant 35%, Competitive 9% | `onTryBoost` blockers (Inner Focus, Oblivious, Own Tempo, Scrappy, Clear Body, Mirror Armor, Hyper Cutter…) and `onAfterEachBoost` | White Herb, Guard Dog (excluded from v1, 0.1%). No Clear Amulet, Mirror Herb or Adrenaline Orb in this format. |
| Unburden + seeds/berries | yes / yes | 34.6% | Unburden condition; `onAfterUseItem` | Item-consumption events must be exact. |
| Illusion (Zoroark, Zoroark-Hisui) | yes / yes | 0.04% / 1.1% | 4 handlers plus identity spoofing in protocol and observations | Needs an "apparent species" field in the observation. Breaks on damaging hit. |
| Transform / Imposter (Ditto) | yes / yes | 0.04% / 0.5%; 0.03% / 0.4% | `pokemon.transformInto` (core), copying species, types, stats, boosts, moves (5 PP each) and ability | Interacts with Mega, Illusion and choice lock. |
| Ally Switch | yes / yes | 0.03% / 2.2% | Gen 9 consecutive-use failure chance (stall-like) plus `swapPosition` | Targets are stored as slots, so they resolve correctly. |
| Instruct, After You, Quash | yes / yes | 0.01% / 0.7%, 0.2% / 1.2%, 0.05% / 0.9% | queue manipulation (re-run a move, reorder actions) | Needs queue-cursor state. |
| Form changes: Disguise, Stance Change, Zero to Hero, Hunger Switch | yes / yes (Forecast excluded) | 0.07% / 0.6%, 0.4% / 1.3%, 0.1% / 0.9%, 0 / 0.1% | `formeChange` (Champions override) | Stance Change also depends on King's Shield. Zero to Hero triggers on switch-out. |
| Trace, Skill Swap, Magic Bounce | yes / yes (Role Play, Entrainment excluded) | Trace 0.5% / 2.0% (and Gardevoir's base ability on 2.6% of real sheets); Skill Swap 0.3% / 2.3%; Magic Bounce 1.0% / 3.7% | ability-copy ban lists; reflection | |
| Substitute, Baton Pass, Shed Tail | yes / yes | 1.1%, 0.4%, 0.02% at 1760 Bo3 | Substitute is #2 in the automatic score (24 entities mention it) | Passing volatiles on switch. |
| Last Respects / Rage Fist / Supreme Overlord | yes / yes | 18% / 0.5% / 0.01% | fainted-ally and hit counters | Per-side and per-mon counters in state. |
| New Champions Mega abilities: Dragonize, Mega Sol, Piercing Drill, Spicy Spray, Fire Mane, Eelevate, Aura Guard | yes / yes | each 0–1.8% at 1760 Bo3 | 1–2 handlers each. Mega Sol works through `Pokemon.effectiveWeather` (the user's moves act as if sun is up). Eelevate is in `isGrounded` (Levitate-like) and also gives a best-stat boost on KO. | Small, but PS's newest code, so the risk is in the oracle rather than in the engine. |
| Commander, Protosynthesis/Quark Drive, Dancer, Neutralizing Gas, Storm Drain, Psychic/Misty Surge setters other than Indeedee's | **not legal** | | | Drop from the engine plan. |

---

## 7. Teams for testing and training

Checked by `scripts/team_sources.js` (`out/team_sources.json`). Every team gets level 50. Placeholder SP and IVs are added because showteam hides SP and leaves IVs blank. The team is then run through `TeamValidator.get('gen9championsvgc2026regmc').validateTeam`.

| source | teams | legal under Reg M-C | inside v1 | notes |
|---|---|---|---|---|
| Public Reg M-C Bo3 replays (OTS `\|showteam\|`), scraped 2026-10-20/21 | 3,211 unique | **100%** | **99.4%** | Real joint sets, but no SP and natures only. Low-mid ladder. Fetch more and filter by rating for stronger teams. |
| Public Reg M-C Bo1 replays (OTS opt-in) | 125 | n/c | 100% | Rarely available. |
| VGC-Bench HF `cameronangliss/vgc-battle-logs`: `logs_gen9championsvgc2026regm{a,b}{,bo3}.json` (M-A Bo3 is 550 MB; M-B Bo3 is 27 MB, 6,230 unique teams) | 4,000 checked | **100%** | 97.7% | Previous regulations, but still legal under M-C. No M-C file yet. |
| VGC-Bench GitHub `teams/reg_ma` (951 files), `teams/reg_mb` (545 files) | ~1.5k | not checked | not checked | Curated Champions team files, same pipeline. |
| `jakegrigsby/metamon-teams` (HF) | n/a | n/a | n/a | Singles only (gen1–4, gen9ou). Not usable. |
| `Daniel10720/pokechamp_vgc_teams` (HF) | n/a | n/a | n/a | SV Reg A–J only. Not usable. |
| PS Champions random doubles generator (`data/random-battles/champions/doubles-sets.json`, 342 species entries; `teams.ts`) | 1,000 generated | 9.1% | 68.2% | Ignores Item Clause (91% of failures) and uses level-scaled levels. Usable for exploration after repair: re-draw duplicate items, set level 50. |
| **Usage-stats sampler prototype** (chaos 1760 Bo3, restricted to v1) | 2,000 sampled | **100%** | **100%** | 159 distinct species, 1,208 distinct species sextets. Examples in `out/sampled_teams_v1_example.txt`. |

**Recommended sampling strategy** (every team must be both TeamValidator-legal and inside v1):
1. **Pool A, real sheets (evaluation and part of training).**
   - Continuously scrape Bo3 OTS teams. Optionally add VGC-Bench M-A/M-B logs and team files.
   - Keep teams inside v1. Deduplicate by sheet.
   - Fill the hidden SP by sampling from chaos `Spreads` for that species, conditioned on the shown nature.
   - Weight by player rating if you want a stronger meta.
2. **Pool B, usage sampler (training diversity).** As prototyped:
   - Draw species with the naive-Bayes teammate chain over chaos `Teammates`, applying species clause by dex number.
   - Per mon, draw ability and item from chaos marginals restricted to v1, enforcing Item Clause.
   - For Mega formes, set the base species, fix the stone, and draw the base ability from the replay distribution.
   - Draw 4 distinct moves without replacement, proportional to chaos move frequency, restricted to v1 and `checkCanLearn`-legal.
   - Draw the spread from chaos `Spreads`, i.e. "Nature:hp/atk/def/spa/spd/spe" in SP.
   - Caveat: chaos marginals lose within-set correlations, so Pool B teams are "plausible", not "real".
3. **Pool C, exploration tail (optional).** Use repaired random-doubles teams and v1-only "wildcard" sets, for robustness and for coverage-guided differential testing (§6a of the background doc).
4. **Run `TeamValidator` offline** when building the team table. The JAX engine then indexes into a fixed `[N_teams, 6, …]` tensor of pre-validated teams. Rebuild the table when PS or the regulation changes.
5. **Mixing:** start with A+B at roughly 50/50 at 1760 weights. Grow the pool over training, since VGC-Bench saw exploitability rise with pool size. Keep a held-out set of real Pool A teams for evaluation.

---

## 8. Caveats

- **Independence assumption.** Synthetic team coverage assumes moves are independent within a set. Real-sheet coverage removes that assumption, but the replay sample is low-mid ladder. At 1760, the chaos MC estimate is probably the better guide.
- **Missing base abilities.** Chaos team coverage uses the Mega forme's ability, so it cannot see base-ability failures. Only the replay check is "strict".
- **Classification thresholds.** The DATA_PLUS_COMMON vs CUSTOM line depends on the template-sharing threshold (≥3). Template ids are recorded per function, so the cost can be re-derived for any threshold.
- **Closure and core code.**
  - The condition closure follows only *creation* references. Conditions that only exist with Dynamax or Commander (dynamax, maxguard, commanded, commanding) are excluded as unreachable.
  - Core-code special cases (`hasAbility('x')` / `move.id === 'x'` in `sim/`) are counted per entity (`core_refs`), but their LoC is not included in handler LoC.
- **The oracle is new.** The Champions mod is recent code: the new Mega abilities and the mod's damage/spread overrides carry oracle risk. Re-pin and re-run the census when PS changes (`out/champions_mod_diff.json` makes the diff quick).

---

## 9. Files and reproduction

**Outputs** (all under `research/dex/`):
- **v1 lists:** `v1_species.json` (382 rows: 293 set-level + 89 battle formes), `v1_moves.json` (454), `v1_abilities.json` (172), `v1_items.json` (160). Each entry has usage stats per cutoff, the replay P(team), class, hooks and inclusion reason; each file also lists the excluded legal entities.
- **Census:** `handler_census_v1.json`, `handler_census_legal.json`.
- **Legal pool:** `legal_pool.json` (species with learnsets and abilities, battle formes, moves, abilities, items).
- **Intermediate outputs in `out/`:**
  - coverage and options: `coverage_summary.json` (mass curves and threshold sweeps for 8 cutoffs), `usage_tables.json` (per-entity P(team) and mass), `dex_options.json` (every candidate dex with cost and coverage, Mega base-ability distributions);
  - census and ranking: `census_raw_legal.json`, `complexity_v1.json`, `champions_mod_diff.json`;
  - teams and drift: `team_sources.json`, `sampled_teams_v1_example.txt`, `trend_check.json`, `synthetic_teams.json`;
  - `tables.md`, the source of the appendix.
- **Raw data in `raw/`:** `stats_2026-09/` (8 chaos files), `stats_2026-08/` (Reg M-B ×3), `replays_2026-10/` (replay indices and showteam lines), `vgcbench/` (M-B Bo3 logs).

**Reproduction** (run from `research/dex/`; PS built at `/tmp/psrepo/ps`):
```sh
node scripts/legal_pool.js /tmp/psrepo/ps/dist/sim gen9championsvgc2026regmc champions legal_pool.json
node scripts/handler_census.js /tmp/psrepo/ps champions legal_pool.json out/census_raw_legal.json
R=raw/stats_2026-09
python3 -I scripts/usage_coverage.py legal_pool.json out bo1_0=$R/gen9championsvgc2026regmc-0.json bo1_1500=$R/gen9championsvgc2026regmc-1500.json bo1_1630=$R/gen9championsvgc2026regmc-1630.json bo1_1760=$R/gen9championsvgc2026regmc-1760.json bo3_0=$R/gen9championsvgc2026regmcbo3-0.json bo3_1500=$R/gen9championsvgc2026regmcbo3-1500.json bo3_1630=$R/gen9championsvgc2026regmcbo3-1630.json bo3_1760=$R/gen9championsvgc2026regmcbo3-1760.json
python3 -I scripts/fetch_replay_teams.py gen9championsvgc2026regmcbo3 40 raw/replays_2026-10   # network; optional
python3 -I scripts/dex_select.py . legal_pool.json out/census_raw_legal.json raw/replays_2026-10/gen9championsvgc2026regmcbo3_teams.jsonl bo1_0=$R/gen9championsvgc2026regmc-0.json bo1_1760=$R/gen9championsvgc2026regmc-1760.json bo3_0=$R/gen9championsvgc2026regmcbo3-0.json bo3_1760=$R/gen9championsvgc2026regmcbo3-1760.json
python3 -I scripts/report_tables.py . out/tables.md
node scripts/team_sources.js /tmp/psrepo/ps . $R/gen9championsvgc2026regmcbo3-1760.json raw/replays_2026-10/gen9championsvgc2026regmcbo3_teams.jsonl raw/vgcbench/logs_gen9championsvgc2026regmbbo3.json 2000
python3 -I scripts/trend_check.py . mb_2026-08_bo3_1760=raw/stats_2026-08/gen9championsvgc2026regmbbo3-1760.json mb_2026-08_bo1_1760=raw/stats_2026-08/gen9championsvgc2026regmb-1760.json mb_2026-08_bo1_0=raw/stats_2026-08/gen9championsvgc2026regmb-0.json mc_2026-09_bo3_1760=$R/gen9championsvgc2026regmcbo3-1760.json
node scripts/mod_diff.js /tmp/psrepo/ps champions out/champions_mod_diff.json
```

---

## Appendix: generated tables

Generated by `scripts/report_tables.py` from the JSON outputs. In T5, a `condition.*` path is a handler on the entity's own condition. In the condition sub-table, the conditions are the closure reachable from v1.

### T1. Slot-mass coverage curves (entities needed for 90/95/99/99.9% of slot mass; total observed)

| format | battles | species (incl. megas) | moves | abilities | items (non-stone) | mega stones | mega slot share |
|---|---|---|---|---|---|---|---|
| bo1_0 | 1631943 | 92/145/236/304 (333) | 121/174/290/414 (510) | 54/77/128/178 (214) | 20/30/54/72 (85) | 32/48/68/79 (81) | 0.277 |
| bo1_1500 | 1631943 | 73/114/208/284 (316) | 107/150/255/383 (510) | 47/66/112/164 (212) | 17/25/48/68 (85) | 26/39/63/77 (81) | 0.282 |
| bo1_1630 | 1631943 | 60/90/178/258 (294) | 98/135/230/349 (508) | 41/57/97/149 (206) | 16/22/43/64 (84) | 23/31/56/75 (80) | 0.284 |
| bo1_1760 | 1631943 | 58/84/167/247 (277) | 95/131/223/340 (502) | 40/55/94/144 (203) | 16/22/42/63 (84) | 22/29/53/74 (80) | 0.285 |
| bo3_0 | 223477 | 54/82/175/266 (302) | 93/128/224/371 (504) | 39/54/97/156 (209) | 15/20/43/66 (84) | 21/31/57/75 (81) | 0.294 |
| bo3_1500 | 223477 | 47/65/135/223 (259) | 84/114/190/318 (491) | 34/47/80/133 (195) | 14/19/36/59 (84) | 19/25/47/70 (80) | 0.296 |
| bo3_1630 | 223477 | 42/56/104/176 (212) | 79/104/166/262 (478) | 32/42/69/110 (174) | 14/18/30/49 (82) | 17/22/38/58 (69) | 0.295 |
| bo3_1760 | 223477 | 39/52/92/158 (191) | 77/100/157/242 (473) | 31/40/65/102 (167) | 14/18/28/48 (81) | 17/21/33/52 (65) | 0.295 |

### T2. Candidate dexes: size, handler cost, team coverage

| dex | species | moves | abilities | items | CUSTOM ents (incl. conds) | conditions | distinct handler templates | handler fns | team cov 1760 Bo3 | 1760 Bo1 | all Bo3 | all Bo1 | real Bo3 replay teams (strict) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mass95_bo3_1760 | 50+20M | 100 | 40 | 36 (17 stones) | 106 | 42 | 212 | 285 | 0.349 | 0.130 | 0.184 | 0.052 | 0.003 |
| mass99_bo3_1760 | 83+34M | 157 | 65 | 60 (31 stones) | 154 | 51 | 292 | 402 | 0.707 | 0.395 | 0.464 | 0.213 | 0.125 |
| usage_bo3_1760>=0.001 | 97+39M | 231 | 107 | 84 (39 stones) | 230 | 59 | 403 | 572 | 0.893 | 0.655 | 0.700 | 0.457 | 0.763 |
| usage_bo3_1760>=0.0001 | 165+61M | 314 | 138 | 120 (61 stones) | 300 | 67 | 509 | 733 | 0.992 | 0.845 | 0.839 | 0.691 | 0.900 |
| usage_bo3_1760>=0.0001+bo1_0>=0.0005 | 236+81M | 426 | 177 | 157 (80 stones) | 396 | 85 | 633 | 940 | 0.999 | 0.994 | 0.978 | 0.949 | 0.980 |
| cost_aware_tau=0.01 | 293+82M | 377 | 110 | 142 (81 stones) | 220 | 57 | 369 | 622 | 0.899 | 0.778 | 0.787 | 0.620 | 0.820 |
| cost_aware_tau=0.003 | 293+82M | 423 | 145 | 151 (81 stones) | 311 | 75 | 519 | 806 | 0.978 | 0.946 | 0.924 | 0.866 | 0.957 |
| cost_aware_tau=0.001 | 293+82M | 454 | 172 | 160 (81 stones) | 376 | 83 | 603 | 919 | 0.997 | 0.991 | 0.975 | 0.957 | 0.994 |
| cost_aware_tau=0.0005 | 293+82M | 469 | 183 | 161 (81 stones) | 402 | 86 | 632 | 963 | 0.999 | 0.995 | 0.984 | 0.972 | 0.997 |
| cost_aware_tau=0.0003 | 293+82M | 481 | 192 | 161 (81 stones) | 423 | 90 | 659 | 996 | 1.000 | 0.998 | 0.992 | 0.985 | 0.999 |
| cost_aware_tau=0.0002 | 293+82M | 493 | 200 | 162 (81 stones) | 445 | 93 | 690 | 1035 | 1.000 | 0.999 | 0.996 | 0.992 | 0.999 |
| cost_aware_tau=0.0001 | 293+82M | 502 | 206 | 163 (81 stones) | 462 | 100 | 712 | 1069 | 1.000 | 1.000 | 0.998 | 0.997 | 0.999 |
| cost_aware_tau=5e-05 | 293+82M | 505 | 208 | 165 (81 stones) | 469 | 100 | 726 | 1085 | 1.000 | 1.000 | 0.999 | 0.999 | 1.000 |
| cost_aware_tau=2e-05 | 293+82M | 507 | 214 | 165 (81 stones) | 477 | 100 | 735 | 1095 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cost_aware_tau=0 | 293+82M | 510 | 215 | 166 (81 stones) | 483 | 102 | 747 | 1107 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cost_aware_bo3_1760_only_tau=0.003 | 293+82M | 361 | 114 | 139 (81 stones) | 208 | 54 | 351 | 592 | 0.886 | 0.691 | 0.721 | 0.482 | 0.804 |
| cost_aware_bo3_1760_only_tau=0.001 | 293+82M | 384 | 129 | 143 (81 stones) | 248 | 60 | 413 | 668 | 0.956 | 0.828 | 0.834 | 0.661 | 0.902 |
| cost_aware_bo3_1760_only_tau=0.0003 | 293+82M | 400 | 142 | 146 (81 stones) | 282 | 65 | 468 | 736 | 0.983 | 0.882 | 0.875 | 0.751 | 0.930 |
| cost_aware_bo3_1760_only_tau=0.0001 | 293+82M | 424 | 150 | 149 (81 stones) | 312 | 69 | 514 | 800 | 0.996 | 0.925 | 0.916 | 0.834 | 0.955 |
| full_legal_pool (v2) | 293+82M | 510 | 215 | 166 (81 stones) | 483 | 102 | 747 | 1107 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### T3-v1. Classification counts (v1)

| kind | n | DATA_ONLY | DATA_PLUS_COMMON | CUSTOM |
|---|---|---|---|---|
| move | 454 | 260 | 65 | 129 |
| ability | 172 | 6 | 23 | 143 |
| item | 160 | 7 | 124 | 29 |
| condition | 83 | 1 | 7 | 75 |

### T3-legal. Classification counts (legal)

| kind | n | DATA_ONLY | DATA_PLUS_COMMON | CUSTOM |
|---|---|---|---|---|
| move | 510 | 260 | 76 | 174 |
| ability | 215 | 6 | 24 | 185 |
| item | 166 | 7 | 124 | 35 |
| condition | 102 | 2 | 11 | 89 |

### T4-v1. Hook frequency (top 45 of 118 hook names; templates = distinct branches after parametric dedup)

| hook | entities | distinct templates | moves | abilities | items | conditions |
|---|---|---|---|---|---|---|
| onTakeItem | 82 | 3 | 0 | 1 | 81 | 0 |
| onStart | 81 | 56 | 0 | 19 | 8 | 54 |
| onHit | 62 | 50 | 56 | 2 | 0 | 4 |
| onBasePower | 52 | 26 | 10 | 15 | 20 | 7 |
| onTryHit | 43 | 36 | 21 | 15 | 0 | 7 |
| basePowerCallback | 25 | 20 | 25 | 0 | 0 | 0 |
| onEnd | 25 | 17 | 0 | 6 | 0 | 19 |
| onModifyMove | 24 | 22 | 9 | 9 | 2 | 4 |
| onSourceModifyDamage | 24 | 7 | 0 | 5 | 17 | 2 |
| onDamagingHit | 24 | 20 | 0 | 19 | 2 | 3 |
| onEat | 24 | 5 | 0 | 0 | 24 | 0 |
| onTry | 23 | 18 | 23 | 0 | 0 | 0 |
| onResidual | 23 | 20 | 0 | 6 | 2 | 15 |
| onUpdate | 20 | 12 | 0 | 10 | 8 | 2 |
| onBeforeMove | 16 | 16 | 0 | 0 | 0 | 16 |
| durationCallback | 16 | 6 | 0 | 0 | 0 | 16 |
| onPrepareHit | 13 | 8 | 10 | 3 | 0 | 0 |
| onModifyAtk | 11 | 7 | 0 | 9 | 1 | 1 |
| onRestart | 11 | 11 | 0 | 0 | 0 | 11 |
| onTryBoost | 10 | 5 | 0 | 10 | 0 | 0 |
| onSwitchIn | 10 | 10 | 0 | 3 | 0 | 7 |
| onSideStart | 10 | 4 | 0 | 0 | 0 | 10 |
| onFieldStart | 10 | 5 | 0 | 0 | 0 | 10 |
| onFieldEnd | 10 | 1 | 0 | 0 | 0 | 10 |
| onModifyType | 9 | 6 | 4 | 5 | 0 | 0 |
| onTryMove | 9 | 6 | 8 | 0 | 0 | 1 |
| onDisableMove | 9 | 6 | 2 | 0 | 0 | 7 |
| onModifySpA | 9 | 6 | 0 | 7 | 1 | 1 |
| onModifySpe | 9 | 8 | 0 | 4 | 2 | 3 |
| onImmunity | 9 | 3 | 0 | 8 | 0 | 1 |
| onTryAddVolatile | 9 | 5 | 0 | 6 | 0 | 3 |
| onSetStatus | 9 | 5 | 0 | 7 | 0 | 2 |
| onAfterSubDamage | 8 | 6 | 7 | 0 | 1 | 0 |
| onDamage | 8 | 8 | 0 | 6 | 1 | 1 |
| onAfterHit | 6 | 4 | 6 | 0 | 0 | 0 |
| onTryImmunity | 6 | 5 | 6 | 0 | 0 | 0 |
| onModifyCritRatio | 6 | 4 | 0 | 2 | 2 | 2 |
| damageCallback | 5 | 4 | 5 | 0 | 0 | 0 |
| onAfterMoveSecondary | 5 | 5 | 0 | 2 | 2 | 1 |
| onWeather | 5 | 5 | 0 | 4 | 0 | 1 |
| onMoveFail | 4 | 2 | 4 | 0 | 0 | 0 |
| onSourceModifyAccuracy | 4 | 4 | 0 | 2 | 2 | 0 |
| onAnyModifyDamage | 4 | 3 | 0 | 1 | 0 | 3 |
| onModifyDef | 4 | 4 | 0 | 3 | 0 | 1 |
| onSourceModifyAtk | 4 | 3 | 0 | 4 | 0 | 0 |

v1: 118 hook names collapse to 93 events after stripping Foe/Ally/Source/Any prefixes.


### T4-legal. Hook frequency (top 45 of 135 hook names; templates = distinct branches after parametric dedup)

| hook | entities | distinct templates | moves | abilities | items | conditions |
|---|---|---|---|---|---|---|
| onStart | 97 | 65 | 0 | 26 | 8 | 63 |
| onHit | 83 | 70 | 76 | 2 | 0 | 5 |
| onTakeItem | 83 | 4 | 0 | 2 | 81 | 0 |
| onBasePower | 55 | 29 | 11 | 17 | 20 | 7 |
| onTryHit | 49 | 42 | 26 | 16 | 0 | 7 |
| onEnd | 31 | 19 | 0 | 7 | 0 | 24 |
| onTry | 30 | 24 | 30 | 0 | 0 | 0 |
| onSourceModifyDamage | 30 | 11 | 0 | 7 | 18 | 5 |
| onResidual | 30 | 26 | 0 | 10 | 2 | 18 |
| basePowerCallback | 29 | 23 | 29 | 0 | 0 | 0 |
| onModifyMove | 28 | 26 | 10 | 11 | 2 | 5 |
| onDamagingHit | 28 | 24 | 0 | 23 | 2 | 3 |
| onEat | 28 | 8 | 0 | 0 | 28 | 0 |
| onUpdate | 25 | 16 | 0 | 11 | 11 | 3 |
| durationCallback | 19 | 6 | 0 | 0 | 0 | 19 |
| onBeforeMove | 16 | 16 | 0 | 0 | 0 | 16 |
| onTryMove | 14 | 8 | 13 | 0 | 0 | 1 |
| onPrepareHit | 13 | 8 | 10 | 3 | 0 | 0 |
| onModifyAtk | 13 | 9 | 0 | 11 | 1 | 1 |
| onFieldStart | 13 | 8 | 0 | 0 | 0 | 13 |
| onTryBoost | 12 | 7 | 0 | 12 | 0 | 0 |
| onModifySpA | 12 | 8 | 0 | 10 | 1 | 1 |
| onImmunity | 12 | 5 | 0 | 8 | 0 | 4 |
| onRestart | 12 | 12 | 0 | 0 | 0 | 12 |
| onFieldEnd | 12 | 2 | 0 | 0 | 0 | 12 |
| onDamage | 11 | 11 | 0 | 8 | 2 | 1 |
| onModifySpe | 11 | 10 | 0 | 6 | 2 | 3 |
| onSetStatus | 11 | 7 | 0 | 8 | 0 | 3 |
| onSideStart | 11 | 4 | 0 | 0 | 0 | 11 |
| onModifyType | 10 | 7 | 4 | 5 | 0 | 1 |
| onDisableMove | 10 | 7 | 2 | 0 | 0 | 8 |
| onSwitchIn | 10 | 10 | 0 | 3 | 0 | 7 |
| onTryAddVolatile | 10 | 6 | 0 | 6 | 0 | 4 |
| onAfterHit | 8 | 6 | 8 | 0 | 0 | 0 |
| onAfterSubDamage | 8 | 6 | 7 | 0 | 1 | 0 |
| onTryImmunity | 7 | 6 | 7 | 0 | 0 | 0 |
| damageCallback | 7 | 5 | 7 | 0 | 0 | 0 |
| onTrapPokemon | 7 | 4 | 0 | 1 | 1 | 5 |
| onModifyCritRatio | 6 | 4 | 0 | 2 | 2 | 2 |
| onAfterMove | 5 | 4 | 3 | 0 | 0 | 2 |
| onAfterMoveSecondary | 5 | 5 | 0 | 2 | 2 | 1 |
| onWeather | 5 | 5 | 0 | 4 | 0 | 1 |
| onTerrainChange | 5 | 2 | 0 | 1 | 4 | 0 |
| onModifyAccuracy | 5 | 5 | 0 | 3 | 1 | 1 |
| onSideEnd | 5 | 1 | 0 | 0 | 0 | 5 |

legal: 135 hook names collapse to 100 events after stripping Foe/Ally/Source/Any prefixes.


### T5. v1 hook -> entity lists, by entity type (paths; `condition.*` = handlers of the entity's own condition)


#### move (24 hook paths)

| hook | n | templates | entities |
|---|---|---|---|
| onHit | 47 | 38 | acupressure, afteryou, allyswitch, banefulbunker, batonpass, bellydrum, bugbite, clangoroussoul, clearsmog, copycat, curse, defog, detect, endure, entrainment, healpulse, instruct, kingsshield, magicpowder, moonlight, morningsun, painsplit, partingshot, pluck, pollenpuff, protect, psychup, quash, rest, shedtail, shellsidearm, simplebeam, skillswap, sleeptalk, soak, speedswap, spikyshield, steelroller, strengthsap, substitute, ... (+7) |
| basePowerCallback | 25 | 20 | acrobatics, assurance, avalanche, beatup, eruption, grassknot, gyroball, hardpress, heatcrash, heavyslam, hex, infernalparade, lastrespects, lowkick, payback, powertrip, ragefist, risingvoltage, round, stompingtantrum, storedpower, temperflare, tripleaxel, watershuriken, waterspout |
| onTry | 23 | 18 | aurawheel, auroraveil, clangoroussoul, counter, fakeout, firstimpression, followme, futuresight, lastresort, mirrorcoat, noretreat, poltergeist, quickguard, ragepowder, rest, round, sleeptalk, snore, steelroller, stockpile, suckerpunch, upperhand, wideguard |
| onTryHit | 21 | 17 | brickbreak, clangoroussoul, curse, disable, entrainment, grassknot, healingwish, heatcrash, heavyslam, helpinghand, lowkick, pollenpuff, poltergeist, psychicfangs, ragingbull, revivalblessing, shedtail, simplebeam, substitute, worryseed, yawn |
| onPrepareHit | 10 | 5 | allyswitch, banefulbunker, destinybond, detect, endure, fling, kingsshield, protect, shellsidearm, spikyshield |
| onBasePower | 10 | 7 | barbbarrage, expandingforce, facade, ficklebeam, knockoff, lashout, mistyexplosion, solarbeam, solarblade, venoshock |
| onModifyMove | 9 | 8 | beatup, blizzard, curse, expandingforce, hurricane, shellsidearm, terrainpulse, thunder, weatherball |
| onTryMove | 8 | 5 | burnup, doubleshock, electroshot, phantomforce, pollenpuff, skyattack, solarbeam, solarblade |
| onAfterSubDamage | 7 | 5 | ceaselessedge, icespinner, mortalspin, rapidspin, shellsidearm, steelroller, stoneaxe |
| onAfterHit | 6 | 4 | ceaselessedge, icespinner, knockoff, mortalspin, rapidspin, stoneaxe |
| onTryImmunity | 6 | 5 | endeavor, leechseed, octolock, switcheroo, trick, worryseed |
| secondary.onHit | 5 | 4 | alluringvoice, burningjealousy, direclaw, throatchop, triattack |
| damageCallback | 5 | 4 | counter, endeavor, finalgambit, mirrorcoat, superfang |
| onModifyType | 4 | 4 | aurawheel, ragingbull, terrainpulse, weatherball |
| onMoveFail | 4 | 2 | axekick, highjumpkick, steelbeam, supercellslam |
| self.onHit | 4 | 2 | batonpass, burnup, doubleshock, shedtail |
| priorityChargeCallback | 2 | 2 | chillyreception, focuspunch |
| beforeTurnCallback | 2 | 1 | counter, mirrorcoat |
| onDisableMove | 2 | 1 | fakeout, firstimpression |
| onHitField | 2 | 2 | haze, perishsong |
| onHitSide | 2 | 1 | quickguard, wideguard |
| beforeMoveCallback | 1 | 1 | focuspunch |
| onEffectiveness | 1 | 1 | freezedry |
| onModifyPriority | 1 | 1 | grassyglide |

#### ability (72 hook paths)

| hook | n | templates | entities |
|---|---|---|---|
| onStart | 19 | 14 | cloudnine, curiousmedicine, drizzle, drought, electricsurge, fairyaura, frisk, grassysurge, hospitality, intimidate, moldbreaker, pressure, psychicsurge, sandstream, screencleaner, snowwarning, supremeoverlord, trace, unnerve |
| onDamagingHit | 19 | 16 | cursedbody, cutecharm, electromorphosis, flamebody, gooey, illusion, innardsout, justified, mummy, poisonpoint, roughskin, seedsower, spicyspray, stamina, static, thermalexchange, toxicdebris, wanderingspirit, weakarmor |
| onBasePower | 15 | 10 | aerilate, dragonize, ironfist, megalauncher, pixilate, punkrock, reckless, refrigerate, sandforce, sharpness, sheerforce, strongjaw, supremeoverlord, technician, toughclaws |
| onTryHit | 15 | 12 | bulletproof, dryskin, eartheater, flashfire, goodasgold, lightningrod, magicbounce, oblivious, overcoat, sapsipper, soundproof, sturdy, telepathy, voltabsorb, waterabsorb |
| onTryBoost | 10 | 5 | clearbody, hypercutter, illuminate, innerfocus, keeneye, mirrorarmor, oblivious, owntempo, scrappy, whitesmoke |
| onUpdate | 10 | 5 | disguise, insomnia, limber, magmaarmor, oblivious, owntempo, thermalexchange, trace, vitalspirit, waterbubble |
| onModifyAtk | 9 | 5 | blaze, firemane, hugepower, hustle, overgrow, purepower, swarm, torrent, waterbubble |
| onModifyMove | 9 | 8 | illuminate, infiltrator, keeneye, moldbreaker, scrappy, sheerforce, skilllink, stalwart, stancechange |
| onImmunity | 8 | 2 | icebody, magmaarmor, oblivious, overcoat, sandforce, sandrush, sandveil, snowcloak |
| onModifySpA | 7 | 4 | blaze, firemane, overgrow, solarpower, swarm, torrent, waterbubble |
| onSetStatus | 7 | 3 | insomnia, leafguard, limber, purifyingsalt, thermalexchange, vitalspirit, waterbubble |
| onDamage | 6 | 6 | berserk, disguise, heatproof, magicguard, rockhead, sturdy |
| onEnd | 6 | 5 | cloudnine, flashfire, illusion, supremeoverlord, unburden, unnerve |
| onResidual | 6 | 6 | cudchew, harvest, healer, hungerswitch, moody, speedboost |
| onTryAddVolatile | 6 | 3 | innerfocus, insomnia, leafguard, owntempo, purifyingsalt, vitalspirit |
| onModifyType | 5 | 2 | aerilate, dragonize, liquidvoice, pixilate, refrigerate |
| onSourceModifyDamage | 5 | 4 | auraguard, filter, multiscale, punkrock, solidrock |
| onModifySpe | 4 | 3 | chlorophyll, sandrush, surgesurfer, swiftswim |
| onWeather | 4 | 4 | dryskin, icebody, raindish, solarpower |
| onSourceModifyAtk | 4 | 3 | heatproof, purifyingsalt, thickfat, waterbubble |
| onSourceModifySpA | 4 | 4 | heatproof, purifyingsalt, thickfat, waterbubble |
| onAllyTryAddVolatile | 3 | 3 | aromaveil, flowerveil, sweetveil |
| onSwitchIn | 3 | 3 | cloudnine, imposter, zerotohero |
| onModifyDef | 3 | 3 | furcoat, grasspelt, marvelscale |
| onPrepareHit | 3 | 3 | libero, parentalbond, protean |
| onAllyTryHitSide | 3 | 3 | magicbounce, sapsipper, soundproof |
| onSwitchOut | 3 | 3 | naturalcure, regenerator, zerotohero |
| onHit | 2 | 2 | angerpoint, owntempo |
| onFoeTryMove | 2 | 2 | armortail, queenlymajesty |
| onAfterMoveSecondary | 2 | 2 | berserk, pickpocket |
| onAfterEachBoost | 2 | 2 | competitive, defiant |
| onSourceModifyAccuracy | 2 | 2 | compoundeyes, hustle |
| onSourceAfterFaint | 2 | 2 | eelevate, moxie |
| onAllySetStatus | 2 | 2 | flowerveil, sweetveil |
| onModifyPriority | 2 | 2 | galewings, prankster |
| onModifyCritRatio | 2 | 2 | merciless, superluck |
| onHitProtect | 2 | 1 | piercingdrill, unseenfist |
| onModifyAccuracy | 2 | 2 | sandveil, snowcloak |
| onModifySTAB | 1 | 1 | adaptability |
| onTryEatItem | 1 | 1 | berserk |
| onChangeBoost | 1 | 1 | contrary |
| onEatItem | 1 | 1 | cudchew |
| onAnyTryMove | 1 | 1 | damp |
| onAnyDamage | 1 | 1 | damp |
| onCriticalHit | 1 | 1 | disguise |
| onEffectiveness | 1 | 1 | disguise |
| onSourceBasePower | 1 | 1 | dryskin |
| onEmergencyExit | 1 | 1 | emergencyexit |
| onAnyBasePower | 1 | 1 | fairyaura |
| onAllyTryBoost | 1 | 1 | flowerveil |
| onAnyModifyDamage | 1 | 1 | friendguard |
| onBeforeSwitchIn | 1 | 1 | illusion |
| onFaint | 1 | 1 | illusion |
| onAnyRedirectTarget | 1 | 1 | lightningrod |
| onAfterMoveSecondarySelf | 1 | 1 | magician |
| onWeatherModifyDamage | 1 | 1 | megasol |
| onAnyInvulnerability | 1 | 1 | noguard |
| onAnyAccuracy | 1 | 1 | noguard |
| onSourceModifySecondaries | 1 | 1 | parentalbond |
| onSourceDamagingHit | 1 | 1 | poisontouch |
| onDeductPP | 1 | 1 | pressure |
| onFractionalPriority | 1 | 1 | quickdraw |
| onFoeTrapPokemon | 1 | 1 | shadowtag |
| onFoeMaybeTrapPokemon | 1 | 1 | shadowtag |
| onModifyDamage | 1 | 1 | sniper |
| onFlinch | 1 | 1 | steadfast |
| onAllyBasePower | 1 | 1 | steelyspirit |
| onAfterSetStatus | 1 | 1 | synchronize |
| onAnyModifyBoost | 1 | 1 | unaware |
| onAfterUseItem | 1 | 1 | unburden |
| onTakeItem | 1 | 1 | unburden |
| onFoeTryEatItem | 1 | 1 | unnerve |

#### item (31 hook paths)

| hook | n | templates | entities |
|---|---|---|---|
| onTakeItem | 81 | 2 | abomasite, absolite, absolitez, aerodactylite, aggronite, alakazite, altarianite, ampharosite, audinite, banettite, barbaracite, baxcalibrite, beedrillite, blastoisinite, blazikenite, cameruptite, chandelurite, charizarditex, charizarditey, chesnaughtite, chimechite, clefablite, crabominite, delphoxite, dragalgite, dragoninite, drampanite, eelektrossite, emboarite, excadrite, falinksite, feraligite, floettite, froslassite, galladite, garchompite, garchompitez, gardevoirite, gengarite, glalitite, ... (+41) |
| onEat | 24 | 5 | aspearberry, babiriberry, chartiberry, cheriberry, chestoberry, chopleberry, cobaberry, colburberry, habanberry, kasibberry, kebiaberry, leppaberry, lumberry, occaberry, passhoberry, payapaberry, rawstberry, rindoberry, roseliberry, shucaberry, sitrusberry, tangaberry, wacanberry, yacheberry |
| onBasePower | 20 | 3 | blackbelt, blackglasses, charcoal, dragonfang, fairyfeather, hardstone, magnet, metalcoat, miracleseed, muscleband, mysticwater, nevermeltice, poisonbarb, sharpbeak, silkscarf, silverpowder, softsand, spelltag, twistedspoon, wiseglasses |
| onSourceModifyDamage | 17 | 1 | babiriberry, chartiberry, chopleberry, cobaberry, colburberry, habanberry, kasibberry, kebiaberry, occaberry, passhoberry, payapaberry, rindoberry, roseliberry, shucaberry, tangaberry, wacanberry, yacheberry |
| onStart | 8 | 5 | airballoon, choicescarf, electricseed, grassyseed, metronome, mistyseed, psychicseed, whiteherb |
| onUpdate | 8 | 5 | aspearberry, cheriberry, chestoberry, leppaberry, lumberry, mentalherb, rawstberry, sitrusberry |
| onTerrainChange | 4 | 1 | electricseed, grassyseed, mistyseed, psychicseed |
| onDamagingHit | 2 | 2 | airballoon, rockyhelmet |
| onModifyMove | 2 | 2 | choicescarf, kingsrock |
| onModifySpe | 2 | 2 | choicescarf, ironball |
| onAfterMoveSecondary | 2 | 2 | ejectbutton, redcard |
| onModifyDamage | 2 | 2 | expertbelt, lifeorb |
| onModifyCritRatio | 2 | 2 | leek, scopelens |
| onResidual | 2 | 2 | leftovers, whiteherb |
| onAfterMoveSecondarySelf | 2 | 2 | lifeorb, shellbell |
| onSourceModifyAccuracy | 2 | 2 | widelens, zoomlens |
| onAfterSubDamage | 1 | 1 | airballoon |
| onTryHeal | 1 | 1 | bigroot |
| onModifyAccuracy | 1 | 1 | brightpowder |
| onDamage | 1 | 1 | focussash |
| onEffectiveness | 1 | 1 | ironball |
| onModifyAtk | 1 | 1 | lightball |
| onModifySpA | 1 | 1 | lightball |
| onAfterSetStatus | 1 | 1 | lumberry |
| onSourceTryPrimaryHit | 1 | 1 | normalgem |
| onFractionalPriority | 1 | 1 | quickclaw |
| onTryEatItem | 1 | 1 | sitrusberry |
| onAnySwitchIn | 1 | 1 | whiteherb |
| onAnyAfterMega | 1 | 1 | whiteherb |
| onAnyAfterMove | 1 | 1 | whiteherb |
| onUse | 1 | 1 | whiteherb |

#### condition (54 hook paths)

| hook | n | templates | entities |
|---|---|---|---|
| onStart | 54 | 38 | allyswitch, aquaring, attract, banefulbunker, brn, charge, choicelock, confusion, counter, curse, destinybond, disable, dragoncheer, encore, endure, flashfire, focusenergy, focuspunch, followme, frz, futuremove, glaiverush, healblock, healreplacement, helpinghand, imprison, kingsshield, leechseed, lockedmove, metronome, mirrorcoat, mustrecharge, noretreat, octolock, par, partiallytrapped, protect, psn, ragepowder, roost, ... (+14) |
| onEnd | 19 | 12 | attract, charge, confusion, disable, encore, flashfire, futuremove, healblock, lockedmove, partiallytrapped, perishsong, saltcure, stockpile, substitute, taunt, throatchop, twoturnmove, wish, yawn |
| onBeforeMove | 16 | 16 | attract, chillyreception, choicelock, confusion, destinybond, disable, flinch, frz, glaiverush, gravity, healblock, mustrecharge, par, slp, taunt, throatchop |
| durationCallback | 16 | 6 | auroraveil, electricterrain, grassyterrain, gravity, healblock, lightscreen, mistyterrain, partiallytrapped, psychicterrain, raindance, reflect, sandstorm, snowscape, sunnyday, tailwind, trickroom |
| onResidual | 15 | 13 | aquaring, brn, curse, encore, futuremove, grassyterrain, leechseed, lockedmove, octolock, partiallytrapped, perishsong, psn, saltcure, tox, wish |
| onRestart | 11 | 11 | allyswitch, charge, futuremove, healblock, helpinghand, lockedmove, minimize, smackdown, stall, stockpile, wish |
| onSideStart | 10 | 4 | auroraveil, lightscreen, quickguard, reflect, spikes, stealthrock, stickyweb, tailwind, toxicspikes, wideguard |
| onFieldStart | 10 | 5 | electricterrain, grassyterrain, gravity, mistyterrain, psychicterrain, raindance, sandstorm, snowscape, sunnyday, trickroom |
| onFieldEnd | 10 | 1 | electricterrain, grassyterrain, gravity, mistyterrain, psychicterrain, raindance, sandstorm, snowscape, sunnyday, trickroom |
| onTryHit | 7 | 7 | banefulbunker, kingsshield, protect, psychicterrain, quickguard, spikyshield, wideguard |
| onBasePower | 7 | 6 | charge, electricterrain, gem, grassyterrain, helpinghand, mistyterrain, psychicterrain |
| onDisableMove | 7 | 5 | choicelock, disable, encore, gravity, healblock, taunt, throatchop |
| onSwitchIn | 7 | 7 | healingwish, healreplacement, spikes, stealthrock, stickyweb, tox, toxicspikes |
| onSideEnd | 4 | 1 | auroraveil, lightscreen, reflect, tailwind |
| onHit | 4 | 4 | banefulbunker, focuspunch, kingsshield, spikyshield |
| onModifyMove | 4 | 4 | frz, gravity, healblock, throatchop |
| onFieldResidual | 4 | 2 | raindance, sandstorm, snowscape, sunnyday |
| onAnyModifyDamage | 3 | 2 | auroraveil, lightscreen, reflect |
| onMoveAborted | 3 | 3 | charge, destinybond, twoturnmove |
| onDamagingHit | 3 | 2 | counter, frz, mirrorcoat |
| onTryAddVolatile | 3 | 3 | electricterrain, focuspunch, mistyterrain |
| onTrapPokemon | 3 | 3 | octolock, partiallytrapped, trapped |
| onModifySpe | 3 | 3 | par, tailwind, unburden |
| onUpdate | 2 | 2 | attract, fling |
| onAfterMove | 2 | 2 | charge, lockedmove |
| onRedirectTarget | 2 | 1 | counter, mirrorcoat |
| onModifyCritRatio | 2 | 2 | dragoncheer, focusenergy |
| onSetStatus | 2 | 2 | electricterrain, mistyterrain |
| onFoeRedirectTarget | 2 | 2 | followme, ragepowder |
| onAccuracy | 2 | 2 | glaiverush, minimize |
| onSourceModifyDamage | 2 | 2 | glaiverush, minimize |
| onLockMove | 2 | 2 | lockedmove, twoturnmove |
| onWeatherModifyDamage | 2 | 2 | raindance, sunnyday |
| onSideRestart | 2 | 1 | spikes, toxicspikes |
| onFaint | 1 | 1 | destinybond |
| onDamage | 1 | 1 | endure |
| onModifyAtk | 1 | 1 | flashfire |
| onModifySpA | 1 | 1 | flashfire |
| onAfterMoveSecondary | 1 | 1 | frz |
| onModifyAccuracy | 1 | 1 | gravity |
| onTryHeal | 1 | 1 | healblock |
| onSwap | 1 | 1 | healingwish |
| onFoeDisableMove | 1 | 1 | imprison |
| onFoeBeforeMove | 1 | 1 | imprison |
| onTryMove | 1 | 1 | metronome |
| onModifyDamage | 1 | 1 | metronome |
| onType | 1 | 1 | roost |
| onModifySpD | 1 | 1 | sandstorm |
| onWeather | 1 | 1 | sandstorm |
| onModifyDef | 1 | 1 | snowscape |
| onStallMove | 1 | 1 | stall |
| onTryPrimaryHit | 1 | 1 | substitute |
| onImmunity | 1 | 1 | sunnyday |
| onFieldRestart | 1 | 1 | trickroom |

### T6. v1 data-only field frequency


**move**: `type` 454, `target` 454, `accuracy` 454, `critRatio` 454, `category` 454, `pp` 454, `flags` 450, `basePower` 290, `secondaries` 181, `secondary` 176, `ignoreImmunity` 145, `self` 64, `boosts` 45, `volatileStatus` 39, `priority` 38, `multihit` 14, `sideCondition` 10, `recoil` 9, `selfSwitch` 8, `drain` 8, `selfBoost` 7, `stallingMove` 6, `selfdestruct` 6, `weather` 5, `heal` 5, `forceSwitch` 4, `terrain` 4, `ohko` 4, `hasCrashDamage` 3, `willCrit` 3, `slotCondition` 3, `thawsTarget` 3, `callsMove` 2, `ignoreDefensive` 2, `ignoreEvasion` 2, `breaksProtect` 2, `pseudoWeather` 2, `damage` 2, `multiaccuracy` 2, `sleepUsable` 2, `overrideOffensiveStat` 1, `smartTarget` 1, `hasSheerForceBoost` 1, `overrideOffensivePokemon` 1, `tags` 1, `overrideDefensiveStat` 1, `noPPBoosts` 1, `tracksTarget` 1, `mindBlownRecoil` 1

**ability**: `flags` 63, `onBasePowerPriority` 15, `onModifyAtkPriority` 8, `onModifySpAPriority` 6, `onResidualOrder` 6, `onModifyTypePriority` 5, `onResidualSubOrder` 5, `onSourceModifyAtkPriority` 4, `onSourceModifySpAPriority` 4, `onDamagingHitOrder` 3, `onModifyDefPriority` 3, `onTryHitPriority` 3, `onModifyMovePriority` 3, `onSourceModifyAccuracyPriority` 2, `onDamagePriority` 2, `onSwitchInPriority` 2, `onModifyAccuracyPriority` 2, `suppressWeather` 1, `onSourceBasePowerPriority` 1, `onAnyBasePowerPriority` 1, `onWeatherModifyDamagePriority` 1, `onAnyInvulnerabilityPriority` 1, `onFractionalPriorityPriority` 1, `onFractionalPriority` 1, `onAllyBasePowerPriority` 1

**item**: `fling` 159, `itemUser` 83, `megaStone` 81, `isBerry` 24, `naturalGift` 24, `onBasePowerPriority` 20, `onSwitchInPriority` 4, `boosts` 4, `onResidualOrder` 2, `onSourceModifyAccuracyPriority` 2, `onTryHealPriority` 1, `onModifyAccuracyPriority` 1, `isChoice` 1, `onAfterMoveSecondaryPriority` 1, `onDamagePriority` 1, `onModifyMovePriority` 1, `onResidualSubOrder` 1, `onModifyAtkPriority` 1, `onModifySpAPriority` 1, `onAfterSetStatusPriority` 1, `isGem` 1, `onFractionalPriorityPriority` 1, `onDamagingHitOrder` 1, `onAfterMoveSecondarySelfPriority` 1, `onAnySwitchInPriority` 1

### T7. v1 CUSTOM entities ranked by complexity score (= LoC + 6*fns + 4*core-code refs + 3*mentions by other entities)

| # | kind | id | score | fns | LoC | core refs | mentioned by | flag | hooks |
|---|---|---|---|---|---|---|---|---|---|
| 1 | condition | sunnyday | 190 | 6 | 40 | 3 | 34 |  | durationCallback, onFieldEnd, onFieldResidual, onFieldStart, onImmunity, onWeatherModifyDamage |
| 2 | move | substitute | 189 | 5 | 59 | 7 | 24 | Substitute | onEnd, onHit, onStart, onTryHit, onTryPrimaryHit |
| 3 | condition | sandstorm | 165 | 6 | 29 | 1 | 32 |  | durationCallback, onFieldEnd, onFieldResidual, onFieldStart, onModifySpD, onWeather |
| 4 | condition | substitute | 164 | 3 | 46 | 7 | 24 | Substitute | onEnd, onStart, onTryPrimaryHit |
| 5 | condition | raindance | 156 | 5 | 32 | 1 | 30 |  | durationCallback, onFieldEnd, onFieldResidual, onFieldStart, onWeatherModifyDamage |
| 6 | condition | snowscape | 134 | 5 | 26 | 0 | 26 |  | durationCallback, onFieldEnd, onFieldResidual, onFieldStart, onModifyDef |
| 7 | condition | gravity | 127 | 7 | 66 | 1 | 5 | field rule changes | durationCallback, onBeforeMove, onDisableMove, onFieldEnd, onFieldStart, onModifyAccuracy, onModifyMove |
| 8 | condition | slp | 126 | 2 | 29 | 7 | 19 |  | onBeforeMove, onStart |
| 9 | condition | lockedmove | 119 | 6 | 28 | 4 | 13 |  | onAfterMove, onEnd, onLockMove, onResidual, onRestart, onStart |
| 10 | condition | confusion | 117 | 3 | 31 | 2 | 20 |  | onBeforeMove, onEnd, onStart |
| 11 | condition | healblock | 113 | 8 | 52 | 1 | 3 | move lock | durationCallback, onBeforeMove, onDisableMove, onEnd, onModifyMove, onRestart, onStart, onTryHeal |
| 12 | condition | partiallytrapped | 103 | 5 | 25 | 9 | 4 |  | durationCallback, onEnd, onResidual, onStart, onTrapPokemon |
| 13 | condition | frz | 99 | 5 | 38 | 4 | 5 |  | onAfterMoveSecondary, onBeforeMove, onDamagingHit, onModifyMove, onStart |
| 14 | condition | futuremove | 99 | 4 | 39 | 3 | 8 |  | onEnd, onResidual, onRestart, onStart |
| 15 | condition | twoturnmove | 99 | 4 | 23 | 1 | 16 |  | onEnd, onLockMove, onMoveAborted, onStart |
| 16 | condition | electricterrain | 91 | 6 | 37 | 0 | 6 |  | durationCallback, onBasePower, onFieldEnd, onFieldStart, onSetStatus, onTryAddVolatile |
| 17 | condition | stall | 86 | 3 | 16 | 1 | 16 | turn order | onRestart, onStallMove, onStart |
| 18 | condition | grassyterrain | 85 | 5 | 34 | 0 | 7 |  | durationCallback, onBasePower, onFieldEnd, onFieldStart, onResidual |
| 19 | condition | mistyterrain | 84 | 6 | 36 | 0 | 4 |  | durationCallback, onBasePower, onFieldEnd, onFieldStart, onSetStatus, onTryAddVolatile |
| 20 | move | disable | 83 | 5 | 44 | 0 | 3 | move lock | onBeforeMove, onDisableMove, onEnd, onStart, onTryHit |
| 21 | condition | psychicterrain | 82 | 5 | 37 | 0 | 5 | priority block | durationCallback, onBasePower, onFieldEnd, onFieldStart, onTryHit |
| 22 | condition | charge | 81 | 6 | 33 | 0 | 4 |  | onAfterMove, onBasePower, onEnd, onMoveAborted, onRestart, onStart |
| 23 | move | curse | 79 | 5 | 34 | 3 | 1 |  | onHit, onModifyMove, onResidual, onStart, onTryHit |
| 24 | item | metronome | 76 | 4 | 32 | 5 | 0 | Metronome | onModifyDamage, onStart, onTryMove |
| 25 | condition | psn | 76 | 2 | 10 | 3 | 14 |  | onResidual, onStart |
| 26 | move | auroraveil | 75 | 5 | 27 | 0 | 6 |  | durationCallback, onAnyModifyDamage, onSideEnd, onSideStart, onTry |
| 27 | condition | tox | 73 | 3 | 19 | 3 | 8 |  | onResidual, onStart, onSwitchIn |
| 28 | condition | disable | 72 | 4 | 39 | 0 | 3 | move lock | onBeforeMove, onDisableMove, onEnd, onStart |
| 29 | move | throatchop | 70 | 6 | 28 | 0 | 2 | move lock | onBeforeMove, onDisableMove, onEnd, onHit, onModifyMove, onStart |
| 30 | ability | waterbubble | 69 | 6 | 33 | 0 | 0 |  | onModifyAtk, onModifySpA, onSetStatus, onSourceModifyAtk, onSourceModifySpA, onUpdate |
| 31 | condition | attract | 69 | 4 | 33 | 0 | 4 |  | onBeforeMove, onEnd, onStart, onUpdate |
| 32 | condition | brn | 69 | 2 | 12 | 3 | 11 |  | onResidual, onStart |
| 33 | condition | encore | 69 | 4 | 39 | 0 | 2 | Encore | onDisableMove, onEnd, onResidual, onStart |
| 34 | move | shedtail | 68 | 3 | 20 | 6 | 2 | Shed Tail | onHit, onTryHit |
| 35 | move | weatherball | 67 | 2 | 39 | 1 | 4 |  | onModifyMove, onModifyType |
| 36 | condition | metronome | 67 | 3 | 29 | 5 | 0 | Metronome | onModifyDamage, onStart, onTryMove |
| 37 | move | yawn | 66 | 3 | 12 | 0 | 12 |  | onEnd, onStart, onTryHit |
| 38 | ability | disguise | 66 | 4 | 34 | 2 | 0 | Disguise (form change) | onCriticalHit, onDamage, onEffectiveness, onUpdate |
| 39 | ability | flashfire | 66 | 6 | 30 | 0 | 0 |  | onEnd, onModifyAtk, onModifySpA, onStart, onTryHit |
| 40 | condition | auroraveil | 66 | 4 | 24 | 0 | 6 |  | durationCallback, onAnyModifyDamage, onSideEnd, onSideStart |
| 41 | move | counter | 65 | 6 | 26 | 0 | 1 |  | beforeTurnCallback, damageCallback, onDamagingHit, onRedirectTarget, onStart, onTry |
| 42 | move | mirrorcoat | 65 | 6 | 26 | 0 | 1 |  | beforeTurnCallback, damageCallback, onDamagingHit, onRedirectTarget, onStart, onTry |
| 43 | move | stockpile | 65 | 4 | 35 | 0 | 2 |  | onEnd, onRestart, onStart, onTry |
| 44 | item | whiteherb | 65 | 6 | 29 | 0 | 0 |  | onAnyAfterMega, onAnyAfterMove, onAnySwitchIn, onResidual, onStart, onUse |
| 45 | move | revivalblessing | 63 | 1 | 5 | 13 | 0 | Revival Blessing | onTryHit |
| 46 | condition | smackdown | 61 | 2 | 27 | 1 | 6 |  | onRestart, onStart |
| 47 | condition | throatchop | 61 | 5 | 25 | 0 | 2 | move lock | onBeforeMove, onDisableMove, onEnd, onModifyMove, onStart |
| 48 | condition | par | 60 | 3 | 20 | 1 | 6 |  | onBeforeMove, onModifySpe, onStart |
| 49 | condition | taunt | 60 | 4 | 23 | 1 | 3 | move lock | onBeforeMove, onDisableMove, onEnd, onStart |
| 50 | condition | toxicspikes | 60 | 3 | 21 | 0 | 7 |  | onSideRestart, onSideStart, onSwitchIn |
| 51 | condition | trickroom | 60 | 4 | 20 | 4 | 0 | turn-order inversion | durationCallback, onFieldEnd, onFieldRestart, onFieldStart |
| 52 | move | allyswitch | 57 | 4 | 33 | 0 | 0 | Ally Switch | onHit, onPrepareHit, onRestart, onStart |
| 53 | ability | illusion | 56 | 4 | 32 | 0 | 0 | Illusion | onBeforeSwitchIn, onDamagingHit, onEnd, onFaint |
| 54 | condition | stockpile | 56 | 3 | 32 | 0 | 2 |  | onEnd, onRestart, onStart |
| 55 | move | destinybond | 55 | 5 | 25 | 0 | 0 |  | onBeforeMove, onFaint, onMoveAborted, onPrepareHit, onStart |
| 56 | move | leechseed | 55 | 3 | 17 | 2 | 4 |  | onResidual, onStart, onTryImmunity |
| 57 | condition | kingsshield | 55 | 3 | 26 | 2 | 1 |  | onHit, onStart, onTryHit |
| 58 | condition | yawn | 55 | 2 | 7 | 0 | 12 |  | onEnd, onStart |
| 59 | move | fling | 54 | 2 | 39 | 0 | 1 | item consumption | onPrepareHit, onUpdate |
| 60 | move | focuspunch | 53 | 5 | 20 | 0 | 1 | priority charge | beforeMoveCallback, onHit, onStart, onTryAddVolatile, priorityChargeCallback |

### T8. Notorious mechanics: legal in Reg M-C? in v1?

| id | legal as | in v1 | note |
|---|---|---|---|
| commander | not legal | - | Commander (Tatsugiri/Dondozo) |
| allyswitch | move,condition | yes | Ally Switch |
| protosynthesis | not legal | - | Paradox |
| quarkdrive | not legal | - | Paradox |
| illusion | ability | yes | Illusion |
| transform | move | yes | Transform |
| imposter | ability | yes | Imposter |
| dancer | not legal | - | Dancer |
| instruct | move | yes | Instruct |
| encore | move,condition | yes | Encore |
| disguise | ability | yes | Disguise (form change) |
| stancechange | ability | yes | Stance Change (form change) |
| zerotohero | ability | yes | Zero to Hero (form change) |
| hungerswitch | ability | yes | Hunger Switch (form change) |
| forecast | ability | no | Forecast (form change) |
| trace | ability | yes | Trace |
| skillswap | move | yes | Skill Swap |
| roleplay | move | no | Role Play |
| magicbounce | ability | yes | Magic Bounce |
| neutralizinggas | not legal | - | Neutralizing Gas |
| substitute | move,condition | yes | Substitute |
| batonpass | move | yes | Baton Pass |
| shedtail | move | yes | Shed Tail |
| revivalblessing | move,condition | yes | Revival Blessing |
| afteryou | move | yes | After You |
| quash | move | yes | Quash |
| followme | move,condition | yes | redirection |
| ragepowder | move,condition | yes | redirection |
| lightningrod | ability | yes | redirection |
| stormdrain | not legal | - | redirection |
| wideguard | move,condition | yes | spread block |
| quickguard | move,condition | yes | priority block |
| fakeout | move | yes |  |
| trickroom | move,condition | yes | turn-order inversion |
| tailwind | move,condition | yes |  |
| gravity | move,condition | yes | field rule changes |
| perishsong | move,condition | yes | perish counter |
| unburden | ability,condition | yes | item-loss trigger |
| dragonize | ability | yes | Champions mega ability |
| megasol | ability | yes | Champions mega ability |
| piercingdrill | ability | yes | Champions mega ability |
| spicyspray | ability | yes | Champions mega ability |
| firemane | ability | yes | Champions mega ability |
| eelevate | ability | yes | Champions mega ability |
| auraguard | ability | yes | Champions mega ability |
