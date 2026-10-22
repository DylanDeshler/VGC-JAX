# Champions mechanics delta: PS `champions` mod vs base Gen 9, and Reg M-C format rules

*Research date: 2026-10-21. Oracle: Pokémon Showdown commit `d601443` (2026-10-21 14:24 -0500, "Add help for /setbot").*

## How to read this document

**Tags.** Every claim about PS carries one of two tags:
- **[V] VERIFIED**: I read the cited source lines, or ran PS and observed the result.
- **[I] INFERRED**: a deduction, or a single secondary source that I could not confirm.

For claims about the real game, [V] means confirmed by an official page or by two independent community sources that agree.

**Citations.** Citations are `file:line` against the PS tree.
- The original checkout at `/tmp/psrepo/ps` disappeared during this session (`/tmp` was cleaned).
- I re-cloned the same commit `d601443` into `/tmp/psrepo-delta/ps` and built it with `node build`. Line numbers are identical. I modified no PS source.
- `scripts.ts`, `moves.ts`, `items.ts`, `abilities.ts`, `conditions.ts` and `rulesets.ts` refer to files in `data/mods/champions/` unless a path says otherwise.

**Artifacts.** All are in `research/champions/`. Every script takes the PS root as an argument.

| File | What it is |
|---|---|
| `enumerate_legal.js` | Legality enumeration through PS's own `TeamValidator`. Writes `legal_*.json`, `ruleset_resolved.json` and `illegal_species_reasons.json`. |
| `ps_checks.js` → `ps_checks_output.txt` | Runtime checks run in PS: stats, PP, OTS output, Mega flow, faint persistence, Trick Room, HP display. |
| `diff_moves.js` → `move_diffs.txt` | Per-move field diff, champions vs gen9. |
| `diff_nonstandard.js` → `nonstandard_deltas.txt` | Item and ability availability deltas. |
| `diff/` | Base vs Champions function bodies used for the line-by-line diffs. |
| `compare_official.py`, `compare_bulba.py` | Eligibility cross-checks against official and Bulbapedia lists. |
| `gh/` | GitHub API commit data and patches. |
| `web/` | Downloaded official, Serebii, Bulbapedia and GitHub pages. Treated as untrusted data. |

---

## 0. TL;DR

The Champions mod is a **thin delta on top of Gen 9**:
- `gen: 9`, inheriting every Gen 9 mechanic.
- Plus `scripts.ts` (590 lines), `conditions.ts` (57), `abilities.ts` (110), `moves.ts` (1,148), `items.ts` (1,018) and `rulesets.ts` (80).
- Plus about 10 `currentMod.startsWith('champions')` branches inside core `sim/` [V].

**Numerically relevant changes an engine must implement:**

| # | Change | Base Gen 9 | Champions | Source |
|---|---|---|---|---|
| 1 | Stat formula | `floor((2B+IV+floor(EV/4))·L/100)+5` (HP `…+L+10`), nature | `B+SP+20` (HP `B+SP+75`), then nature ×1.1/×0.9 floored. **IV and level not used.** | scripts.ts:10-40 vs sim/battle.ts:2370-2388 [V] |
| 2 | Max PP | `pp·8/5` (3 PP Ups) | dex pp capped at 20, then `(pp/5+1)·4`, i.e. 5→8, 10→12, 15→16, 20→20 | scripts.ts:3-9, 41-43 vs sim/battle.ts:2390-2395 [V] |
| 3 | Trick Room speed | `trunc(10000−spe, 13)` (13-bit wrap) | `−spe`, no truncation at all | scripts.ts:46-54 vs sim/pokemon.ts:662-669 [V] |
| 4 | Paralysis full-para | 1/4 | **1/8** | conditions.ts:2-10 vs data/conditions.ts:20-46 [V] |
| 5 | Sleep length | `startTime = random(2,5)` ∈ {2,3,4} | `sample([2,3,3])` | conditions.ts:11-30 [V] |
| 6 | Freeze | 1/5 thaw per attempt, unbounded | counter 3; each attempt `time--`; thaw if `time≤0` **or** 1/4 | conditions.ts:31-56 [V] |
| 7 | Mega after faint | reverts (`formeRegression`) | stays Mega | scripts.ts:56-121 vs sim/pokemon.ts:1489 [V] |
| 8 | Rage Fist counter | never resets | reset on switch (`timesAttacked = 0` in `clearVolatile`) | scripts.ts:168 [V] |
| 9 | Unseen Fist | contact moves ignore Protect (full damage) | hit through Protect at **×0.25** (`bypassProtect`) | abilities.ts:91-100 vs data/abilities.ts:5278-5286 [V] |
| 10 | Healer | 3/10 | **1/2** | abilities.ts:39-49 vs data/abilities.ts:1818-1833 [V] |
| 11 | Salt Cure | 1/8 (1/4 Water/Steel) | 1/16 (1/8) | moves.ts:842-850 vs data/moves.ts:15643-15645 [V] |
| 12 | Move data | — | about 40 BP/acc/secondary/type/flag/PP edits (§6) | moves.ts [V] |
| 13 | Encore | override at execution time (keeps queue slot/priority) | re-queues the forced move with its **own** priority (`queue.changeAction`), target re-rolled | moves.ts:309-340 [V] |
| 14 | Fake Out / First Impression | fail via `onTry` if `activeMoveActions > 1` | additionally **disabled** (unselectable) once `activeMoveActions > 0` | moves.ts:349-356, 381-389 [V] |
| 15 | Sheer Force | suppresses AfterMoveSecondary (Berserk, Anger Shell, Pickpocket, Eject Button, Red Card, Emergency Exit) | does **not** suppress them | scripts.ts:575-585, abilities.ts:2-17 [V] |
| 16 | AfterHit after user faints | skipped if attacker fainted | runs anyway (Knock Off, Rapid Spin, Ceaseless Edge, Stone Axe, Ice Spinner, Mortal Spin, Thief, Covet) | scripts.ts:411-415 vs sim/battle-actions.ts:1180 [V] |
| 17 | Eject Button + pivot | Eject Button cancels attacker's U-turn/Volt Switch switch | both switch | items.ts:262-277 vs data/items.ts:1707-1708 [V] |
| 18 | Emergency Exit / Wimp Out | clears every active's `switchFlag` | doesn't | abilities.ts:26-34, 101-108 vs data/abilities.ts:1254-1258 [V] |
| 19 | Run Away | no battle effect | immune to trapping | abilities.ts:76-86 [V] |
| 20 | Tera | Gen 9 Tera | disabled (`canTerastallize → null`) | scripts.ts:179-181 [V] |

**Format.** `[Gen 9 Champions] VGC 2026 Reg M-C` resolves to:
- **Flat Rules** (Champions override): Obtainable, Team Preview, Species Clause, Nickname Clause, **Item Clause = 1**, **Adjust Level = 50**, **Picked Team Size = Auto → 4**, **Min Team Size = 6**, Cancel Mod, `-Mythical`, `-Restricted Legendary`;
- plus **VGC Timer** and **Open Team Sheets**;
- **no** Endless Battle Clause, Sleep Clause, OHKO Clause or Evasion Clause [V].

**Legal content** (via PS's `TeamValidator`) [V]:

| Content | Count |
|---|---|
| Species, battle-distinct formes | **263** (231 dex numbers) |
| Mega formes | **82**, from **81** Mega Stones |
| Items | **166**: 85 non-Mega items + 81 stones |
| Abilities | **216**: 203 on base formes, 13 only via Megas |
| Moves reachable through legal learnsets | **510** |

**Stat Points:**
- ≤ **32 per stat** (sim/team-validator.ts:1314-1319);
- ≤ **66 total** (`EV Limit = Auto → 66`, sim/dex-formats.ts:323-331);
- **IVs must all be 31** (team-validator.ts:1167-1169);
- **free nature** [V].

**Oracle velocity:**
- 90 commits touch `data/mods/champions*` since the mod landed on 2026-04-10/11.
- 32 commits in the last 3 months, of which about 16 change battle mechanics [V].
- The oracle changes mechanics roughly **weekly**. Re-pin PS and re-run conformance at least monthly.

---

## 1. `scripts.ts`: every override, base vs Champions

### 1.1 `init()`: PP cap (scripts.ts:3-9) [V]
- `for each move: if (pp > 20) modData('Moves', id).pp = 20`. This runs over the full dex.
- **45 moves** have no explicit entry in `moves.ts` and are capped only by this loop, e.g. Quick Attack 30→20, Bullet Punch 30→20, Ice Shard 30→20, Shadow Sneak 30→20, Light Screen 30→20, Rapid Spin 40→20, Metal Sound 40→20, Haze 30→20, Screech 40→20. The full list is in `move_diffs.txt` [V].
- `moves.ts` also sets explicit PP values that override the base (see §6).

### 1.2 `calculatePP(move, ppUps)` (scripts.ts:41-43) [V]
- **Base** (sim/battle.ts:2390-2395): `noPPBoosts ? pp : pp·(5+ppUps)/5`, with `ppUps = 3` (sim/pokemon.ts:382), so 5→8, 10→16, 15→24.
- **Champions:** `noPPBoosts ? pp : (pp/5 + 1)·4`. The `ppUps` argument is ignored.
- Observed in PS (ps_checks §1):

| Move | Dex pp | Max PP in battle | Base Gen 9 |
|---|---|---|---|
| Protect | 5 | 8 | 16 |
| Earthquake | 10 | 12 | 16 |
| Dragon Claw | 15 | 16 | 24 |
| Scale Shot | 20 | 20 | 32 |
| Fake Out | 10 | 12 | 16 |
| Quick Attack | 30→20 | 20 | 48 |

- **Engine:** `maxpp = 4·(pp_champ/5) + 4`. Precompute it per move in the data table. Showdown's chat `/dt` uses the same formula (server/chat.ts:2471) [V].
- The real-game PP values (8/12/16/20) match Bulbapedia and Serebii (`web/bulba_champions.wiki` lines 95-100; Serebii "Updated Attacks" shows Crabhammer 12, Iron Head 16, Beak Blast 8, Make It Rain 8, Meteor Assault 8) [V].

### 1.3 `statModify(baseStats, set, statName)` (scripts.ts:10-40) [V]
- **Base** (sim/battle.ts:2370-2388):
  - HP: `trunc(trunc(2B + IV + trunc(EV/4) + 100)·L/100 + 10)`;
  - other stats: `trunc(trunc(2B + IV + trunc(EV/4))·L/100 + 5)`, then nature;
  - nature uses 16-bit truncation: `tr(tr(stat·110,16)/100)`.
- **Champions**, without Level Clause Mod (which is not in Reg M-C):
  - `HP = B + SP + 75`;
  - `stat = B + SP + 20`, then the same nature code. Plus nature → `floor(stat·1.1)`, minus nature → `floor(stat·0.9)`.
  - `set.evs` holds Stat Points. **IVs and level are never read.**
  - `Overflow Stat Mod` clamps (595/728) still apply but are irrelevant here.
- **Level Clause Mod branch** (scripts.ts:14-22, random formats only): `tr((2B + 31 + max(2·SP−1, 0))·L/100) + 5` (HP `+L+10`). "The first stat point gives 4 EVs and the others give 8 EVs." At L50 this equals the main formula.
- **Equivalence (ps_checks §1):** for base ∈ {5…255} and SP ∈ 0..32, Champions(SP) equals mainline(IV 31, L50, EV = 8·SP−4 for SP ≥ 1, else 0) in **660/660** cases, for both HP and non-HP. Spot check: Garchomp 32 HP / 32 Atk / 2 Spe, Jolly → HP 215, Atk 182, Spe 136, SpA 90, matching `108+32+75`, `130+32+20`, `floor(124·1.1)`, `floor(100·0.9)` [V].
- **Engine:** compute the stats once at team load. They also need recomputing on Mega Evolution, since `setSpecies` calls `spreadModify` with the Mega base stats (sim/pokemon.ts:1408-1446).
  - **HP does not change on Mega**: `updateMaxHp` recomputes `B_hp + SP + 75`, and every Mega keeps its base HP stat [V].

### 1.4 `pokemon.getActionSpeed()` (scripts.ts:46-54) [V]
- **Base** (sim/pokemon.ts:662-669): `speed = getStat('spe')`; under Trick Room `speed = 10000 − speed`; return `trunc(speed, 13)`, i.e. modulo 8192.
- **Champions:** `speed = getStat('spe')`; under Trick Room `speed = −speed`; return `speed` with no truncation.
- Observed (ps_checks §6), base gen9 under TR:

| spe | 60 | 1808 | 1809 | 2000 |
|---|---|---|---|---|
| Base action speed | 1748 | 0 | 8191 | 8000 |
| Champions action speed | −60 | −1808 | −1809 | −2000 |

  So in base Gen 9 a Pokémon at ≥1809 effective Speed moves *first* in Trick Room (the in-game underflow). Champions removes this. At L50, ≥1809 needs, for example, +6 plus Tailwind on a ~280 Speed Pokémon [V].
- Outside Trick Room, base also wraps at 8192. Unreachable in practice: `getStat` caps Speed at 10000 (sim/pokemon.ts:657), and real values stay < 8192.
- **Engine:** the sort key is `−spe` if TR else `spe`. Plain int32, no masking.

### 1.5 `pokemon.formeChange(...)`: Mega persistence (scripts.ts:56-121) [V]
- A line-for-line copy of base `formeChange` (sim/pokemon.ts:1448-1517), with one deletion: the `this.formeRegression = true;` at the end of the `source.effectType === 'Item'` branch (base sim/pokemon.ts:1489) is removed.
- `formeRegression` drives the faint-time revert in `faintMessages` (sim/battle.ts:2606-2623): restore `baseSpecies`/`baseAbility` from the set, then `detailschange` and `updateMaxHp`.
- **Effect:** a Mega that faints keeps `species`, `baseSpecies`, `details` and `ability` as the Mega.
  - ps_checks §4: after fainting, Charizard-Mega-Y still has `species = baseSpecies = Charizard-Mega-Y` and `ability = baseAbility = drought`.
  - The control in gen9 National Dex Doubles reverts to `Charizard` (ps_checks §5).
- **Engine relevance:** only Revival Blessing (Pawmot is legal in M-C; Revival Blessing is reachable by 1 species) brings a fainted Pokémon back. It returns as a Mega with Mega stats and ability.
  - PS removed the "TODO: confirm interaction with Revival Blessing" comment in fbafa725 (2026-09-30) [V].
  - Otherwise this affects only the replay/protocol `details` string.

### 1.6 `pokemon.clearVolatile(includeSwitchFlags = true)` (scripts.ts:123-176) [V]
- Identical to base (sim/pokemon.ts:1529-1577) plus `this.timesAttacked = 0;` (scripts.ts:168).
- In base, `timesAttacked` is set only in the constructor (sim/pokemon.ts:492) and copied by Transform (1329), so Rage Fist's counter persists across switches.
- In Champions it resets on any `clearVolatile`: switch-out, faint, Baton Pass is not applicable, etc. The marker comment is at moves.ts:797-800.
- Rage Fist BP stays `min(350, 50 + 50·timesAttacked)` (data/moves.ts:14585).
- The real game confirms "Rage Fist's power now resets if the user is switched out" (Bulbapedia; Serebii Updated Attacks) [V].

### 1.7 `actions.canTerastallize` / `actions.canMegaEvo` (scripts.ts:179-194) [V]
- **`canTerastallize`** always returns `null`.
  - Base (sim/battle-actions.ts:1978-1983) returns `teraType` unless a Z-crystal, a Mega, or gen ≠ 9.
  - Other Champions branches strip Tera: no `teraType` in team data (sim/pokemon.ts:1206) and no tera validation (sim/team-validator.ts:713-727).
- **`canMegaEvo`:**
  - Mega Rayquaza (only `gen ≤ 7` or `+tag:past`/`+tag:future`; dead code here), else `item.megaStone?.[species.name] || null`.
  - Base (sim/battle-actions.ts:1923-1942) first tries `megaStone[species.name]` (only if the Mega is gen ≥ 9), then falls back to `megaStone[species.baseSpecies]`.
  - So Champions requires an **exact forme-name key**. Slowbro-Galar cannot use Slowbronite (commit 0855a802); Floette-Eternal → Floettite; Meowstic and Meowstic-F each map to their own Mega via Meowsticite.

### 1.8 `actions.modifyDamage` (scripts.ts:196-312) [V]
Line-by-line diff against sim/battle-actions.ts:1778-1903 (`diff/base_modifyDamage.ts` vs `diff/champ_modifyDamage.ts`):
- **Numerically identical**:
  - `+2`;
  - spread `modify(·, 0.75)` (0.5 in FFA);
  - Parental Bond `0.25`;
  - `priorityEvent('WeatherModifyDamage')`;
  - crit `trunc(·×1.5)`;
  - `randomizer`;
  - STAB 1.5 / `ModifySTAB` (Adaptability 2.0);
  - type ×2/÷2 loop clamped to ±6;
  - burn 0.5 (Facade exempt, Guts exempt);
  - `runEvent('ModifyDamage')` (final chain);
  - `bypassProtect` ×0.25;
  - min 1;
  - `tr(·,16)`.
- **Only changes:** protocol messages `-supereffective target <min(typeMod,2)>` and `-resisted target <min(−typeMod,2)>` (scripts.ts:270, 277; base 1854, 1861 have no magnitude). These drive the in-game "extremely effective"/"mostly ineffective" text.
- **Engine:** reuse the base Gen 9 damage pipeline unchanged. Tera and Stellar branches are dead code.
- The **0.25 `bypassProtect`** path (base: Z/Max moves only) now matters for **Unseen Fist** and **Piercing Drill** (§4).
  - It is applied **after** the final `ModifyDamage` chain and before the min-1 check.
  - It prints `-ability` (if the source is an ability) and then `-zbroken` (scripts.ts:298-305).

### 1.9 `actions.spreadMoveHit` (scripts.ts:314-423) [V]
Line-by-line diff against sim/battle-actions.ts:1080-1192:
- `if (moveData.onAfterHit && pokemon.hp)` (base 1180) becomes `if (moveData.onAfterHit)` (scripts.ts:411). Everything else is identical, apart from TS type annotations.
- **Effect:** `AfterHit` runs even if the attacker fainted mid-move, e.g. to Rocky Helmet, Rough Skin or Iron Barbs. Affected legal moves (`onAfterHit` in the mod):
  - **Knock Off** removes the item;
  - **Rapid Spin** / **Mortal Spin** clear hazards, Leech Seed and binding. Rapid Spin's +1 Speed is a separate `secondary.self` boost (data/moves.ts:14732-14736), not part of AfterHit;
  - **Ceaseless Edge** sets Spikes; **Stone Axe** sets Stealth Rock;
  - **Ice Spinner** clears terrain;
  - **Thief** / **Covet**.
- Tests: test/sim/champions.js:316, 724, 739, 788, 803 [V].

### 1.10 `actions.hitStepMoveHitLoop` (scripts.ts:426-588) [V]
Line-by-line diff against sim/battle-actions.ts:916-1079:
1. `resolveFaints` is unconditional; base guards it with `gen >= 5`. No effect in gen 9.
2. **Parental Bond `-hitcount`** is not printed when only one hit landed: `!(move.hit === 1 && multihitType === 'parentalbond')` (scripts.ts:546-547). Protocol only. Mega Kangaskhan is legal.
3. **Sheer Force no longer gates** `afterMoveSecondaryEvent` and the Emergency Exit check (scripts.ts:575-585). Base wraps both in `if (!this.battle.suppressingSecondaries())` (sim/battle-actions.ts:1064; helper at sim/battle.ts:371).
   - So when a Sheer Force user hits with a boosted move, the target-side `onAfterMoveSecondary` handlers still fire: **Berserk, Anger Shell, Pickpocket, Eject Button (data/items.ts:1699), Red Card (data/items.ts:5176), Color Change, and Emergency Exit/Wimp Out** [V].
   - In M-C, the legal ones among these are Berserk (Drampa, Mega Drampa), Pickpocket, Emergency Exit (Golisopod), Eject Button and Red Card. Anger Shell, Color Change and Wimp Out are not on any M-C species [V, `legal_abilities.json`].
   - The attacker-side `AfterMoveSecondarySelf` path (base sim/battle-actions.ts:600) is *not* overridden. Sheer Force still cancels **Life Orb recoil** (data/items.ts:3431) and **Shell Bell** (5681) [V].
   - Sheer Force holders in M-C include Mega Camerupt, Feraligatr, Steelix, Mawile, Conkeldurr, Kleavor and Tauros.
   - Tests: test/sim/champions.js:33, 71, 176, 255.

---

## 2. Champions code inside core `sim/`, outside the mod [V]

| Location | Effect |
|---|---|
| sim/dex-formats.ts:323-331 | `EV Limit = Auto` → **66** for any `champions*` mod |
| sim/team-validator.ts:1156-1169 | `useStatPoints`; **all IVs must be 31** |
| sim/team-validator.ts:1314-1319 | **≤ 32 SP per stat** |
| sim/team-validator.ts:1338-1345 | 0 total SP is an error only with nature `Serious` (use another neutral nature to signal "intentional") |
| sim/team-validator.ts:1358-1364 | Total > 66 is an error |
| sim/team-validator.ts:713-727 | No Tera type validation; `set.teraType` deleted |
| sim/dex-species.ts:748 | `learnsetParent` returns `null` for prevos: **no learnset inheritance from pre-evolutions**. Champions learnsets are complete per species. |
| sim/pokemon.ts:1206 | No `teraType`/`terastallized` in team data |
| sim/pokemon.ts:2081-2097 | Shared HP = `floor(100·hp/maxhp) \|\| 1` (base gen≥7: `ceil`, with 100→99 when not full). Colour suffix at exactly 20% (`y`/`r`) and 50% (`g`/`y`). Observed: 92/183→`50/100g`, 91/183→`49/100`, 37/183→`20/100y`, 1/183→`1/100`. |
| sim/battle-queue.ts:263-266 | Curse used by a non-Ghost: `target = 'self'` set at queue time |
| sim/battle.ts:3253 | Open Team Sheets includes **nature** for Champions |
| server/chat.ts:2471 | PP display formula |

---

## 3. Conditions (`conditions.ts`) [V]

| Status | Base (data/conditions.ts) | Champions | RNG consumption |
|---|---|---|---|
| `par` | `onBeforeMove` `randomChance(1,4)` (l.39-45); Speed ×0.5 unless Quick Feet (l.31-38) | `randomChance(1,8)` (conditions.ts:4-9); Speed and `onBeforeMovePriority: 1` inherited | 1 draw per move attempt |
| `slp` | `startTime = random(2,5)` → {2,3,4} (l.58) | `startTime = sample([2,3,3])` (conditions.ts:23). `onBeforeMove` inherited: `time--` (twice with Early Bird); wake when `time ≤ 0` and act that turn; Sleep Talk/Snore usable | 1 draw at infliction |
| `frz` | `onBeforeMove`: `randomChance(1,5)` thaw (l.96-104) | `onStart` sets `time = startTime = 3` (conditions.ts:43-44); `onBeforeMove`: `time--`; `if (time ≤ 0 \|\| randomChance(1,4))` cure and act (l.46-55). Defrost-flag moves exempt (Burn Up only if Fire). Other thaw paths inherited: Fire-type damaging hit, `thawsTarget`, defrost moves via `onModifyMove`. | 1 draw on attempts 1 and 2; **no draw on attempt 3** (`\|\|` short-circuits) |

- **Sleep semantics:** startTime 2 → asleep for 1 move attempt and wakes on the 2nd (p = 1/3). startTime 3 → asleep for 2 attempts and wakes on the 3rd (p = 2/3). The base allowed up to 3 asleep attempts. This matches official/community text: "33.3% chance of waking on Turn 2, 100% on Turn 3" (Serebii Changed Status Conditions; Bulbapedia) [V].
- **Freeze:** maximum 2 frozen attempts, guaranteed thaw on the 3rd. The counter decrements only on move attempts, not on switching or end of turn [V].
- **Engine:** `slp_ctr` ∈ {2,3} drawn as `u < 1/3 → 2 else 3`. `frz_ctr` starts at 3. Paralysis threshold 1/8. Forced-outcome test IDs: `PARA`, `SLEEP_LEN`, `FRZ_THAW`, with no `FRZ_THAW` draw when `ctr` reaches 0.

---

## 4. Abilities (`abilities.ts`) [V]

### 4.1 Modified existing abilities

| Ability | Base (data/abilities.ts) | Champions | Engine relevance |
|---|---|---|---|
| Anger Shell (2-7) | `checkedAngerShell = !(Move && !multihit && !suppressingSecondaries())` (143-149) | Sheer Force clause removed: Sheer Force hits can trigger it (and its berry gating) | No M-C species has Anger Shell [V]. Irrelevant for M-C. |
| Berserk (12-17) | same pattern (414-420) | Sheer Force clause removed | Mega Drampa (Berserk) is legal |
| Emergency Exit (26-34) / Wimp Out (101-108) | adds a loop setting every active's `switchFlag = false` (1254-1258, 5499-5503) | loop removed: other pending switches (U-turn user, Eject Button holder, a second EE holder) are not cancelled | test/sim/champions.js:49, 84, 100. In M-C only Golisopod has Emergency Exit [V]. |
| Healer (39-49) | `randomChance(3,10)` per statused adjacent ally, residual order 5.3 (1818-1833) | `randomChance(1,2)` | Mega Audino has Healer |
| Natural Cure (54-63) | `onCheckShow` reveal logic and conditional message (2812-2894) | `onCheckShow` removed; always cures on switch-out, message `[silent]` | Mechanically identical; protocol only |
| Regenerator (68-75) | `heal(baseMaxhp/3)` on switch-out, silent (3834-3842) | same heal, explicit `-heal` line | Protocol only |
| Run Away (76-86) | no battle effect (3951-3956) | `onTrapPokemon` (priority −10) sets `trapped = false`; `onMaybeTrapPokemon` sets `maybeTrapped = false` | Immune to Shadow Tag (Mega Gengar), Arena Trap, Mean Look, Block, Spirit Shackle, Octolock (legal), binding moves, Snap Trap… Affects the switch mask. |
| Unseen Fist (91-100) | `onModifyMove`: contact moves delete `flags.protect`, so full damage through every protection (5278-5286) | `onHitProtect`: contact moves set `bypassProtect = effect`, then **×0.25** damage and `-zbroken` | Mega Golurk has Unseen Fist. Covers Protect/Detect, King's Shield, Spiky Shield, Baneful Bunker, Obstruct, Silk Trap, Burning Bulwark, Wide Guard, Quick Guard, Mat Block: every protection that calls `checkMoveBypassesProtect` (data/moves.ts:1009, 2044, 9932, 11012, 12895, 13982, 14512, 16425, 17552, 20831; sim/battle.ts:1314-1323). On bypass the protection's own `onTryHit` returns early, so e.g. Spiky Shield contact damage is skipped [I from code flow]. |

### 4.2 New Champions abilities

All are defined in base `data/abilities.ts` with `isNonstandard: "Future"`; Champions sets `isNonstandard: null`.

| Ability (holder in M-C) | Implementation | Notes |
|---|---|---|
| **Aura Guard** (Mega Lucario Z) | `onSourceModifyDamage`: contact → `chainModify(0.5)`; `flags.breakable` with "TODO check breakable" (data/abilities.ts:310-319) | Made legal only on **2026-10-21** (b19c6f7a). Before that PS rejected Mega Lucario Z's ability. Very fresh. |
| **Dragonize** (Mega Feraligatr) | `-ate` pattern: Normal → Dragon, `typeChangerBoosted`, BP ×4915/4096 (1.2) at `onBasePowerPriority 23` (1036-1057) | Same exclusions as Pixilate |
| **Eelevate** (Mega Eelektross) | Ground immunity via `isGrounded` (sim/pokemon.ts:2177; also Spikes/Toxic Spikes/Sticky Web via grounding); `onSourceAfterFaint` by a move: +`length` to best stat (unboosted, unmodified) (1147-1160) | `breakable` |
| **Fire Mane** (Mega Pyroar) | `onModifyAtk`/`onModifySpA` ×1.5 for Fire moves (1296-1316) | **Open issue #12377** (2026-10-11): in-game rolls suggest a **BP modifier** (max 100 vs PS 102 in the reported case). Also filed as smogon/damage-calc#859 [V that the issue exists; game behaviour I]. |
| **Mega Sol** (Mega Meganium) | `onWeatherModifyDamage` (priority 1) calls Sunny Day's modifier, then returns (fast-exit) (2559-2572). `Pokemon.effectiveWeather()` returns `sunnyday` whenever the **active move user** has Mega Sol and the source effect is a move, weather or Mega Sol (sim/pokemon.ts:2211-2228). | So Weather Ball is Fire 100, Solar Beam has no charge, **Electro Shot charges even in rain** (89905975), Fire ×1.5 / Water ×0.5 apply for its own attacks, and defensive weather boosts (Sand SpD, Snow Def) are bypassed. Growth gives +2/+2 (the Growth handler checks `effectiveWeather`). |
| **Piercing Drill** (Mega Excadrill) | `onHitProtect` like Champions Unseen Fist: ×0.25 through protection (3283-3296) | |
| **Spicy Spray** (Mega Scovillain) | `onDamagingHit`: `source.trySetStatus('brn', target)` on **any** damaging move, not only contact (4467-4476) | Future-move edge case fixed in a19a2fba |

Mega abilities that already exist (Drought, Sand Stream, Snow Warning, Intimidate, Shadow Tag, Parental Bond, Huge Power, Magic Bounce, Trace, Contrary, Mold Breaker, Levitate, Electric Surge, No Guard, etc.) use their unchanged base implementations. The full Mega → ability table is in `legal_megas.json`.
- It matches Serebii's "New Mega Evolution Abilities" list for all 39 newly abilitied Megas (`serebii_mega_abilities.txt`) [V].

---

## 5. Items (`items.ts`, `championsregmb/items.ts`) [V]

- `items.ts` has about 250 entries.
  - Almost all only toggle `isNonstandard`.
  - It **enables 82 items** that are Past/Future in base gen9: every Champions Mega Stone, plus **Leek**.
  - It **disables 165 base-standard items** (`nonstandard_deltas.txt`).
- **The only behavioural override is Eject Button** (items.ts:262-277).
  - Base (data/items.ts:1698-1713): after `target.useItem()` succeeds it sets `source.switchFlag = false`, cancelling the attacker's U-turn, Volt Switch or Flip Turn.
  - Champions: the attacker still switches. Both the Eject Button holder's replacement and the pivot happen.
  - `onAfterMoveSecondaryPriority: 2` is inherited.
  - Only one Eject Button activates per move (the `getAllActive().some(switchFlag === true)` guard).
  - Tests: test/sim/champions.js:270, 288. The real game confirms it (Bulbapedia: "Eject Button activating no longer stops U-turn…").
- **Removed overrides**, now identical to base: White Herb desync and Disguise (removed in 10f47c9d, 2026-09-09).

**Legal non-Mega items in Reg M-C (85)** [V via `TeamValidator.checkItem`; all also on Serebii's Champions item list]:
- Utility and choice: Eject Button, Leek, Spell Tag, Air Balloon, Big Root, Binding Band, Bright Powder, **Choice Scarf** (no Choice Band or Choice Specs), Damp Rock, Expert Belt, Focus Band, **Focus Sash**, Heat Rock, Icy Rock, Iron Ball, King's Rock, **Leftovers**, Light Ball, Light Clay, **Life Orb**, Mental Herb, Metronome, Muscle Band, Normal Gem, Quick Claw, Red Card, **Rocky Helmet**, Scope Lens, Shed Shell, Shell Bell, Smooth Rock, Terrain Extender, White Herb, Wide Lens, Wise Glasses, Zoom Lens.
- Terrain seeds: Electric, Grassy, Misty and Psychic Seed.
- Type-boosting items: Black Belt, Black Glasses, Charcoal, Dragon Fang, Fairy Feather, Hard Stone, Magnet, Metal Coat, Miracle Seed, Mystic Water, Never-Melt Ice, Poison Barb, Sharp Beak, Silk Scarf, Silver Powder, Soft Sand, Twisted Spoon.
- Status and healing berries: Aspear, Cheri, Chesto, Leppa, Lum, Oran, Pecha, Persim, Rawst, Sitrus.
- Resist berries: Babiri, Charti, Chilan, Chople, Coba, Colbur, Haban, Kasib, Kebia, Occa, Passho, Payapa, Rindo, Roseli, Shuca, Tanga, Wacan, Yache.
- **Not legal:** Assault Vest, Booster Energy, Choice Band/Specs, Covert Cloak, Clear Amulet, Safety Goggles, Weakness Policy, Throat Spray, Mirror Herb, Loaded Dice, Eject Pack, Sticky Barb, Black Sludge, Heavy-Duty Boots, pinch berries (Liechi etc.), Ability Shield, Punching Glove, Protective Pads, Utility Umbrella.

**M-B → M-C deltas** (championsregmb vs champions) [V]:
- Items legal in M-C but Past in M-B: Absolite Z, Baxcalibrite, **Eject Button**, Garchompite Z, Golisopite, Leek, Lucarionite Z, Salamencite, Air Balloon, Binding Band, the four terrain Seeds, Normal Gem, Red Card, Rocky Helmet, Terrain Extender.
- Moves: `championsregmb/moves.ts` keeps Wish and Strength Sap at pp 10 (max 12). M-C uses 5 (max 8), matching the 1.2.0 patch notes "Wish/Strength Sap 12→8" (Serebii patch page).
- `championsregmb/learnsets.ts` keeps Archaludon's Mirror Coat and Metal Burst, which 1.2.0 removed.
- 33 species/formes are standard in M-C but not M-B. This matches 1.2.0's added list: Wigglytuff, Persian(-Alola), Farfetch'd, Mr. Mime, Swalot, Salamence(+Mega), Megas Absol Z, Garchomp Z, Lucario Z, Golisopod(+Mega), Baxcalibur(+Mega), Gogoat, the Galar starters, Thievul, Toxtricity(-Low-Key), Grapploct, Perrserker, Sirfetch'd, Pincurchin, Indeedee(-F), Pawmot, Arboliva, Squawkabilly ×4, Mabosstiff.
- `championsregmb/scripts.ts` is `inherit: 'champions'`, so the mechanics are identical [V].

---

## 6. Moves (`moves.ts`) [V]

`moves.ts` has 257 entries. 190 only set `isNonstandard: "Past"`, which is legality, not mechanics. The rest are below. "Reach" is the number of M-C species that learn the move; 0 means not legal in M-C, but the change is kept for future regulations.

### 6.1 Numeric and flag changes (verified with `diff_moves.js`; matches Bulbapedia's change list)

| Move | Change | Reach |
|---|---|---|
| Anchor Shot | BP 80→90 | 0 |
| Apple Acid | BP 80→90 | 2 |
| Astral Barrage | BP 120→110 (Past) | 0 |
| Baneful Bunker | pp 10→5 (max 8) | 1 |
| Beak Blast | BP 100→120, pp 15→5 | 1 |
| Blood Moon | BP 140→130 (Past) | 0 |
| Bolt Beak / Fishious Rend | BP 85→80 | 0 |
| Bone Rush | BP 25→30 | 1 |
| Clangorous Soul | acc 100→true | 1 |
| Crabhammer | acc 90→95 | 3 |
| Crush Claw, Dragon Claw, Shadow Claw, Metal Claw | +slicing flag (Sharpness: Mega Absol Z) | 4 / 29 / 67 / 0 |
| Dire Claw | secondary 50%→30%, +slicing | 1 |
| Double Shock | +punch flag (Iron Fist: Mega Crabominable) | 1 |
| Dragon Cheer | +sound flag (Throat Spray is not legal; matters for Soundproof, Liquid Voice, Punk Rock) | 23 |
| Dragon Hammer | BP 90→100 (Past) | 0 |
| Fire Lash | BP 80→90 | 1 |
| First Impression | BP 90→100 | 6 |
| Freeze-Dry | no 10% freeze (`secondary: undefined`) | 8 |
| Gear Grind | BP 50→60, acc 85→90 | 0 |
| Grav Apple | BP 80→90 | 2 |
| Growth | type Normal→Grass | 22 |
| Howl | pp 40→20; +bypasssub | 17 |
| Hyper Drill | BP 100→120 (Past) | 0 |
| Infernal Parade | BP 60→65 | 1 |
| Iron Head | flinch 30%→20% | 77 |
| King's Shield | pp 10→5; un-Past | 1 |
| Make It Rain | acc 100→95; self −1→−2 SpA | 1 |
| Meteor Assault | BP 150→170; un-Past | 1 |
| Milk Drink | target self→`adjacentAllyOrSelf` | 1 |
| Moonblast | SpA-drop chance 30%→10% | 17 |
| Mountain Gale | BP 100→120 | 1 |
| Night Daze | BP 85→90 | 1 |
| Night Slash | pp 15→20 | 37 |
| Nihil Light / Obstruct / Purify | pp→5 | 0 |
| Protect | pp 10→5 (max **8**). Detect is already pp 5. | 311 |
| Psyshield Bash | BP 70→90 | 1 |
| Revelation Dance | BP 90→100 (Past) | 0 |
| Sandstorm / Snowscape | pp 10→5 | 65 / 35 |
| Shell Trap | pp 5→10 | 0 |
| Slash | BP 70→80 | 37 |
| Snap Trap | type Grass→Steel; un-Past | 1 |
| Snipe Shot | BP 80→85 | 2 |
| Spiky Shield | pp 10→5 | 2 |
| Spin Out | pp 5→10 (Past) | 0 |
| Spirit Shackle | BP 80→90 | 1 |
| Strength Sap / Wish | pp 10→5 (M-C only) | 7 / 27 |
| Syrup Bomb | acc 85→90 | 1 |
| Toxic Thread | Spe −1→−2 | 1 |
| Triple Dive | BP 30→35 (Past) | 0 |
| Trop Kick | BP 70→85 | 1 |
| Un-Past only | Burn Up, Corrosive Gas, Electrify, Light of Ruin, Octolock, Power Shift, Storm Throw, Trick-or-Treat | various |

### 6.2 Behavioural (callback) overrides

| Move | Base | Champions |
|---|---|---|
| **Belch / Stuff Cheeks** (moves.ts:52-55, 989-992) | `onDisableMove` makes them unselectable without a (consumed) berry (data/moves.ts:1206-1207, 18257-18258) | `onDisableMove: undefined`: **selectable**, then fail via `onTry`. Legal-action mask difference. |
| **Curse** (moves.ts:165-196; sim/battle-queue.ts:263-266) | `onModifyMove` retargets; `onTryHit` rewrites a non-Ghost Curse into self-boosts; volatile via `volatileStatus` (data/moves.ts:3266-3308) | Non-Ghost: `target = 'self'` at queue time. Ghost targeting an ally: `randomNormal` foe. The Ghost check happens again at hit time (`onHit`), so a type change mid-turn (Protean, Electrify, Trick-or-Treat, Double Shock/Burn Up) decides boost vs curse. Volatile is added manually after `directDamage(maxhp/2)`. 21 tests (test/sim/champions.js:330-675). Changed 3× in Sept 2026 (23451198, 700b4f75). |
| **Dire Claw** (moves.ts:217-227) | 50%; `onHit` samples status | 30%; `sample(['psn','par','slp'])` |
| **Disable** (moves.ts:228-239) | `onBeforeMove` blocks the disabled move (data/moves.ts:3700-3706) | also exempts `flags.cantusetwice` moves, so Gigaton Hammer/Blood Moon forced by Encore pass through Disable (10f47c9d) |
| **Encore** (moves.ts:309-340) | `onStart` (+1 duration if the target already moved); `onOverrideAction` swaps the move at execution, keeping the queue position and priority (data/moves.ts:4740-4759) | `onStart`: if the target hasn't moved and its queued move ≠ the encored move (and no Mental Herb): `queue.changeAction(target, {choice:'move', moveid})`, which **cancels and re-inserts the action with re-resolved priority** (sim/battle-queue.ts:301-305, 375-410) and **no `targetLoc`**, so `getRandomTarget` re-rolls the target (sim/battle-queue.ts:268-272). `onOverrideAction` is removed. Real game: "forced move uses its usual priority" (Bulbapedia) [V]. Re-insertion can draw RNG for tie placement (`battle.random(first, last+1)`, sim/battle-queue.ts:406) [V]. Target re-roll vs the game: [I]. Tests: test/sim/champions.js:677, 691. |
| **Fake Out / First Impression** (moves.ts:349-356, 381-389) | `onTry` fails if `activeMoveActions > 1` | adds `onDisableMove` if `activeMoveActions > 0`, so **unselectable after the first turn on the field**. If forced by Encore → Struggle (Bulbapedia). |
| **Freeze-Dry** (410-413) | 10% freeze | no secondary |
| **Iron Head, Moonblast, Make It Rain, Milk Drink, Snap Trap, Toxic Thread** | — | data only (§6.1) |
| **No Retreat** (692-700) | condition `onStart`: `if (!addVolatile('trapped')) return false;` (data/moves.ts:12802-12806) | `return;` instead of `return false`. The noretreat volatile is still applied when already trapped, so it can't be used again (d12b9c1e, 2026-10-16). |
| **Rage Fist** (797-800) | — | counter reset implemented in `clearVolatile` (§1.6) |
| **Salt Cure** (842-850) | 1/8, Water/Steel 1/4 | 1/16, Water/Steel 1/8 |

---

## 7. `rulesets.ts` (mod) [V]

| Rule | Champions definition | Base difference |
|---|---|---|
| `flatrules` (35-40) | `['Obtainable','Team Preview','Species Clause','Nickname Clause','Item Clause = 1','Adjust Level = 50','Picked Team Size = Auto','Min Team Size = 6','Cancel Mod']`, banlist `['Mythical','Restricted Legendary']` | Base (data/rulesets.ts:37-43) uses `Adjust Level Down = 50`, has no Min Team Size, and also bans `Greninja-Bond` |
| `teampreview` (41-61) | Only Xerneas/Zacian/Zamazenta formes are masked. **Gourgeist sizes, Greninja and other formes are revealed**; shininess is not stripped. | Base (634-666) masks `Greninja\|Gourgeist\|Pumpkaboo\|Xerneas\|Silvally\|Urshifu\|Dudunsparce` and strips `, shiny` |
| `standardag` / `standard` / `standarddoubles` / `standarddraft` (2-34) | Champions variants for OU, Doubles and Draft | Not used by Reg M-C |
| `levelclausemod` (62-70) | Flag read by `statModify` | Champions-only |
| `natdexmod` (71-79) | NatDex Draft | — |

---

## 8. Mega Evolution in PS: full flow [V unless marked]

1. **Eligibility** is computed once at construction: `pokemon.canMegaEvo = actions.canMegaEvo(this)` (sim/pokemon.ts:511). In Champions that is `item.megaStone[baseSpecies.name]` (scripts.ts:182-194).
   - Mega Stones can't be removed from a matching holder: `onTakeItem: !item.megaStone?.[source.baseSpecies.baseSpecies]`, data/items.ts:27-28 etc.
   - So Knock Off, Trick and Thief fail, and eligibility is stable.
2. **Request:** `canMegaEvo: true` is sent per active Pokémon unless it is in a locked move (sim/pokemon.ts:1158-1161).
3. **Choice parsing:** `move <n> [target] mega` (sim/side.ts:1223-1243 parses the suffix).
   - `chooseMove` rejects it if `!pokemon.canMegaEvo` ("can't mega evolve").
   - It also rejects it if `side.choice.mega` is already set this turn ("You can only mega-evolve once per battle", sim/side.ts:767-781), so a player can't mega both actives in one turn.
   - It sets `this.choice.mega = true` (842).
4. **Queue:** `resolveAction` for a move with `action.mega` (and not Sky-Dropped) unshifts a `{choice: 'megaEvo'}` action (sim/battle-queue.ts:211-216).
   - Order constants: `switch 103`, `megaEvo 104`, `runDynamax 105`, `terastallize 106`, `priorityChargeMove 107`, moves 200, residual 300 (177-197).
   - The megaEvo action has priority 0 and `speed = pokemon.getActionSpeed()` at turn start (sim/battle.ts:2672-2715).
   - The queue is sorted by `comparePriority` (order ↑, priority ↓, speed ↓, subOrder, effectOrder; sim/battle.ts:409-416), with random shuffles of exact ties (`speedSort`, 434-466).
5. **Execution** (`runAction` case `'megaEvo'` → `actions.runMegaEvo`, sim/battle.ts:2766-2768; sim/battle-actions.ts:1952-1971):
   - `formeChange(megaSpecies, item, isPermanent = true)` runs `setSpecies`: new types, weight and stats via `spreadModify`. HP is unchanged (the Mega has the same base HP).
   - Then `baseSpecies = mega`, protocol `detailschange` then `-mega`, `moveThisTurnResult = true` (Truant), `updateMaxHp`.
   - Then `setAbility(megaAbility, null, null, isFromFormeChange = true)` and `baseAbility = megaAbility` (sim/pokemon.ts:1504-1513).
   - `setAbility` fires the old ability's `End` and **immediately fires the new ability's `Start`** (sim/pokemon.ts:1929-1970). **Switch-in-style abilities activate at the moment of Mega Evolution**: Drought, Sand Stream, Snow Warning, Intimidate (Mega Manectric, Mega Scrafty), Electric Surge (Mega Raichu X), Trace (Mega Alakazam/Meowstic), Mold Breaker announcement, etc.
   - Observed (ps_checks §4): Mega Tyranitar's `-weather|Sandstorm|[from] ability: Sand Stream` prints directly after its `-mega` line, before any move. Mega Charizard Y's Drought printed nothing because sun was already up from Torkoal: `setWeather` to the same weather fails silently.
   - Then **every** Pokémon on the side gets `canMegaEvo = false`, which is the one-per-battle rule.
   - Then `runEvent('AfterMega')` (Eject Pack's `onAnyAfterMega`; Eject Pack isn't legal).
   - Champions keeps `formeRegression = false` (§1.5).
6. **Ordering among megas:** all megaEvo actions (both sides) run before any move.
   - They run in **start-of-turn speed order**. After a megaEvo action the gen 8+ re-sort happens only if the *next* action is a `move`/`runDynamax` (sim/battle.ts:2977-2984), so back-to-back megas are not re-sorted.
   - Mega ability Start effects therefore resolve in pre-Mega speed order. Weather wars between two Mega weather setters go to the **slower pre-Mega Pokémon's** weather (last writer wins) [V from code; ps_checks §4 shows Charizard (faster) then Tyranitar].
7. **Same-turn speed consequences (gen 8+ dynamic speed):**
   - After the last megaEvo action, `updateSpeed()` and `getActionSpeed()` recompute every queued action and `queue.sort()` runs.
   - **The Mega's new Speed decides its move order this very turn.** So does any new Speed-modifying ability, e.g. Mega Swampert's Swift Swim in rain via `ModifySpe`. Speed Boost only acts at end of turn.
   - Observed (ps_checks §3): Camerupt (Spe 60) Mega Evolves into Camerupt-Mega (Spe 40) and then moved **after** Slowbro (Spe 50) in the same turn.
   - Gen 7's special-case (re-insert only the mega user's move, sim/battle.ts:2911-2921) does **not** apply (gen 9).
   - Priority is unaffected, except Prankster (Mega Banette): priority is recomputed in the re-sort through `ModifyPriority`, so a Mega Banette's status move gains +1 on the Mega turn [I from code: `getActionSpeed` recomputes priority from the base move each call].
8. **Persistence:** the Mega stays a Mega through switches (`clearVolatile` → `setSpecies(baseSpecies)` = Mega) and, in Champions, through fainting (§1.5).

**Legal Megas in Reg M-C (82 formes, 81 stones)** [V; all validate with a full set]:
- Mega abilities, types and stats are in `legal_megas.json`.
- The list matches the official M-C notice (newly added in M-C: Mega Absol Z, Mega Salamence, Mega Garchomp Z, Mega Lucario Z, Mega Golisopod, Mega Baxcalibur; `web/official/page_816.html`).

| Stone(s) | Species → Mega |
|---|---|
| Abomasite, Absolite, **Absolite Z**, Aerodactylite, Aggronite, Alakazite, Altarianite, Ampharosite, Audinite, Banettite, Barbaracite, **Baxcalibrite** | Abomasnow, Absol (Mega and **Mega-Z**), Aerodactyl, Aggron, Alakazam, Altaria, Ampharos, Audino, Banette, Barbaracle, Baxcalibur |
| Beedrillite, Blastoisinite, Blazikenite, Cameruptite, Chandelurite, Charizardite X/Y, Chesnaughtite, Chimechite, Clefablite, Crabominite, Delphoxite, Dragalgite, Dragoninite, Drampanite | Beedrill, Blastoise, Blaziken, Camerupt, Chandelure, Charizard X/Y, Chesnaught, Chimecho, Clefable, Crabominable, Delphox, Dragalge, Dragonite, Drampa |
| Eelektrossite, Emboarite, Excadrite, Falinksite, Feraligite, Floettite, Froslassite, Galladite, Garchompite, **Garchompite Z**, Gardevoirite, Gengarite, Glalitite, Glimmoranite, **Golisopite**, Golurkite, Greninjite, Gyaradosite | Eelektross, Emboar, Excadrill, Falinks, Feraligatr, Floette-Eternal, Froslass, Gallade, Garchomp (Mega and **Mega-Z**), Gardevoir, Gengar, Glalie, Glimmora, Golisopod, Golurk, Greninja, Gyarados |
| Hawluchanite, Heracronite, Houndoominite, Kangaskhanite, Lopunnite, Lucarionite, **Lucarionite Z**, Malamarite, Manectite, Mawilite, Medichamite, Meganiumite, Meowsticite, Metagrossite | Hawlucha, Heracross, Houndoom, Kangaskhan, Lopunny, Lucario (Mega and **Mega-Z**), Malamar, Manectric, Mawile, Medicham, Meganium, Meowstic-M and Meowstic-F (one stone), Metagross |
| Pidgeotite, Pinsirite, Pyroarite, Raichunite X/Y, Sablenite, **Salamencite**, Sceptilite, Scizorite, Scolipite, Scovillainite, Scraftinite, Sharpedonite, Skarmorite, Slowbronite, Staraptite, Starminite, Steelixite, Swampertite, Tyranitarite, Venusaurite, Victreebelite | Pidgeot, Pinsir, Pyroar, Raichu X/Y, Sableye, Salamence, Sceptile, Scizor, Scolipede, Scovillain, Scrafty, Sharpedo, Skarmory, Slowbro (not Galar), Staraptor, Starmie, Steelix, Swampert, Tyranitar, Venusaur, Victreebel |

---

## 9. Format legality: `[Gen 9 Champions] VGC 2026 Reg M-C`

### 9.1 Ruleset chain (resolved by `Dex.formats.getRuleTable`; `ruleset_resolved.json`) [V]

- **Format** (config/formats.ts:296-300): `mod: 'champions'`, `gameType: 'doubles'`, `bestOfDefault: true`, `ruleset: ['Flat Rules', 'VGC Timer', 'Open Team Sheets']`.
- **The Bo3 variant** (303-306): `['Flat Rules', 'VGC Timer', 'Force Open Team Sheets', 'Best of = 3']`.

**Flat Rules (Champions override):**
- `Obtainable`, which expands to `Obtainable Moves/Abilities/Formes/Misc`, `EV Limit = Auto` (→ 66), `-Unreleased`, `-Unobtainable`, `-Nonexistent` (data/rulesets.ts:166-171).
- `Team Preview`, `Species Clause` (by dex number, so Rotom formes are exclusive; data/rulesets.ts:788-805), `Nickname Clause`, `Item Clause = 1` (806-869), `Adjust Level = 50`, `Picked Team Size = Auto`, `Min Team Size = 6`, `Cancel Mod`.
- `-tag:mythical`, `-tag:restrictedlegendary`. None exist in the Champions dex anyway.

**Derived values:**
- `pickedTeamSize = 4` (doubles; sim/dex-formats.ts:315-321): **bring 6, pick 4** (`|teampreview|4`, `maxChosenTeamSize: 4`).
- `minTeamSize = maxTeamSize = 6`; `adjustLevel = 50` (every Pokémon battles at L50); `evLimit = 66`; `maxMoveCount = 4`; `minSourceGen = 9`.

**VGC Timer** (data/rulesets.ts:778-787): `Timer Starting = 420`, `Grace = 90`, `Add Per Turn = 0`, `Max Per Turn = 55`, `Max First Turn = 90`, `Timeout Auto Choose`, `DC Timer Bank`. Server-side only (server/room-battle.ts:438-460).

**Open Team Sheets** (data/rulesets.ts:1960-1981):
- An opt-in button at team preview.
- If **both** players accept before turn 1 (server/chat-commands/core.ts:971-999), the stream receives `>show-openteamsheets` and calls `battle.showOpenTeamSheets()` (sim/battle-stream.ts:209-210).
- **Force OTS** (Bo3, data/rulesets.ts:1982-1995) calls it unconditionally `onTeamPreview`.

**Absent from Reg M-C:** Endless Battle Clause, Sleep Clause Mod, OHKO Clause, Evasion Clause, Gravity Sleep Clause, Terastal Clause.

### 9.2 What OTS reveals [V]

`showOpenTeamSheets` (sim/battle.ts:3242-3281) emits one `|showteam|<side>|<packed team>` line per side during team preview (`turn === 0` only).
- **Revealed:** species, item, ability, moves, **nature** (Champions only, line 3253), gender, level.
- **Not revealed:** nickname (blank), Stat Points/EVs, IVs (both `null`), shininess, Tera (none).
- Observed (ps_checks §2), Bo3: `|showteam|p1|Garchomp||Garchompite|RoughSkin|protect,earthquake|Jolly||M|||50|]Incineroar||SitrusBerry|Intimidate|fakeout|Hardy||F|||50|…`, printed right after `|teampreview|4`.
- **Engine hidden information under OTS:** the Stat Point spread (66 points over 6 stats, ≤ 32 each), which 4 were brought, the leads, and current PP/HP. Nature is known, so the Speed tier is ambiguous only through Speed SP.

### 9.3 Legal content (from `enumerate_legal.js` → `legal_*.json`) [V]

| Category | Count | Notes |
|---|---|---|
| Species accepted by validator | 312 names | Includes 19 Gmax placeholder names, which the validator rewrites to the base forme; not real |
| Excluding Gmax placeholders | 293 | |
| **Battle-distinct formes** | **263** | Cosmetic duplicates collapsed: Vivillon ×19, Alcremie ×7, Polteageist-Antique, Sinistcha-Masterpiece, Squawkabilly colour pairs |
| Distinct dex numbers | 231 | |
| **Mega formes** | **82** (81 stones) | |
| **Items** | **166** | 85 non-Mega + 81 Mega Stones |
| **Abilities** | **216** | 203 on base formes; 13 Mega-only: Aerilate, Aura Guard, Dragonize, Eelevate, Fairy Aura, Fire Mane, Innards Out, Mega Sol, Parental Bond, Piercing Drill, Shadow Tag, Spicy Spray, Unseen Fist |
| **Moves reachable** | **510** | Plus 5 generic-legal but unlearnable: Pound, Power Shift, Soft-Boiled, Spore, Struggle |

**Cross-checks:**
- **Official list.** The M-C eligible list (`web/official/mc_pokemon.html`, 262 entries, from the in-game web-view linked from the official news page 816) matches PS's 263 battle-distinct formes **exactly modulo naming**: "Rotom (Heat Rotom)" = Rotom-Heat, Gourgeist varieties, M/F formes, and Squawkabilly counted as 1 vs 2 ability variants [V].
- **Bulbapedia.** Bulbapedia's M-C page omits **Kingambit**, but the official list and PS include it. Bulbapedia is wrong there [V].
- Spot check: Incineroar abilities Blaze/Intimidate with 77 moves. Garchomp has 59 moves (e.g. no Knock Off). Both are consistent with Bulbapedia's note that Incineroar cannot use Knock Off in Champions [V].

### 9.4 Stat Points, natures, IVs, level [V]

- **Per stat:** 0..32 (sim/team-validator.ts:1314-1319; error "more than 32 Stat Points").
- **Total:** ≤ 66 (`evLimit = 66`, sim/dex-formats.ts:328-330; error at team-validator.ts:1358-1364).
- **Negative SP:** rejected (1298-1303).
- **0 total SP:** rejected only with nature `Serious` (1340-1345), as an intentional-mistake heuristic.
- **IVs:** must all be 31 (1167-1169). They are ignored by `statModify` anyway.
- **Nature:** any of the 25 natures, no restriction. In-game it is called "stat alignment" and can be set freely via training (Bulbapedia `Stat_point`; `web/bulba_Stat_point.wiki`).
- **Level:** `Adjust Level = 50`. Level is unused by `statModify` but **is** used in the damage formula (`2·50/5+2 = 22`) and level-based moves (Seismic Toss and Night Shade deal 50).
- **Real game** (Bulbapedia): "32 stat points per stat, 66 in total; HP = Base + SP + 75; Other = ⌊(Base + SP + 20) × Alignment⌋". This matches PS exactly [V].

---

## 10. Timers, Bo3, turn limits (engine relevance) [V]

- **PS** has no Endless Battle Clause in Reg M-C. It keeps the **hard 1000-turn auto-tie**, which is not part of EBC (sim/battle.ts:1855-1869), with warnings from turn 500.
- The timer is server-side (420 s bank, 55 s/turn, 90 s first turn and preview, auto-choose on turn timeout, loss when the bank empties). It doesn't exist in the sim.
- **Real game, Reg M-C** (official notice page 816 and Bulbapedia):

| Timer | Value |
|---|---|
| Total game time | 20 min |
| Player time | 7 min |
| Turn time | 45 s |
| Preview | 90 s |

  - PS has **no 20-minute game clock**, and the turn timer is 55 s vs 45 s.
  - I found no public source for the in-game tiebreak when the 20-minute limit is hit [I: traditionally VGC uses remaining-Pokémon count, then HP%].
- **Bo3:** `Best of = 3` with Force OTS. Each game is an independent battle; the series state is server-side.
- **Engine recommendation:** a configurable `max_turns` (truncate with a tie or an HP-based score) for training efficiency, plus Bo3 handled by the training harness. Don't emulate timers.

---

## 11. Real game vs PS: matches, discrepancies, open questions

### 11.1 Confirmed matches (PS behaviour equals official/community data) [V]
- Stat formula, SP caps, IV31, L50.
- PP 8/12/16/20 and every listed exception: Protect/King's Shield/Spiky Shield/Baneful Bunker 8, Beak Blast 8, Night Slash 20, Sandstorm/Snowscape 8, Wish/Strength Sap 8 from 1.2.0.
- Paralysis 12.5%, freeze rules, sleep 2–3.
- Healer 50%, Unseen Fist 25%, Run Away vs trapping, Eject Button not stopping pivots.
- Every move BP/accuracy/secondary/flag change in Bulbapedia's list and Serebii's "Updated Attacks". The hidden "not yet in game" list in Bulbapedia's source comments (Gear Grind, Anchor Shot, etc.) also matches PS values.
- New abilities as described by Serebii, except Fire Mane (below).
- The legal species list (official), the legal item list (Serebii; all 166 PS-legal items appear on it), and Mega abilities (Serebii).

### 11.2 Known or possible discrepancies

| Item | Status |
|---|---|
| **Fire Mane** ×1.5 Atk/SpA vs possible BP modifier | **Open** PS issue #12377 (2026-10-11), damage-calc#859. One Japanese in-game measurement (note.com/yua_freewind/n/n21207ecbfe3d). Affects only Mega Pyroar. |
| **Simultaneous stat-change resolution.** In-game, Intimidate-style multi-target drops "are now resolved on both opposing Pokémon at once" (Bulbapedia). PS still runs `boost()` per target sequentially. | The order of reactions (Defiant/Competitive, Mirror Herb is not legal, White Herb, Adrenaline Orb is not legal) could differ in edge cases [I]. No PS issue found. |
| **Speed changes from held items vs ability activation order.** Fixed in game 1.0.3 ("changes in speed caused by held items are not reflected in the order in which abilities activate"). | PS already orders switch-in abilities by current speed including items [I]. Nothing to change. |
| **Encore target re-roll.** PS re-rolls the target randomly when Encore re-queues a different move (no `targetLoc`). | Not documented for the real game [I]. PR #12041 ("preserve original target location", May 2026) was closed without merging. |
| **Aura Guard `breakable`** marked "TODO check" (data/abilities.ts:315); made legal in PS only today (b19c6f7a). | Unverified [I] |
| **Open PRs:** #12358 (frozen-turns message, UI) and #12048 (Champions HP mechanics without HP Percentage Mod). | Protocol only |
| **HP display.** PS shows the opponent `floor(%)`. Game behaviour per PR #11920. | Matches the in-game bar, assumed [I] |
| **PS-side bans** (Champions OU/UU tier bans, e.g. "Ban Mega Raichu Y" 1daa64d3) | **Do not** affect VGC Reg M-C, because Flat Rules has no tier bans [V] |

### 11.3 Regulation timeline (official news list `web/official/list.json`, Bulbapedia, Serebii) [V]

| Event | Date |
|---|---|
| Champions release | **2026-04-08** on Switch (04-07 in North America); **2026-06-17** on iOS/Android |
| Reg M-A | 2026-04-08 02:00 UTC → 2026-06-17 01:59 UTC |
| Reg M-B | 2026-06-17 → 2026-09-09. The 2026 Worlds, San Francisco, Aug 28-30, used Champions with **Reg M-B**. |
| Reg M-C | **2026-09-09 02:00 UTC → 2026-12-02 01:59 UTC** (official notice, page 816). Ranked Seasons M-6 (09-09→10-07) and M-7 (10-07→11-04). |
| Next regulation | Not announced as of 2026-10-21. No "M-D" page on Bulbapedia or in the official news list [V absence]. By the end date, the next set is expected **2026-12-02** [I]. |

Game patches:
- **1.0.3** (2026-04-23): bug fixes, including Unnerve and item-speed ability order.
- **1.1.0** (2026-06-17): new Pokémon, M-B. PS "Implement 1.1.0 changes", 665f507b.
- **1.2.0** (2026-09-09): 33 additions, Wish/Strength Sap PP, movepool removals. PS "Add Champions Regulation M-C", 812501ed.
- A maintenance on 2026-10-07 with unspecified "some bugs have been fixed" (official page 844).

### 11.4 PS commit history for `data/mods/champions*`

Sources: GitHub API listing (`gh/commits_champions.json`, 87 commits) and the local full-history clone [V].
- **Totals:** 90 commits since "Implement Champions" (b555f4d8, 2026-04-10/11). 32 in the last 3 months (since 2026-07-21).
- **Line churn in the last 3 months:** learnsets +1814/−138, formats-data +661/−356, moves +79/−82, items +41/−69, abilities +36/−26, scripts +18/−24, conditions +3/−3.

Mechanics-changing commits in the last 3 months, newest first:

| Commit | Date | Change |
|---|---|---|
| b19c6f7a | 10-21 | Aura Guard made legal (was inheriting `Future`) |
| 035ba1a5 | 10-18 | Mega Stone inheritance (Darkranite/Heatranite entries removed; non-M-C species) |
| d12b9c1e | 10-16 | No Retreat can't be re-used when already trapped (base also changed) |
| a7ce30c0 | 10-05 | +780 test lines; Emergency Exit/Wimp Out signature `onEmergencyExit(originalHp, target)`; Sheer Force + Emergency Exit; base `suppressingSecondaries()` refactor |
| 89905975 | 10-01 | Mega Sol forces Electro Shot charge; Encore re-queue simplified (`onOverrideAction: undefined`) |
| fbafa725 | 09-30 | Sheer Force doesn't suppress AfterMoveSecondary |
| 700b4f75 | 09-30 | Curse + Follow Me |
| 23451198 | 09-20 | Curse + type changes, including sim/battle-queue.ts |
| 57ecb348 | 09-20 | Emergency Exit doesn't cancel other switches |
| aa6d5f08 | 09-13 | Eject Button doesn't cancel pivot |
| d849b220 | 09-10 | Run Away trapping immunity |
| bc4eb07b | 09-10 | Double Shock punch flag |
| d3de52a1 | 09-10 | Learnset fixes |
| 812501ed | 09-09 | **Reg M-C** data |
| 10f47c9d | 09-09 | Multiple M-A bugs: Disguise and White Herb overrides dropped, Disable vs `cantusetwice`, Ceaseless Edge/Stone Axe after faint |
| f0039f57 | 08-08 | Spicy Spray/Dire Claw immune messages |

The rest are tier bans and format plumbing.

**Rate:** roughly 1–2 behavioural fixes per week during Sept–Oct 2026, clustered around regulation changes. Each new regulation brings large learnset, formats-data and item diffs. Expect the next burst around 2026-12-02.

---

## 12. Recommended Champions-specific engine features (with priority)

**P0** = required for any faithful Reg M-C training. **P1** = needed before a conformance target of ≥ 90% turn match. **P2** = observation and mask fidelity. **P3** = optional or later.

| # | Feature | Priority | Notes and how |
|---|---|---|---|
| 1 | **Ruleset/regulation config** (M-B vs M-C data tables: species, items, PP overrides, learnsets) generated from PS `Dex.mod('champions'/'championsregmb')` | P0 | Use `enumerate_legal.js` output as the generator. Expect a refresh about every 12 weeks. |
| 2 | **Stat Points stat calc** (`B+SP+20`, HP `B+SP+75`, nature floor) at team load; recompute non-HP stats on Mega | P0 | int16 |
| 3 | **PP table** `maxpp = 4·(pp_c/5)+4` with the Champions pp overrides and the cap of 20 | P0 | Protect 8 is important for Protect-stall dynamics |
| 4 | **Status RNG**: para 1/8; sleep ctr {2: 1/3, 3: 2/3}; freeze ctr 3 with 1/4 thaw draws only while ctr > 0 | P0 | Fixed-shape counters |
| 5 | **Mega Evolution**: action flag (`move … mega`); order-104 phase before moves; mega order by **start-of-turn** speed; species/types/stats/weight/ability swap; immediate ability `Start` (weather, terrain, Intimidate, Trace…); per-side "mega used" bit plus per-turn exclusivity in the joint action mask; full dynamic re-sort of the remaining actions with post-Mega speed and priority; Mega persists on switch and faint | P0 | Biggest Champions-specific control-flow item. 82 Mega formes means a Mega species table of [82] stat/type/ability rows. |
| 6 | **Trick Room speed = −spe** (no 13-bit wrap) | P0 | Simpler than base |
| 7 | **No Tera** anywhere (drop tera branches in STAB, type, obs) | P0 | |
| 8 | **Champions move data** (§6.1 numbers, flags slicing/sound/punch, Growth Grass, Snap Trap Steel, Freeze-Dry no freeze, Make It Rain −2, Milk Drink ally-target) | P0 | Data only |
| 9 | **Fake Out / First Impression unselectable after the first active turn**; Belch/Stuff Cheeks selectable-but-fail | P0 (mask) | Legal-action-mask parity with PS requests |
| 10 | **Eject Button + pivot both switch**; **Emergency Exit doesn't cancel other switch flags**; only one Eject Button per move | P1 | Mid-turn switch phase must allow 2 pending switches from one move |
| 11 | **AfterHit runs even if the attacker fainted** (Knock Off, Rapid Spin, Ceaseless Edge, Stone Axe, Ice Spinner, Thief/Covet) | P1 | Knock Off is reachable by 81 species; Rocky Helmet is legal in M-C |
| 12 | **Sheer Force doesn't suppress AfterMoveSecondary** (Berserk, Anger Shell, Pickpocket, Red Card, Eject Button, Emergency Exit) but still suppresses Life Orb recoil and secondaries | P1 | |
| 13 | **Encore = re-queue with the forced move's own priority** (plus a random target re-roll and a possible tie-placement draw) | P1 | Matters with Prankster Encore (Whimsicott, Mega Banette) vs priority moves |
| 14 | **Protect-bypass ×0.25** for Unseen Fist (Mega Golurk) and Piercing Drill (Mega Excadrill) across all 10 protection moves; skip the protection's punish effect | P1 | Uses the existing `bypassProtect` damage step |
| 15 | **New abilities**: Mega Sol (sun-for-user override of `effectiveWeather`, including Electro Shot/Solar Beam/Weather Ball/defensive weather bypass), Dragonize, Fire Mane (as PS: ×1.5 Atk/SpA, with a flag for a future BP-mod fix), Aura Guard, Eelevate, Spicy Spray, Piercing Drill | P1 | Seven handlers; only on Megas |
| 16 | **Small handler deltas**: Healer 1/2, Run Away anti-trap (switch mask), Salt Cure 1/16 (1/8), Rage Fist reset on switch, Curse Champions targeting, Disable vs `cantusetwice`, No Retreat re-use block | P1 | Each one is tiny |
| 17 | **Mega persistence through faint** (only visible with Revival Blessing; Pawmot is legal) | P2 | Free if the faint code doesn't revert species |
| 18 | **Observation**: opponent HP as `floor(100·hp/maxhp)` (min 1) with the 20%/50% colour bits; OTS reveals species/item/ability/moves/nature/gender; hidden = SP spread, the brought 4, leads | P2 | Belief model over SP spreads (66 points). Nature is known, so Speed tiers are ambiguous only via Speed SP. |
| 19 | **Team preview**: bring 4 of 6 with ordered leads (180 options); Gourgeist sizes visible | P2 | |
| 20 | **Turn cap / tiebreak**: configurable (PS's own is 1000 turns; in-game 20 min) | P3 | Training-efficiency knob |
| 21 | **Conformance CI pinned to a PS commit**, re-pinned at least monthly and on every regulation change. Track the Champions commit list (`gh api …/commits?path=data/mods/champions`) as a mechanics changelog. | P0 (process) | The oracle moved about 16 mechanics times in 3 months |
| 22 | **Feature flags for known-disputed mechanics** (Fire Mane BP vs Atk, simultaneous Intimidate resolution) so the engine can follow PS or the game | P3 | |

---

## Appendix A: Reproduction commands

All run from `research/champions/`, with the PS root as an argument:

```
node enumerate_legal.js /tmp/psrepo-delta/ps gen9championsvgc2026regmc .
node ps_checks.js /tmp/psrepo-delta/ps > ps_checks_output.txt
node diff_moves.js /tmp/psrepo-delta/ps > move_diffs.txt
node diff_nonstandard.js /tmp/psrepo-delta/ps > nonstandard_deltas.txt
python3 -I compare_official.py web/official/mc_pokemon.html legal_species.json legal_megas.json
```

`diff_moves.js` was written against the old path; it accepts any PS root.

## Appendix B: Source URLs (web facts)

- Official Champions news, mirrored in `web/official/` and listed in `list.json`: Regulation Set M-C notice (page 816), Regular Roster M-C (821), maintenance 2026-10-07 (844). Base URL: `https://champions-news.pokemon-home.com/en/page/<id>.html` [I for exact URL form; content V]. Eligible list: `https://web-view.app.pokemonchampions.jp/battle/pages/…/en/pokemon.html`.
- Bulbapedia: "Pokémon Champions", "Stat point", "Regulation Set M-A/M-B/M-C", "Regulation Sets in Pokémon Champions", "2026 Pokémon World Championships", and the new-ability pages (mirrored in `web/bulba_*.wiki`).
- Serebii Champions section: statusconditions, updatedattacks, newabilities, megaabilities, items, patch (mirrored in `web/serebii/`).
- GitHub: smogon/pokemon-showdown issue #12377; issue/PR search results (`web/github/`); commits API (`gh/`).
