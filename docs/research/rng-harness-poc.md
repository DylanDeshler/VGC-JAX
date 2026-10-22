# RNG oracle harness: proof of concept and RNG call-site inventory

*Pokémon Showdown at commit `d601443` (2026-10-21), format `gen9championsvgc2026regmc`, mod `champions`. Node v26.8.1 on an Apple M4 Max (14 cores). Code is in `research/poc/`; the PS checkout was not modified.*

This document de-risks section 6a of `jax-vgc-engine-scope.md`. It covers three things:
- whether every PS random draw can be recorded with its meaning, forced, and keyed semantically;
- whether state can be serialized, crafted and diffed;
- how fast the oracle runs.

All numbers come from the scripts in §1 and the JSON files in `research/poc/results/`.

---

## 0. Verdicts

| Question | Verdict | Evidence |
|---|---|---|
| Can every draw be intercepted? | **Yes.** | All draws go through `battle.prng`. A drop-in proxy passed as `new Battle({prng})` gave a byte-identical log in 300/300 battles. It recorded 100 draws/battle, and none of them were unclassified. |
| Can each draw be mapped to an EVENT_KIND? | **Yes, two ways.** | (a) Stack frame → TS call site via source maps, then a lookup in `rng_callsites.json`. (b) A stack-free classifier using wrappers on 8 core methods plus `battle.effect`. Over 30k draws the two agreed 100%. Actor/target extraction was checked against the log: 184/184 misses and 275/275 crits on the right mon. |
| Does stream replay (Mode 1) work? | **Yes.** | 300/300 battles were identical (log and state) when draws were answered from the recording with a different underlying seed. |
| Does context-keyed forcing (Mode 2) work? | **Yes, with the right key.** | Exact-log reproduction by key: naive `(turn, kind, actor, target, move, hit)` 45%; adding the sub-step (current queue action) 96%; adding an occurrence ordinal 100%. The engine-implementable key reproduced 300/300 state trajectories with the don't-care kinds left free (§6.2). |
| Do override policies (Mode 3) work? | **Yes.** | "Max roll / never crit / always hit" forced 16.5k draws over 300 battles. That produced 0 crits, and the only misses were semi-invulnerable targets (no draw involved). |
| Does serialize/deserialize round-trip mid-battle? | **Yes.** | 300/300 continuations matched the uninterrupted battle exactly. A round trip costs about 1 ms; a state is about 35 KB of JSON, including the log. |
| Can crafted states be built? | **Yes, with caveats.** | Live mutation through PS APIs and JSON editing gave identical states and identical next turns. Inconsistent states (for example a Sitrus Berry holder below 50%) are "repaired" by PS at the first `Update`. |
| Are legal-action masks available from the request? | **Mostly.** | 99,242 of 99,320 request-derived joint choices were accepted. All 78 rejections were hidden-information cases (`maybeTrapped`/`maybeDisabled`). 0 of 36,976 out-of-mask probes were accepted. |
| Throughput | **About 40–54 battles/s per thread.** | Random play, about 27 decisions per battle. Under heavy load from other jobs, 12 workers reached 193 battles/s (about 17M battles/day). The uncontended estimate is about 450 battles/s. Instrumentation costs +6% without stack capture and +25% with it. |

---

## 1. Code layout and how to run

```
research/poc/
  lib/ps.js            loads the pinned PS build (PS_ROOT env, default /tmp/psrepo/ps, fallback /tmp/psrepo-rng/ps)
  lib/teams.js         team sources: PS champions random-doubles (fixed up) + Smogon chaos-stats sampler; HarnessRNG
  lib/player.js        request JSON -> legal per-slot options + joint constraints; random legal chooser
  lib/runner.js        runDirect (Battle API, sync) / runStream (BattleStream, async) / drive(battle, script)
  lib/context.js       per-battle instance hooks: current action, last runEvent target/source per event id,
                       getRandomTarget/getRandomSwitchable args, fieldEvent/eachEvent ids, insertChoice; labels
  lib/callsite.js      structured V8 stack capture + dist->TS source-map lookup (cached)
  lib/classify.js      classifyByStack / classifyByMarkers -> {kind, actor, target, key, keyNaive}
  lib/rng.js           InstrumentedPRNG (record | replay | keyed | override), buildKeyedDict, keyFor
  lib/project.js       canonical state projection + field-level diff
  lib/experiment.js    deterministic battle specs (teams, seeds), one-call run helper
  tools/scan_callsites.js        greps PS sources for PRNG call sites (+ --check drift alarm)
  tools/inventory_annotations.js hand annotations (kind, distribution, context exprs, reachability)
  tools/build_inventory.js       merges -> rng_callsites.json (+ --table prints the markdown table)
  rng_callsites.json             the inventory (88 call sites)
  scripts/01_teams.js … 08_masks.js   experiments; each writes results/<name>.json
```

Run everything from `research/poc/`:

```
node tools/build_inventory.js           # regenerate rng_callsites.json (and print table)
node tools/scan_callsites.js --check    # fails if PS moved/added/removed a PRNG call site
node scripts/01_teams.js 300            # team sources + validation
node scripts/02_instrument.js 300 [random]   # transparency, classification, context checks, coverage
node scripts/03_replay.js 300           # Modes 1/2/3 + per-kind observability   (~2.5 min)
node scripts/04_serialize.js 100        # serialize/deserialize round trips + cost
node scripts/05_crafted.js              # crafted Trick Room / Tailwind / Sitrus state, two ways
node scripts/06_throughput.js 200       # single-thread throughput + instrumentation overhead
node scripts/07_workers.js 8 1,4,8,12,14 plain|oracle   # worker_threads scaling
node scripts/08_masks.js 60             # request-derived legal mask vs PS acceptance
```

The build needs `/tmp/psrepo/ps` (or `$PS_ROOT`) at `d601443`, built with `npm install && node build`. The chaos sampler reads `/tmp/stats_dl/gen9championsvgc2026regmcbo3-1760.json`, from https://www.smogon.com/stats/2026-09/chaos/; override the path with `CHAOS_STATS`. Mid-task, `/tmp` was wiped by a reboot, so I re-cloned PS at `d601443` into `/tmp/psrepo-rng/ps` and re-downloaded the stats. Another agent later recreated `/tmp/psrepo/ps` at the same commit; `diff -rq` showed the two `dist/` trees were identical.

---

## 2. RNG call-site inventory

### 2.1 Scope and method

`tools/scan_callsites.js` greps for `this.random(`, `randomChance(`, `sample(`, `shuffle(`, `speedSort(` and `prng.` across `sim/*.ts`, base `data/*.ts` and `data/mods/champions/*.ts`. It records the enclosing function or effect handler for each hit.

Reachability was computed against the Champions dex:
- 382 legal species and 510 learnable moves, from `TeamValidator.checkSpecies` / `checkCanLearn`;
- ability availability on legal species;
- `isNonstandard` for items;
- hand review for conditions and scripts.

Every grepped site has an annotation in `tools/inventory_annotations.js`. The build fails on unannotated or stale sites, and `--check` is the drift alarm required by §6a.

**Totals:** 88 call sites. 52 are reachable in Reg M-C, 2 only in rare configurations, and 34 unreachable.

| Why unreachable | Sites |
|---|---|
| Overridden by the Champions mod | `sim/battle-actions.ts` multihit/accuracy 929–997; base `par`, `slp`, `frz`, Healer and Dire Claw |
| Gen-gated | gen 2/3 Quick Claw; gen<5 multihit table |
| Mechanic absent | Dynamax / G-Max; triples targeting |
| Item unavailable | Loaded Dice (scripts.ts:441/450); Starf Berry |
| Move or ability not on any legal mon | Toxic Chain, Neutralizing Gas, Metronome, Assist, Magnitude, Present, Psywave, Conversion 2 |
| No such move | No legal move has a multihit range other than [2,5] (scripts.ts:447) |
| Rule not in format | Category Swap Mod |

Other observations from the scan:
- In this format, all randomness passes through the PRNG.
- The only `Math.random` use is seed generation when no seed is given.
- The only other nondeterminism is the `|t:|` timestamp lines.
- Teams with no `gender` set make PS draw gender from `battle.prng` at construction (`sim/pokemon.ts:361`). The team generators now always set gender.

### 2.2 EVENT_KIND taxonomy and keys

Every key has the form `turn | step | KIND | actor | target | extras`:
- `step`: the PS queue action running when the draw happens, e.g. `move:p2a:heatwave`, `residual`, `runSwitch:p1b`, or `commit` for draws at choice resolution.
- `actor` / `target`: slot labels, `p1a`, `p1b`, `p2a` or `p2b`.
- `extras`: kind-specific fields.

**How actor and target are recovered.** For core draws, the target is a local variable inside PS. `lib/context.js` recovers it from the most recent `runEvent` of a specific event id:

| Kind | Event used | Mapping |
|---|---|---|
| ACCURACY | `Accuracy` | target = defender, source = attacker |
| CRIT | `ModifyCritRatio` | target = attacker, source = defender |
| DAMAGE_ROLL | `WeatherModifyDamage` (via priorityEvent) | target = attacker, source = defender |
| SECONDARY | `ModifySecondaries` | target = defender, source = attacker |

Handler draws use `battle.event.target` / `.source` together with `battle.effect.id`.

| EVENT_KIND | PS call sites (reachable) | Outcome | Key extras | Per battle (chaos / random-doubles teams) | Changes the state trajectory when randomized alone (§6.3) |
|---|---|---|---|---|---|
| ACCURACY | battle-actions.ts:800, champions scripts.ts:507 (per-hit) | bool | move, hit | 19.8 / 22.8 | 64% of battles |
| CRIT | battle-actions.ts:1695 | bool | move, hit | 19.6 / 20.7 | 72% |
| DAMAGE_ROLL | battle.ts:2409 (from champions modifyDamage) | int 0–15 (0 = max) | move, hit | 19.6 / 20.8 | 100% |
| CONFUSION_SELFHIT_DAMAGE_ROLL | battle.ts:2409 via getConfusionDamage | int | – | 0.02 | – |
| SECONDARY | battle-actions.ts:1399 | int <100 | move, hit, **secondary index** | 8.2 / 9.4 | 69% |
| SELF_EFFECT_ROLL | battle-actions.ts:1381 | int <100 | move, hit | 1.4 / 1.8 | **0%** (self.chance is never set; always applies) |
| MULTIHIT_COUNT | champions scripts.ts:439 | int 2–5 | move | 0.03 | – |
| RANDOM_TARGET | side.ts:372 (via getRandomTarget), battle.ts:2523, conditions.ts:305 | index | move, phase (resolve/getTarget/other) | 7.0 / 7.5 | 4% |
| RANDOM_TARGET_NOMINAL | side.ts:372 for spread/side moves | index | move, phase | 12.6 / 11.2 | **0%** (only the `\|move\|` line's target) |
| SPEED_TIE_ACTION_ORDER | battle-queue.ts:426 | perm | tied set | 0.82 / 1.8 | 18% |
| SPEED_TIE_SWITCHIN_ORDER | battle-actions.ts:235 | perm | tied set | 0.35 / 0.78 | 2% |
| SPEED_TIE_FIELD_EVENT | battle.ts:512 (Residual, SwitchIn handler order) | perm | event, tied set | 3.4 / 2.3 | 0% in 275 battles |
| SPEED_TIE_EACH_EVENT | battle.ts:473 (mostly `Update`) | perm | event, tied set | 5.4 / 11.0 | 0% |
| SPEED_TIE_HANDLER_ORDER | battle.ts:801 (runEvent ties) | perm | event, tied set | 0.16 / 0.32 | 0% |
| SPEED_TIE_ABILITY | Magician (abilities.ts:2483) | perm | tied set | not observed | – |
| QUEUE_INSERT_TIE | battle-queue.ts:406 | int (queue position) | inserted action | 0.08 / 0.15 | 0% |
| PARALYSIS | champions conditions.ts:5 (1/8) | bool | – | 0.38 / 2.0 | – |
| SLEEP_DURATION | champions conditions.ts:23 (`sample([2,3,3])`) | index | – | 0.25 / 0.11 | – |
| FREEZE_THAW | champions conditions.ts:49 (1/4, 3-turn cap) | bool | – | 0.07 / 0.03 | – |
| CONFUSION_DURATION / CONFUSION_SELFHIT | conditions.ts:174 / 187 | int / bool | – | 0.04 / 0.05 | – |
| PARTIAL_TRAP_DURATION | conditions.ts:227 | int 5–6 | – | 0.03 / 0.08 | – |
| LOCKEDMOVE_DURATION | conditions.ts:265 | int 2–3 | – | not observed | – |
| PROTECT_STALL | conditions.ts:456 (1/3^n) | bool | – | 0.48 / 0.31 | – |
| ALLY_SWITCH_STALL | moves.ts:343 | bool | – | not observed | – |
| INFATUATION | moves.ts:745 | bool | – | not observed | – |
| FORCED_SWITCH_PICK | battle.ts:1586 (Roar, Whirlwind, Dragon Tail, Circle Throw) | index | side, move | 0.01 / 0.02 | – |
| ABILITY_ONHIT_PROC | Static, Flame Body, Poison Point, Cute Charm, Cursed Body, Effect Spore, Poison Touch | bool/int | effect, hit | 0.48 / 0.41 | – |
| ABILITY_RESIDUAL_PROC | Shed Skin, Healer (Champions 1/2), Harvest | bool | effect | – / 0.19 | – |
| ABILITY_TRACE_TARGET, _FOREWARN_PICK, _PICKUP_TARGET, _MOODY_STAT | abilities.ts | index | effect (+ord for Moody) | Trace 0.06 | – |
| PRIORITY_PROC / QUICK_CLAW | Quick Draw 3/10, Quick Claw 1/5 (drawn at choice commit) | bool | effect | not observed | – |
| ITEM_FOCUS_BAND | items.ts:2273 (drawn on every damage instance) | bool | hit | not observed | – |
| SECONDARY_STATUS_PICK | Tri Attack, Dire Claw (Champions) | index | move, hit | 0.09 / 0.01 | – |
| MOVE_FICKLE_BEAM, MOVE_SHELLSIDEARM_TIE, MOVE_ACUPRESSURE_STAT, MOVE_CALL_PICK (Sleep Talk) | moves.ts | bool/index | move | Fickle Beam 0.01 | – |
| GENDER | pokemon.ts:361 (construction) | index | team slot | 0 (gender always set) | – |

**Coverage.** 300 random battles hit 27 distinct sites with chaos-stats teams and 33 with PS random-doubles teams. Reachable sites never hit include Quick Claw/Draw, Focus Band, Effect Spore, Moody, Pickup, Ally Switch, Attract, Sleep Talk, Tri Attack and Lockedmove (`results/02_instrument*.json`, `reachableSitesNotObserved`). This is the concrete case for coverage-guided team generation and crafted scenarios: usage-weighted random play does not exercise a large share of the RNG surface.

### 2.3 PS RNG semantics the JAX engine must mirror (or deliberately ignore)

1. **Speed ties are re-rolled after every action.** In gen ≥ 8, PS re-sorts the remaining queue after each action (`battle.ts:2975–2983`), and each re-sort reshuffles the ties. The key is therefore (turn, step, tied set), not (turn, tied set): the same tie appeared 4 times in one turn.
2. **Draws happen even when the outcome cannot matter.** Examples:
   - an ACCURACY roll for 100%-accurate moves (`randomChance(100,100)`), the largest group;
   - a SECONDARY roll for 100%/`undefined` chances;
   - SELF_EFFECT_ROLL for Close Combat and Draco Meteor;
   - `sample` over 1 element;
   - `randomChance(1,1)` at critRatio ≥ 4;
   - RANDOM_TARGET for every spread move: `getTarget` falls through to `randomFoe()` for `allAdjacentFoes`, `allAdjacent` and `foeSide` targets. This is 12.6 draws per battle and changes only the `|move|…|target` log annotation.

   About 24% of all draws are degenerate (only one possible outcome), mostly 100%-accuracy rolls and random-target picks with one foe left.
3. **Some draws happen at choice-commit time,** before the turn's first action: Quick Claw, Quick Draw, and random targets for `randomNormal` moves (Outrage, Thrash). The key's `step` is then `commit`.
4. **Two secondaries per move.** Fire Fang, Ice Fang and Thunder Fang each have two, so the key needs the secondary's index.
5. **Per-hit accuracy for hits 2+** (Triple Axel, Population Bomb) is a separate ACCURACY draw keyed by `hit`. A miss on these silently ends the loop with no `|-miss|` line.
6. **Crit vetoes happen after the draw.** Shell Armor and Battle Armor veto through `CriticalHit` after the draw (seen with Mega Slowbro). A forced `crit=true` therefore does not guarantee a crit.
7. **Focus Band draws on every damage instance** to its holder, not only on lethal ones.

### 2.4 Full call-site table

Generated by `node tools/build_inventory.js --table`. The machine-readable version, with the context expressions for actor and target, is `research/poc/rng_callsites.json`.

| # | call site | enclosing | method(args) | decides | EVENT_KIND | out | key extras | reach |
|---|---|---|---|---|---|---|---|---|
| 1 | `sim/battle.ts:461` | speedSort | `shuffle(list, sorted, sorted + nextIndexes....)` | Fisher-Yates order of a run of equal-priority elements inside speedSort | SPEED_TIE (primitive) | perm | caller,tied | yes |
| 2 | `sim/battle.ts:473` | eachEvent | `speedSort(actives, (a, b) => b.speed - a.speed)` | order in which equal-speed active mons run an eachEvent (e.g. Update after Weather) | SPEED_TIE_EACH_EVENT | perm | event,tied | yes |
| 3 | `sim/battle.ts:512` | fieldEvent | `speedSort(handlers)` | order of tied handlers in fieldEvent (Residual, SwitchIn) | SPEED_TIE_FIELD_EVENT | perm | event,tied | yes |
| 4 | `sim/battle.ts:801` | runEvent | `speedSort(handlers)` | order of tied handlers inside runEvent | SPEED_TIE_HANDLER_ORDER | perm | event,tied | yes |
| 5 | `sim/battle.ts:1586` | getRandomSwitchable | `sample(canSwitchIn)` | which benched mon is dragged in (Roar, Whirlwind, Dragon Tail, Circle Throw, Red Card) | FORCED_SWITCH_PICK | index | slot | yes |
| 6 | `sim/battle.ts:1652` | endTurn | `speedSort(dynamaxEnding)` | order of Dynamax-ending messages | SPEED_TIE_DYNAMAX_END | perm | - | no |
| 7 | `sim/battle.ts:1814` | endTurn | `randomChance(60, 256)` | gen 2 Quick Claw | QUICK_CLAW_GEN2 | bool | - | no |
| 8 | `sim/battle.ts:1815` | endTurn | `randomChance(1, 5)` | gen 3 Quick Claw | QUICK_CLAW_GEN3 | bool | - | no |
| 9 | `sim/battle.ts:2409` | randomizer | `random(16)` | damage random factor (100 - r)% | DAMAGE_ROLL | int | move,hit | yes |
| 10 | `sim/battle.ts:2523` | getRandomTarget | `sample(adjacentAllies)` | random ally for an adjacentAlly move used without a target | RANDOM_TARGET | index | move | cond |
| 11 | `sim/battle.ts:2532` | getRandomTarget | `sample(adjacentFoes)` | random adjacent foe (triples) | RANDOM_TARGET | index | move | no |
| 12 | `sim/battle-actions.ts:235` | runSwitch | `speedSort(allActive)` | speedOrder snapshot for simultaneous switch-ins (used to order SwitchIn handlers) | SPEED_TIE_SWITCHIN_ORDER | perm | tied | yes |
| 13 | `sim/battle-actions.ts:800` | hitStepAccuracy | `randomChance(accuracy, 100)` | move hits this target | ACCURACY | bool | move,hit | yes |
| 14 | `sim/battle-actions.ts:929` | hitStepMoveHitLoop | `sample([2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3...)` | 2-5 hit count | MULTIHIT_COUNT | int | move | no |
| 15 | `sim/battle-actions.ts:931` | hitStepMoveHitLoop | `random(2)` | Loaded Dice 4-5 | MULTIHIT_COUNT | int | move | no |
| 16 | `sim/battle-actions.ts:934` | hitStepMoveHitLoop | `sample([2, 2, 2, 3, 3, 3, 4, 5])` | gen<5 2-5 hits | MULTIHIT_COUNT | int | move | no |
| 17 | `sim/battle-actions.ts:937` | hitStepMoveHitLoop | `random(targetHits[0], targetHits[1] + 1)` | generic [m,n] hit count | MULTIHIT_COUNT | int | move | no |
| 18 | `sim/battle-actions.ts:940` | hitStepMoveHitLoop | `random(7)` | Loaded Dice Population Bomb | MULTIHIT_COUNT | int | move | no |
| 19 | `sim/battle-actions.ts:997` | hitStepMoveHitLoop | `randomChance(accuracy, 100)` | per-hit accuracy (Triple Axel / Population Bomb) | ACCURACY | bool | move,hit | no |
| 20 | `sim/battle-actions.ts:1381` | selfDrops | `random(100)` | chance of a move's self stat change (move.self.boosts) | SELF_EFFECT_ROLL | int | move,hit | yes |
| 21 | `sim/battle-actions.ts:1399` | secondaries | `random(100)` | whether secondary effect i applies to this target | SECONDARY | int | move,hit,ord | yes |
| 22 | `sim/battle-actions.ts:1695` | getDamage | `randomChance(1, critMult[critRatio])` | critical hit on this target | CRIT | bool | move,hit | yes |
| 23 | `sim/battle-queue.ts:406` | insertChoice | `random(firstIndex, lastIndex + 1)` | insertion point of a mid-turn action among equal-priority queued actions | QUEUE_INSERT_TIE | int | inserted,tiedWith | yes |
| 24 | `sim/battle-queue.ts:426` | sort | `speedSort(this.list)` | order of speed-tied actions in the turn queue (the classic speed tie) | SPEED_TIE_ACTION_ORDER | perm | tied | yes |
| 25 | `sim/pokemon.ts:361` | constructor | `sample(['M', 'F'])` | gender for a set with no gender on a mixed-gender species | GENDER | index | side,teamIndex | yes |
| 26 | `sim/side.ts:372` | randomFoe | `sample(actives)` | random foe when a move has no valid chosen target (randomNormal moves, retarget after the chosen target fainted, Dragon Darts) | RANDOM_TARGET | index | move | yes |
| 27 | `data/abilities.ts:788` | cursedbody.onDamagingHit | `randomChance(3, 10)` | Cursed Body disables the move | ABILITY_ONHIT_PROC | bool | effect,hit | yes |
| 28 | `data/abilities.ts:801` | cutecharm.onDamagingHit | `randomChance(3, 10)` | Cute Charm infatuates attacker | ABILITY_ONHIT_PROC | bool | effect,hit | yes |
| 29 | `data/abilities.ts:1164` | effectspore.onDamagingHit | `random(100)` | Effect Spore: slp (<11) / par (<21) / psn (<30) / none | ABILITY_ONHIT_PROC | int | effect,hit | yes |
| 30 | `data/abilities.ts:1320` | flamebody.onDamagingHit | `randomChance(3, 10)` | Flame Body burns attacker | ABILITY_ONHIT_PROC | bool | effect,hit | yes |
| 31 | `data/abilities.ts:1526` | forewarn.onStart | `sample(warnMoves)` | which tied highest-BP foe move Forewarn reveals | ABILITY_FOREWARN_PICK | index | effect | yes |
| 32 | `data/abilities.ts:1805` | harvest.onResidual | `randomChance(1, 2)` | Harvest restores berry (always in sun) | ABILITY_RESIDUAL_PROC | bool | effect | yes |
| 33 | `data/abilities.ts:1823` | healer.onResidual | `randomChance(3, 10)` | Healer (base 30%) | ABILITY_RESIDUAL_PROC | bool | effect | no |
| 34 | `data/abilities.ts:2483` | magician.onAfterMoveSecondarySelf | `speedSort(hitTargets)` | Magician steal order among speed-tied hit targets | SPEED_TIE_ABILITY | perm | effect,tied | yes |
| 35 | `data/abilities.ts:2715` | moody.onResidual | `sample(stats)` | Moody stat to raise | ABILITY_MOODY_STAT | index | effect,ord | yes |
| 36 | `data/abilities.ts:2726` | moody.onResidual | `sample(stats)` | Moody stat to lower | ABILITY_MOODY_STAT | index | effect,ord | yes |
| 37 | `data/abilities.ts:2952` | neutralizinggas.onEnd | `speedSort(sortedActive)` | Neutralizing Gas end order | SPEED_TIE_ABILITY | perm | tied | no |
| 38 | `data/abilities.ts:3272` | pickup.onResidual | `sample(pickupTargets)` | whose consumed item Pickup takes | ABILITY_PICKUP_TARGET | index | effect | yes |
| 39 | `data/abilities.ts:3348` | poisonpoint.onDamagingHit | `randomChance(3, 10)` | Poison Point poisons attacker | ABILITY_ONHIT_PROC | bool | effect,hit | yes |
| 40 | `data/abilities.ts:3376` | poisontouch.onSourceDamagingHit | `randomChance(3, 10)` | Poison Touch poisons target | ABILITY_ONHIT_PROC | bool | effect,hit | yes |
| 41 | `data/abilities.ts:3739` | quickdraw.onFractionalPriority | `randomChance(3, 10)` | Quick Draw moves first in bracket | PRIORITY_PROC | bool | effect | yes |
| 42 | `data/abilities.ts:4192` | shedskin.onResidual | `randomChance(33, 100)` | Shed Skin cures status | ABILITY_RESIDUAL_PROC | bool | effect | yes |
| 43 | `data/abilities.ts:4540` | static.onDamagingHit | `randomChance(3, 10)` | Static paralyzes attacker | ABILITY_ONHIT_PROC | bool | effect,hit | yes |
| 44 | `data/abilities.ts:5097` | toxicchain.onSourceDamagingHit | `randomChance(3, 10)` | Toxic Chain badly poisons | ABILITY_ONHIT_PROC | bool | effect,hit | no |
| 45 | `data/abilities.ts:5145` | trace.onUpdate | `sample(possibleTargets)` | which adjacent foe Trace copies | ABILITY_TRACE_TARGET | index | effect | yes |
| 46 | `data/items.ts:2273` | focusband.onDamage | `randomChance(1, 10)` | Focus Band survives at 1 HP | ITEM_FOCUS_BAND | bool | hit | yes |
| 47 | `data/items.ts:5017` | quickclaw.onFractionalPriority | `randomChance(1, 5)` | Quick Claw moves first in bracket | QUICK_CLAW | bool | effect | yes |
| 48 | `data/items.ts:6036` | starfberry.onEat | `sample(stats)` | Starf Berry stat | ITEM_STARF_STAT | index | - | no |
| 49 | `data/moves.ts:155` | acupressure.onHit | `sample(stats)` | stat boosted by Acupressure | MOVE_ACUPRESSURE_STAT | index | move | yes |
| 50 | `data/moves.ts:343` | allyswitch.condition.onRestart | `randomChance(1, counter)` | consecutive Ally Switch succeeds | ALLY_SWITCH_STALL | bool | - | yes |
| 51 | `data/moves.ts:632` | assist.onHit | `sample(moves)` | Assist move | MOVE_CALL_PICK | index | - | no |
| 52 | `data/moves.ts:745` | attract.condition.onBeforeMove | `randomChance(1, 2)` | infatuated mon is immobilized | INFATUATION | bool | - | yes |
| 53 | `data/moves.ts:2839` | conversion2.onHit | `sample(possibleTypes)` | Conversion 2 type | MOVE_CONVERSION2_TYPE | index | - | no |
| 54 | `data/moves.ts:3641` | direclaw.secondary.onHit | `sample(['psn', 'par', 'slp'])` | Dire Claw status (base) | SECONDARY_STATUS_PICK | index | move | no |
| 55 | `data/moves.ts:5231` | ficklebeam.onBasePower | `randomChance(3, 10)` | Fickle Beam doubles power | MOVE_FICKLE_BEAM | bool | move | yes |
| 56 | `data/moves.ts:6728` | gmaxbefuddle.self.onHit | `sample(['slp', 'par', 'psn'])` | G-Max Befuddle status | GMAX | index | - | no |
| 57 | `data/moves.ts:7110` | gmaxreplenish.self.onHit | `randomChance(1, 2)` | G-Max Replenish | GMAX | bool | - | no |
| 58 | `data/moves.ts:7202` | gmaxsnooze.onHit | `randomChance(1, 2)` | G-Max Snooze | GMAX | bool | - | no |
| 59 | `data/moves.ts:7207` | gmaxsnooze.onAfterSubDamage | `randomChance(1, 2)` | G-Max Snooze | GMAX | bool | - | no |
| 60 | `data/moves.ts:7288` | gmaxstunshock.self.onHit | `random(2)` | G-Max Stun Shock | GMAX | int | - | no |
| 61 | `data/moves.ts:10903` | magnitude.onModifyMove | `random(100)` | Magnitude power | MOVE_MAGNITUDE | int | - | no |
| 62 | `data/moves.ts:11803` | metronome.onHit | `sample(moves)` | Metronome move | MOVE_CALL_PICK | index | - | no |
| 63 | `data/moves.ts:13926` | present.onModifyMove | `random(10)` | Present power/heal | MOVE_PRESENT | int | - | no |
| 64 | `data/moves.ts:14294` | psywave.damageCallback | `random(50, 151)` | Psywave damage | MOVE_PSYWAVE | int | - | no |
| 65 | `data/moves.ts:16232` | shellsidearm.onModifyMove | `randomChance(1, 2)` | Shell Side Arm category when physical == special | MOVE_SHELLSIDEARM_TIE | bool | move | yes |
| 66 | `data/moves.ts:16887` | sleeptalk.onHit | `sample(moves)` | move called by Sleep Talk | MOVE_CALL_PICK | index | move | yes |
| 67 | `data/moves.ts:19853` | triattack.secondary.onHit | `sample(['brn', 'par', 'frz'])` | Tri Attack status (after the 20% SECONDARY passed) | SECONDARY_STATUS_PICK | index | move,hit | yes |
| 68 | `data/conditions.ts:41` | par.onBeforeMove | `randomChance(1, 4)` | full paralysis (base 25%) | PARALYSIS | bool | - | no |
| 69 | `data/conditions.ts:59` | slp.onStart | `random(2, 5)` | sleep turns (base 1-3) | SLEEP_DURATION | int | - | no |
| 70 | `data/conditions.ts:99` | frz.onBeforeMove | `randomChance(1, 5)` | thaw (base 20%) | FREEZE_THAW | bool | - | no |
| 71 | `data/conditions.ts:174` | confusion.onStart | `random(min, 6)` | confusion turns | CONFUSION_DURATION | int | - | yes |
| 72 | `data/conditions.ts:187` | confusion.onBeforeMove | `randomChance(33, 100)` | confused mon hits itself | CONFUSION_SELFHIT | bool | - | yes |
| 73 | `data/conditions.ts:227` | partiallytrapped.durationCallback | `random(5, 7)` | binding turns (4-5 effective) | PARTIAL_TRAP_DURATION | int | - | yes |
| 74 | `data/conditions.ts:265` | lockedmove.onStart | `random(2, 4)` | Outrage/Thrash/Petal Dance/Raging Fury length | LOCKEDMOVE_DURATION | int | - | yes |
| 75 | `data/conditions.ts:305` | twoturnmove.onStart | `sample(attacker.foes(true))` | target for a called two-turn move whose target fainted | RANDOM_TARGET | index | move | cond |
| 76 | `data/conditions.ts:456` | stall.onStallMove | `randomChance(1, counter)` | consecutive Protect/Detect/Endure/... succeeds | PROTECT_STALL | bool | - | yes |
| 77 | `data/rulesets.ts:2623` | categoryswapmod.onModifyMove | `randomChance(1, 2)` | Category Swap Mod tie | RULE_CATEGORY_SWAP | bool | - | no |
| 78 | `data/mods/champions/abilities.ts:43` | healer.onResidual | `randomChance(1, 2)` | Healer cures adjacent ally (Champions 50%) | ABILITY_RESIDUAL_PROC | bool | effect | yes |
| 79 | `data/mods/champions/conditions.ts:5` | par.onBeforeMove | `randomChance(1, 8)` | full paralysis (Champions 12.5%) | PARALYSIS | bool | - | yes |
| 80 | `data/mods/champions/conditions.ts:23` | slp.onStart | `sample([2, 3, 3])` | sleep length (Champions: 2 w.p. 1/3, else 3) | SLEEP_DURATION | index | - | yes |
| 81 | `data/mods/champions/conditions.ts:49` | frz.onBeforeMove | `randomChance(1, 4)` | thaw before the 3-turn cap (Champions 25%) | FREEZE_THAW | bool | - | yes |
| 82 | `data/mods/champions/moves.ts:223` | direclaw.secondary.onHit | `sample(['psn', 'par', 'slp'])` | Dire Claw status | SECONDARY_STATUS_PICK | index | move,hit | yes |
| 83 | `data/mods/champions/scripts.ts:439` | actions.hitStepMoveHitLoop | `sample([2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3...)` | 2-5 hit count | MULTIHIT_COUNT | int | move | yes |
| 84 | `data/mods/champions/scripts.ts:441` | actions.hitStepMoveHitLoop | `random(2)` | Loaded Dice reroll to 4-5 | MULTIHIT_COUNT | int | move | no |
| 85 | `data/mods/champions/scripts.ts:444` | actions.hitStepMoveHitLoop | `sample([2, 2, 2, 3, 3, 3, 4, 5])` | gen<5 table | MULTIHIT_COUNT | int | move | no |
| 86 | `data/mods/champions/scripts.ts:447` | actions.hitStepMoveHitLoop | `random(targetHits[0], targetHits[1] + 1)` | generic [m,n] range | MULTIHIT_COUNT | int | move | no |
| 87 | `data/mods/champions/scripts.ts:450` | actions.hitStepMoveHitLoop | `random(7)` | Loaded Dice Population Bomb | MULTIHIT_COUNT | int | move | no |
| 88 | `data/mods/champions/scripts.ts:507` | actions.hitStepMoveHitLoop | `randomChance(accuracy, 100)` | per-hit accuracy for hits 2..n (Triple Axel, Population Bomb) | ACCURACY | bool | move,hit | yes |

---

## 3. Team sources

| Source | Valid for `gen9championsvgc2026regmc` | Cost | Diversity |
|---|---|---|---|
| `Teams.getGenerator('gen9championsrandomdoublesbattle')` raw | 10% | 2.6 ms/team | – |
| same + fix-ups (level 50, drop duplicate items) | **100%** (300/300) | 1.9 ms/team incl. validation | 284 species |
| chaos-stats sampler (`lib/teams.js`) | **100% first try** (300/300) | 1.5 ms/team incl. validation | 86 species, 27 Mega stones |

The raw generator fails on Item Clause (269 of 300 teams have duplicate items) and on levels 50–60. Two quick checks after the fix-up showed no further issues; Champions SP (stat points, 66 total) are emitted as `evs`, which the validator accepts.

How the chaos sampler builds a team:
- Species are drawn by usage weight, with Species Clause and Item Clause.
- Mega entries such as `Salamence-Mega` become the base species plus the stone. Their ability distribution comes from the base forme's entry, because the Mega entry lists the Mega ability.
- Spreads use the stats file's `Nature:hp/atk/def/spa/spd/spe` SP format directly.
- Moves are the top-20 by usage, sampled without replacement.
- Gender is always set explicitly.

Recommendation: mix both sources. Chaos sets match the training distribution; random-doubles sets give coverage, hitting 33 sites against 27.

## 4. Running battles

- **Direct API vs BattleStream.** Direct is `new Battle({formatid, seed, prng, strictChoices: true})`, then `setPlayer` twice, then loop over `side.activeRequest`. The stream route sends `>start`/`>player` and parses `|request|` JSON.
  - Measured back to back on the same pre-generated teams: direct 53.7 battles/s vs stream 45.6 (~18% faster) on a quieter machine. Under load: 40 vs 40.
  - Direct is synchronous, gives access to the `Battle` object for instrumentation, and avoids async overhead. Use it for the oracle; keep the stream path only to check that the request JSON a client sees is identical.
- **Requests handled.** The runner handles team preview (`team 1234`: bring 4, the first two lead), `forceSwitch`, `wait`, fainted or Commanding slots (`pass`), Mega (`move N T mega`, at most one per side) and trapping.
- **`[Unavailable choice]`.** With `strictChoices`, PS rejects choices that depend on hidden information: `maybeTrapped` (Shadow Tag, Arena Trap) and `maybeDisabled` (Imprison and similar). It then pushes a refreshed request. The runner retries; this happens once or twice per 100 battles.
- **Random-play statistics.** 10.4 turns, 26.7 decisions and about 100 draws per battle, with usage-stats teams.

## 5. PRNG instrumentation

**Proxy (`lib/rng.js`).** The proxy implements `random`, `randomChance`, `sample`, `shuffle`, `getSeed`, `setSeed`, `clone` and `startingSeed`, and is passed as `options.prng`.
- It records at the **decision level**: int for `random`, bool for `randomChance`, index for `sample`, permutation for `shuffle`.
- It consumes the underlying PRNG exactly as PS does (same Fisher-Yates loop). Record mode is therefore transparent: the log was byte-identical in 300/300 battles.
- `Battle.randomChance` checks `forceRandomChance` first. That is null outside debug mode, so all draws reach the proxy.

**Context (`lib/context.js`).** Instance-level wrappers are installed on one battle object; no prototype or source changes. They wrap:
- `runAction` (current action and the action index within the turn);
- `runEvent` (per-event-id last target and source);
- `fieldEvent` / `eachEvent` (event id);
- `getTarget` / `queue.resolveAction` (random-target phase);
- `getRandomTarget` / `getRandomSwitchable` (arguments);
- `queue.insertChoice` (the inserted action).

At each draw the snapshot also records `battle.turn`, `battle.effect`, `battle.event` (id, target, source), `activeMove` (id, hit), `activePokemon` and `activeTarget`. The context object is non-enumerable because `State.serializeBattle` throws on class instances.

**Call site (`lib/callsite.js`).**
- `Error.captureStackTrace` with a `prepareStackTrace` that returns structured CallSites, so no string formatting.
- It skips PRNG plumbing frames and maps dist positions to TS lines through PS's own `.js.map` files, caching per position.
- It returns the site plus 2 caller frames, needed for `speedSort` → caller, `randomizer` → `modifyDamage` vs `getConfusionDamage`, and `randomFoe` → `getRandomTarget` → caller.
- Cost: 7.7 µs per capture in a microbenchmark, against 15.4 µs for `new Error().stack`. In real battles stacks are deeper and the measured overhead is about 59 µs per draw.

**Results (300 battles each, chaos and random-doubles teams):**

| Check | Result |
|---|---|
| Unmapped draws | 0 |
| Stack classifier vs marker classifier | 0 disagreements on kind or key |
| ACCURACY=false draws whose `\|-miss\|` line names the extracted actor and target | 184/184 and 259/259 (per-hit accuracy for hits 2+ excluded: no miss line) |
| CRIT=true draws followed by `\|-crit\|` on the extracted target | 275/275 and 306/307. The exception was Mega Slowbro's Shell Armor vetoing after the draw. |

**What did not map cleanly at first** (all fixed, but each is a class of maintenance issue):
1. `secondary.onHit` handlers run with `battle.effect` set to the anonymous secondary object, which has no id. The classifier falls back to the active move.
2. `partiallytrapped.durationCallback` runs outside any event, so `battle.effect` is the move. The marker classifier keys it by its `random(5,7)` signature; the stack classifier gets it from the frame.
3. Random-target draws need the `getRandomTarget` arguments: `activeMove` is not yet set during `getTarget` and at commit time.
4. `speedSort` shuffles need the caller frame, or element-type and event-context heuristics in the marker version.

**Recommendation:** run the marker classifier for production fuzzing; it is nearly free. Run the stack classifier in CI or periodically as the drift check: any `UNMAPPED` result or classifier disagreement raises the alarm.

## 6. Forced replay

All forced runs use a different underlying seed (`1,2,3,4`), so anything not forced diverges. Choices are replayed from the recording. "State" means the canonical projection (§7) at every decision point plus the final state.

### 6.1 Mode 1: stream replay

Draws are answered in recorded order, with method and arguments checked. **300/300 battles were identical in both log and state, with 0 desyncs.**

### 6.2 Mode 2: context-keyed forcing

The dictionary is built from the recording. Each draw is answered by key lookup; a miss goes to an unrelated PRNG.

| Key variant | Exact log | Same state trajectory | Battles with dict-key conflicts | First-divergence kinds (log) |
|---|---|---|---|---|
| naive `turn\|kind\|actor\|target\|move,hit` | 136/300 (45%) | 287/300 (96%) | 235 | RANDOM_TARGET_NOMINAL 143, EACH_EVENT 14, SECONDARY 3, ACTION_ORDER 3, RANDOM_TARGET 1 |
| + step (current queue action) | 288/300 (96%) | 297/300 (99%) | 65 | RANDOM_TARGET_NOMINAL 9, SECONDARY 3 |
| + step + occurrence ordinal | 300/300 | 300/300 | 0 | – |
| **engine** (step; ordinal only for SECONDARY/Moody; don't-care kinds randomized) | 147/300 (expected: randomized don't-care draws change log annotations) | **300/300** | 1 in a forced kind (two switch-in sorts in one step); 274 in don't-care kinds | first differing draw is always a don't-care kind: RANDOM_TARGET_NOMINAL 84, FIELD_EVENT 34, SELF_EFFECT_ROLL 17, EACH_EVENT 10, QUEUE_INSERT_TIE 8 |

Why the failures happen:
- **Naive keys are not unique.** Each tie is re-rolled after each action. A spread move consumes one nominal random-target draw per use, and some are drawn twice in one action (Expanding Force in Psychic Terrain). `eachEvent('Update')` shuffles the same tied pair about 3 times per action. Two-secondary moves give two draws with the same key.
- **Adding `step` fixes ordering.** The remaining collisions are don't-care draws (nominal targets, repeated `Update` ties) and the secondary index, which is engine-observable.
- **The "engine" variant** uses only information a JAX engine has: who is acting with which move, the hit index, the secondary index and the tied set. It leaves the §6.3 don't-care kinds unforced and still reproduced **300/300 state trajectories**. Only 1 battle had a key collision in a forced kind (two switch-in sorts in one step), and it did not change the outcome. Exact logs differ in about half the battles, but only in annotations: nominal targets and the order of simultaneous messages.

### 6.3 Observability of each kind

Everything was forced by key except kind K, which was answered by an unrelated PRNG. The column records how often the state trajectory changed.

| Kind | Battles containing it | State changed | Log changed |
|---|---|---|---|
| DAMAGE_ROLL | 300 | 100% | 100% |
| CRIT | 300 | 72% | 91% |
| SECONDARY | 297 | 69% | 71% |
| ACCURACY | 300 | 64% | 64% |
| SPEED_TIE_ACTION_ORDER | 51 | 17.6% | 45% |
| RANDOM_TARGET | 249 | 4.0% | 3.6% |
| SPEED_TIE_SWITCHIN_ORDER | 56 | 1.8% | 5% |
| RANDOM_TARGET_NOMINAL | 242 | **0%** | 60% |
| SPEED_TIE_FIELD_EVENT | 275 | **0%** | 2% |
| SPEED_TIE_EACH_EVENT | 63 | **0%** | 3% |
| SPEED_TIE_HANDLER_ORDER | 5 | **0%** | 0% |
| QUEUE_INSERT_TIE | 24 | **0%** | 4% |
| SELF_EFFECT_ROLL | 189 | **0%** | 0% |

**Implication for the JAX engine.** About 23 of 100 draws per battle can be ignored: nominal targets, handler, field and each-event ties, queue-insert ties and self-effect rolls. The JAX diff should compare state, not raw logs. Log-only differences such as the nominal target and the order of simultaneous messages should be normalised.

The 0% kinds are 0% *under random play on these teams*. Field-event ties do matter in principle, for example two speed-tied Leftovers holders when one is about to faint. Keep them keyed (cheap) and flag them as "low priority" rather than deleting them.

### 6.4 Mode 3: override policies

300 battles of fresh random play per policy:

| Policy | Forced draws | Verification |
|---|---|---|
| max roll, never crit, always hit | 16,471 | 0 `-crit`. The 5 `-miss` lines were all semi-invulnerable targets (Phantom Force, Dive), where PS draws nothing. |
| min roll, always crit, never full paralysis | 8,841 | 4,396 `-crit` lines, 0 `cant\|par` |
| no speed-tie randomness (identity permutations) | 2,953 | all battles completed |

Policies are plain functions `(draw) => outcome | undefined` over the classified draw. Use them for deterministic unit scenarios, such as damage-range endpoints, and for distribution tests where only one kind is left free.

## 7. Serialization and state projection

**Round trip.** For 100 battles, 3 random decision points each:
1. replay to decision k;
2. `State.serializeBattle` → `JSON.stringify` → `State.deserializeBattle`;
3. continue with the remaining recorded choices.

300/300 battles reproduced the uninterrupted log exactly, and the state projections were equal straight after the round trip. They were also exact with a fresh `InstrumentedPRNG` seeded from `battle.prng.getSeed()`; deserialization replaces `battle.prng`, so the proxy must be re-attached.

| Cost (mean / p95) | |
|---|---|
| serialize | 0.25 / 0.37 ms |
| stringify | 0.16 / 0.20 ms |
| deserialize (parse + rebuild) | 0.55 / 0.74 ms |
| JSON size | 35 KB (most of it the battle log) |
| state projection (`lib/project.js`) | 0.065 ms |

**Fields to diff.** These come from the serialized Battle/Side/Pokemon/Field, reduced by `lib/project.js` into deterministic, label-referenced JSON:
- **battle:** `turn`, `ended`, `winner`;
- **field:** `weather` + `weatherState.duration`, `terrain` + `terrainState.duration`, `pseudoWeather{id: {duration, …}}` (Trick Room, Gravity, …);
- **side:** `pokemonLeft`, `sideConditions{id: {duration, layers, …}}` (Tailwind, screens, …), `slotConditions[slot]` (Wish and similar), active slot → species;
- **pokemon:** current species/forme (Mega), `types`, `hp`/`maxhp`, `fainted`, `status` + `statusState` (sleep `time`/`startTime`, toxic `stage`, freeze `time`), `boosts`, `storedStats`, `volatiles{id: {duration, counter, time, …}}`, `item` + `itemState`, `lastItem`, `ability` + `abilityState`, `moves[{id, pp, maxpp, disabled}]`, `canMegaEvo`, `lastMove`, `trapped`, `weightkg`.

Effect states keep only primitive fields. Bookkeeping such as `effectOrder`, `source`, `sourceSlot` and `sourceEffect` is dropped, and references become `p1a` or `p1:speciesid`.

**Not diffed:**
- the log;
- request bookkeeping (`side.choice`, `activeRequest`);
- `prng`;
- `effectOrder`, which is a PS-internal tie-break counter. It matters only for SPEED_TIE_HANDLER_ORDER-style ordering, which §6.3 shows to be state-irrelevant.

The JAX side needs a per-effect whitelist mapping, e.g. PS `volatiles.confusion.time` ↔ JAX `confusion_turns`. Generate that table from the projection keys seen across the fuzz corpus.

## 8. Throughput

The machine was heavily contended during all measurements. Another job used about 12 cores, and the load average was 80–120. Treat these numbers as lower bounds. The earliest, quieter run gave **53.7 battles/s direct and 45.6 battles/s stream** on one thread.

Single thread, random legal play, teams pre-generated (`results/06_throughput.json`):

| Config | battles/s | decisions/s |
|---|---|---|
| direct, uninstrumented (gen5 seed / sodium seed) | 39.9 / 40.7 | 1,052 / 1,042 |
| BattleStream, uninstrumented | 40.4 | 1,062 |
| record decisions only (no context) | 38.6 | 1,018 |
| record + context + marker classifier | 38.6 | 1,016 |
| record + context + stack capture/classifier | 35.0 | 922 |
| record + stack + projection per decision | 33.3 | 878 |

Interleaved battle-by-battle so that load drift affects all configs equally, relative cost against uninstrumented:

| Config | Relative cost | Overhead per draw |
|---|---|---|
| decision recording | 1.055 | 13 µs |
| + context + markers | 1.064 | 15 µs |
| + stack capture | 1.247 | 59 µs |
| + both classifiers + projection | 1.317 | – |

Team generation adds 1.3–1.9 ms per team, about 10% of a battle. Cache team pools if that matters. Sodium (ChaCha20) and Gen5 seeds cost the same at this scale.

**worker_threads, 8 s per run** (`results/07_workers_plain.json`, contended):

| Workers | 1 | 4 | 8 | 12 | 14 |
|---|---|---|---|---|---|
| battles/s | 39.4 | 108 | 161 | **193** | 201 |
| decisions/s | 1,033 | 2,866 | 4,265 | 5,123 | 5,345 |

The full oracle config (record + context + markers + projection per decision) ran at 163 battles/s on 12 workers.

**Extrapolation (uncontended).** The M4 Max has 10 performance and 4 efficiency cores. At about 50 battles/s per P-core and 85% scaling, plus about 15 per E-core, that gives **~450 battles/s ≈ 39M battles/day** uninstrumented, or about 35M/day for the oracle config. Even contended, 12 workers already give about 17M battles/day. This confirms the scoping doc's "tens per second per core, millions per day". PS is not the bottleneck for differential testing. The JAX replay, diffing and shrinking will dominate.

## 9. Crafted states

Target scenario (`scripts/05_crafted.js`): Trick Room with 3 turns left, Tailwind on p2 with 3 turns left, and p1a at 30% HP holding a Sitrus Berry, all at the turn-1 decision point.

- **A: live mutation with PS APIs.** Finish team preview, then call `field.addPseudoWeather('trickroom', source)`, `side.addSideCondition('tailwind', source)`, set `duration`s, then `mon.setItem('sitrusberry')` and `mon.sethp(0.3*maxhp)`, and finally `battle.makeRequest('move')` to rebuild requests that embed HP and items.
- **B: edit serialized JSON.** Add `pseudoWeather.trickroom` and `sideConditions.tailwind` as `{id, effectOrder, source: '[Pokemon:p2a]', sourceSlot, duration}`, set `item`, `itemState` and `hp` on the p1a pokemon entry, delete `activeRequest` so it is recomputed, then `deserializeBattle`.
- **Result.** The projections of A and B were identical before the turn. After playing the same choices they were again identical, and the turn logs matched.
  - Mechanics were correct. Under Trick Room the +2 moves went slowest first (Indeedee 105 before Arcanine 156), then the +0 moves (Basculegion 260 before Charizard 288, with Tailwind-doubled speeds).
  - Trick Room and Tailwind each went from 3 to 2.
  - The Sitrus Berry was eaten at the very first `Update`, before any move.

**What's fragile:**
1. **Not a fixpoint.** PS assumes states are closed under `Update` and similar handlers. A berry holder below threshold, a Booster Energy holder in sun, or a terrain-seed holder on a terrain is "fixed" at the first event. A JAX engine started from the same raw state must apply the same repair, or the crafted state must be normalised first (run an `eachEvent('Update')` and diff the result).
2. **APIs require a `source`.** `addPseudoWeather` and `addSideCondition` throw without one (duration callbacks, Persistent). They also emit log lines and fire `FieldStart`/`SideStart`, which can trigger reactions such as Room Service and Seed items. JSON editing avoids those side effects, but you must keep `effectOrder` monotonic and the refs valid (`[Pokemon:p2a]` = side + position letter).
3. **Stale requests.** Requests embed HP, items, moves and trap/disable flags, so rebuild them after mutation (`makeRequest`, or drop `activeRequest` before deserializing).
4. **Derived fields.** `storedStats`, `speed`, `species` vs `baseSpecies` (Mega), `abilityState`, `canMegaEvo`, `side.pokemonLeft`, `fainted` and `faintQueue` are derived or cached. Changing species, ability or HP to 0 by hand needs the matching API (`formeChange`, `setAbility`, `faint`), or an inconsistent state results.
5. **Durations count the current turn.** PS duration semantics: Tailwind 4 includes the turn it is set. Map these carefully onto the JAX "turns remaining" convention.

## 10. Legal-action masks vs request JSON

`scripts/08_masks.js` tests the request-derived option set from `lib/player.js`, which stands in for the JAX mask.
- **Method.** At every decision of 60 battles (1,457 decision points), the battle is copied by State round trip. Every joint choice the parser enumerates is submitted to `side.choose()`. Probes outside the enumerated set are submitted too: all move/target combinations and switches 1–6.
- **Results.**
  - 99,242 of 99,320 enumerated joint choices were accepted. The 78 rejections were all hidden information: 6 `[Unavailable choice]` plus 67 follow-on trapped and 5 disabled rejections after the request was refreshed.
  - 0 of 36,976 probes were accepted, so the request-derived mask is never too strict.
  - The first version of the parser had a real masking bug, which this harness caught. It allowed `pass` on a forced switch while a bench mon was available. PS allows a pass only when `bench < forced slots`; exactly `min(forced, bench)` slots must switch.
- **What the request JSON does not give the JAX mask diff:**
  - the joint constraints: distinct switch-ins, one Mega per side, the forced-switch count;
  - target locations: only target *types* are given. For doubles they derive as `normal`/`any` → {1, 2, ally}, `adjacentFoe` → {1, 2}, `adjacentAlly` → {ally}, `adjacentAllyOrSelf` → {self, ally};
  - **hidden-information flags** (`maybeTrapped`, `maybeDisabled`, `maybeLocked`). The fully informed JAX engine will legitimately be stricter. The diff should treat "JAX forbids, PS shows `maybe*`" as a match only when PS rejects the choice with `[Unavailable choice]`.

## 11. Problems and gotchas found

1. **Environment.** `/tmp` was wiped mid-task by a reboot. Pin PS in a durable location rather than `/tmp`, and record the commit in every result file (lib/ps.js prefers `$PS_ROOT`).
2. **Draws whose outcome can't matter** (§2.3.2, §6.3). The JAX engine must not try to model them; the harness must not demand they be keyed.
3. **Re-rolled speed ties after every action** (gen ≥ 8): key ties by step.
4. **Hidden-info request flags.** These need special handling in the mask diff, and in the random player's retries.
5. **Unset gender becomes a PRNG draw** at construction. Always set gender.
6. **Projection non-determinism trap.** A retry after `[Unavailable choice]` re-presents the same request. Calling the decision hook twice shifted the projection sequence; fixed by keying on request identity. The same care is needed in the JAX diff driver.
7. **Effect-state bookkeeping.** `effectOrder`, refs and similar must be excluded from diffs.
8. **Load sensitivity of throughput numbers.** Rerun `06`/`07` on a quiet machine before quoting capacity externally.

## 12. Recommended harness architecture

```
            ┌────────────── worker_threads pool (1 per core, Node) ───────────────┐
 teams  ──► │ spec(i): chaos/random-doubles teams + seed + scenario (fresh | crafted JSON)
            │ Battle (direct API, strictChoices) + InstrumentedPRNG(record, markers)  │
            │   └─ context hooks (lib/context.js)                                     │
            │ policy: random | heuristic | current policy  (own RNG, never battle.prng)│
            │ at each decision: request JSON → legal joint-action set                 │
            │                   state projection (lib/project.js)                     │
            │ output record: {spec, choices, draws[kind,key,out], projections, masks}│
            └───────────────────────────────┬────────────────────────────────────────┘
                                            ▼  (NDJSON / Arrow shards)
            JAX side (batched): rebuild initial state → replay choices with outcomes
            forced by `engine` key (step, kind, actor, target, move, hit, secondary idx;
            don't-care kinds unforced) → diff projection + mask at every decision
                                            ▼
            on mismatch: re-run PS from serialized state at the failing decision (1 ms),
            shrink (truncate turns / drop bench / swap moves), save fixture + fired sites
```

- **Recording.** Use the decision-level draw record with the marker classifier in production (≤ 7% overhead). Turn on stack capture in CI, or on 1% of battles, as the call-site drift alarm. Also run `tools/scan_callsites.js --check` on every PS re-pin.
- **Forcing contract with the JAX engine.** The key is `(turn, step, kind, actor, target, move, hit, secondaryIdx)` plus the tied set for permutations. Steps are labelled by the acting mon and action type: `move:p2a:<move>`, `residual`, `runSwitch:p1b`, `commit`, …
  - Ship the don't-care list (§6.3) as part of the contract.
  - Any PS draw with no JAX counterpart that is not on the list fails the test ("draw consumed with no matching engine event").
- **Diff state projections, not logs.** Use logs only to explain divergences.
- **Shrinking and crafted scenarios** rest on State round trips, which are exact and take about 1 ms. Normalise crafted states with one `Update` pass before handing them to both engines.
- **Coverage.** Collect `sitesHit` per battle (already in `02`) and per-handler counts (hook `runEvent`). Steer team generation towards unhit sites: random play left 20–26 reachable sites unexercised in 300 battles.

## 13. Files

- `research/poc/rng_callsites.json`: the inventory (88 sites, kinds, distributions, context expressions, reachability)
- `research/poc/lib/*.js`, `research/poc/tools/*.js`, `research/poc/scripts/*.js`: the harness
- `research/poc/results/*.json`: raw outputs for every number above
