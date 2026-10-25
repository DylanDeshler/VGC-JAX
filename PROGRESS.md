# PROGRESS
## Milestone / task
M0 — Foundations (O, E) — done except M0.6 launchd agents and M0.7 pool-B sampler (infrastructure; not blocking)
- M0.1 repo layout — done (research/dex/raw *copied*, not moved, to data/raw/ so research/ stays reproducible)
- M0.2 PS submodule @ d601443, `make oracle` — done
- M0.3 POC oracle port — done (POC scripts 01–05, 08 reproduce research verdicts byte-for-byte except timings; check-callsites passes)
- M0.4 `oracle gen` + DrawKey + tie records + serve RPC — core done: byte-identical shards for 1 vs 12 workers; 4k battles
  (chaos + randdoubles, both classifiers) with 0 unmapped draws, 0 classifier disagreements, 0 forced-key collisions;
  RPC round trips (run/snapshot/from_snapshot/validate_team) tested at every decision point. Pending: `--legal`, `--stage`
  (needs M0.5), coverage counters (M3.8).
- M0.5 datagen — done: export_dex.js (lossless export, Champions PP asserted vs PS for every move, resolvePriority
  order metadata incl. order-only residual callbacks), legal_pool.js, census.js, dex_stages.py, catalogue stage column
  regenerated. Self-checks at d601443 reproduce exactly: v0 377/110/142 + 220 CUSTOM, v1 454/172/160 + 376,
  v2 510/215/166 + 483, core 197 moves / 6 abilities / 20 species. 17 datagen tests (self-tests 2 and 4).
- M0.6 Python env / Makefile — partial (pins + Makefile + lint + merge.sh; launchd agents todo)
- M0.7 team banks — partial: frozen real-sheet corpus done (3,211 sheets, all TeamValidator-valid, sha256 abacca4c…); pool-B sampler + validator bridge todo
- M0.8 test-support RPCs — done except `resolve_priority` (may slip to M3.1 per plan): fixedpoint, stats, sort_actions,
  legal (+ `gen --legal on|off|<fraction>`; parity with request-derived sets on 732 side-decisions), replay forced by
  DrawKey + tie records (600/600 battles reproduce under a different PS seed), minimal craft, damage (equal to the 4
  verified research examples), trace_events. `bench/oracle_legal_cost.js`: 12.4 ms per decision.


M1 — Core math and data (E) — done
- M1.1 codegen → vgcjax/data/{ids.py, tables.npz}: stage-independent ID spaces (NS 383, NM 512, NA 217, NI 167, NC 104,
  NEFFECT 1009 < 4096), per-stage `supported_*` masks, DATA_VERSION embedded and checked; deterministic; self-test 1
  (1,000 random legal sets: stats, maxpp, types, weight) passes
- M1.2 fixedpoint: 10^6 cases equal PS (domain value·m4096 < 2^31, documented)
- M1.3 stats: 1,000 random sets incl. stages, paralysis, Trick Room action speed equal PS
- M1.4 typechart: every attacking type × pool typing, plus the 10 immunity pseudo-keys, equal PS
- M1.5 damage core: 120k random core-subset cases + 3 verified research examples × 16 rolls equal PS
- M1.6 rng: contract §2 hash (checked vs a Python reference), χ² on distributions incl. 10^7 draws, vmap determinism,
  forced/draw-log/degenerate semantics, tie records (perm rank/unrank, SELECT winner)
- T0 25 s, T1 46 s (local); pre-commit hook installed

M2 — Harness core + core state machine (O, E) — done; **G-core run complete, pending human decision (ADR-102)**
- G-core run `gcore-2026-10-21-a` (50,000 fresh-seed core battles, random policy, --legal on 100%):
  CBR 100%, TMR 100%, MPR 100%, DPR 100%, error rate 0 — gate report:
  reports/conformance/gcore-2026-10-21-a/gate_report.md. Verdict FAIL only on coverage: RANDOM_TARGET 0 and
  SPEED_TIE_FIELD_EVENT 0 (structurally near-impossible / impossible in core). ADR-102 proposes counting RANDOM_TARGET
  from crafted Struggle scenarios (400/400 clean, 360 hits) and waiving FIELD_EVENT for G-core only.
- M2.7 from_ps_snapshot done (10k snapshots project equal; continued replays clean)
- M2.8 done: contract files (harness/contract/*, gates.yaml, waivers.json), gates.py (contract sha256 check),
  streaming corpus reader, coverage counts, parallel replay (~85 battles/s on 12 workers); on-device batched diff
  deferred to M9 (host replay is fast enough for now)
- done: M2.1 state/init/preview, M2.2 queue (re-key, SPEED_TIE_SELECT), M2.3 blocks A/B/C + tail + faint/win + end_turn,
  M2.4 switching (batched runSwitch, SPEED_TIE_SWITCHIN, dragIn), M2.5 core move pipeline (targeting incl. Dragon Darts,
  accuracy/OHKO/multiaccuracy, crit/roll, multihit, spread, drain/recoil/heal, boosts, secondaries, pivots, drags,
  Memento, Struggle), M2.6 masks, M2.9 core teams (oracle/lib/stageteams.js), M2.10 canary (tests/harness/test_canary.py)
- (superseded) M2.0/M2.8 harness — host-side single-battle replay (harness/replay_one.py) + parallel CLI
  (`python -m harness.cli replay <dir>`); still todo: on-device batched diff, triage/report/gates.py, protected
  harness/contract/* files and harness/gates.yaml (human must create docs/decisions/contract-hashes.md)
- todo: M2.7 from_ps_snapshot
- compile time of `step` (core, CPU): ~10 s after converting heavy unrolled loops to fori

M5 — Env API (R) — in progress (infrastructure lane)
- done: actions/canonical preview, to_ps_choice round trip, observations (leak + symmetry tests), api.step/StepOut,
  TeamBank + pair sampling (data/teams/bank_core.npz, 2000 validated core teams), AutoResetEnv + EpisodeStats
- CPU throughput (core, unoptimised): ~3,000 env-steps/s at batch 256-1024 (bench/throughput.py)
- todo: PublicEvents (obs.events is zeros), observe_privileged/hidden_targets, M5.5 PPO smoke trainer, M5.6 export

Comparison-quality work (2026-10-22)
- extended non-gating projection: found + fixed 2 hidden engine bugs; ADR-103 for promotion + SELECT-tie gap
- heuristic policy (M3.8b); 2000 heuristic core battles clean
- don't-care re-validation (07 §4.3): all kinds 0 state impact in 4000 battles except QUEUE_INSERT_TIE (1/424),
  evidence appended to ADR-101
- M3.9 shrinker (harness/shrink.py; oracle RPCs edit_snapshot, shrink_candidates, replay project_after): greedy
  PS+engine reduction, kept iff the signature persists; canary fixtures shrink 32 -> 3-4 moves, empty bench, full HP,
  ~1.5 s each. Fixtures carry `entities` (triage hint); STATE signatures erase mon indices. Condition removal waits
  for v0 conditions. `harness.cli replay --fixtures N` auto-emits shrunk fixtures per signature.
- 07 §11 two-sample distribution test (harness/distribution.py, tests/distribution, slow/T3): engine natural RNG vs
  PS on 12 fresh core states x 4000 runs: 26 per-mon HP tests, Bonferroni min p 0.46; the crit canary is rejected
  (17/26 tests, p ~ 0). Chain checks (Protect, sleep, freeze, confusion) wait for v0 content.
- PublicEvents (03 §4.7) in the engine (vgcjax/battle/events.py) + obs.events; checked against PS's protocol
  (oracle/lib/protocol.js reduction, `gen --events`): 3,000/3,000 core battles identical except 1 ADR-103 switch-order
  tie. Found on the way: PS boost message order = data key order (export now records boost_key_order);
  Struggle's -activate; [still] blanks the move-line target. EngineConfig.events=False compiles them out.
- bench/hlo_kernels.py (09 §1.1a): **119,789 kernel-like ops per env step without events, 138,172 with** (budget
  10,000, enforced at G-v0). Pre-existing: loop nesting (NB 8 x hit loop 10 x target fori 3 x secondaries 6);
  09 J5 asks for vmap over the target axis. Options in ADR-104 (needs a decision).

M3 content (2026-10-23)
- Conditions framework (vgcjax/effects/conditions.py: storage, add/remove/restart, singleEvent, field schema +
  presence bits for the projection, effectOrder), fieldEvent walk for Residual and SwitchIn
  (effects/fieldevent.py: getKey duration, speedOrder-adjusted SwitchIn speeds, effectOrder, FIELD_EVENT tie
  records), BeforeMove / ModifyMove / DamagingHit / AfterMoveSecondary event sites, residual moved out of the
  micro-steps (once per step call), battle.speedOrder state.
- Statuses (Champions): brn/par/psn/tox/slp/frz incl. par 1/8, sleep [2,3,3], freeze cap 3, thaw paths, toxic
  switch-in reset, Corrosion, Early Bird; move status + secondary status. Found: exported ignoreImmunity is already
  resolved (Thunder Wave false), clampIntRange floors (toxic), checkFainted keeps the status state.
- `impl` stage (tools/datagen/impl_stage.py): core + 36 status moves. 2,500 impl battles 100% (CBR/TMR/MPR,
  events). docs/catalogue.csv: 260 v0 entities impl (tools/catalogue_mark.py).
- Flinch + confusion (22 more moves; oracle fix CONFUSION_SELFHIT_DAMAGE_ROLL subject), 1,000 impl battles 100%.
- Protect family: Protect, Detect, stall counter, Wide Guard, Feint (breaksProtect), Fake Out ([C] DisableMove),
  Try / PrepareHit preamble, multi-target TryHit (left-to-right), BreakProtect, side-move path (tryMoveHit),
  move onHit / onHitSide, addSideCondition. Targeted corpora (oracle `--teams stagereq:impl:<req>`): 400 battles
  each Protect+Fake Out / Wide Guard+Feint / Detect: 100%. Quick Guard implemented but unverified (no learner in
  the 20-species impl pool).
- Speed control + screens: Tailwind (ModifySpe chain), Trick Room (addPseudoWeather, restart toggle), Reflect /
  Light Screen (ModifyDamage chain), side/field start-end events; contract MUL tie records reproduced via exact
  mulhook_ord counting (every PS MUL runEvent: getStat('spe') sites, damage calc, accuracy) + rekey at the end of a
  completed runAction (PS skips it after early returns). Targeted corpora 100%.
- Weather (effects/weather.py): Sun / Rain / Sand / Snow from moves (setWeather, rocks, FieldStart/Residual/End,
  eachEvent('Weather') with SPEED_TIE_WEATHER_EACH records, sandstorm chip + typed immunity, WeatherModifyDamage,
  sand SpD / snow Def x1.5, frz immunity in Sun, Cloud Nine suppression), ActiveMove type / basePower / accuracy
  after ModifyType / ModifyMove (Exec.mtype / bp / acc: Weather Ball, Thunder, Hurricane, Blizzard). Targeted
  corpora (200 each: Sun+Weather Ball, Rain+Thunder, Sand, Snow+Blizzard) 100% incl. events.
- Terrain (effects/terrain.py): Electric / Grassy / Psychic / Misty from moves (setTerrain, Terrain Extender,
  BasePower x5325/4096 and halvings, Grassy heal residual, SetStatus / TryAddVolatile blocks, Psychic Terrain
  TryHit), full isGrounded. fieldEvent fix: customHolder rows (field / side effects listed per active mon) are
  callback-only and never tick durations (PS :507-508). Electric / Grassy corpora 100%; Psychic / Misty
  Terrain, Hurricane and Quick Guard have no learner in the 20-species impl pool (implemented, unverified) —
  widening the pool needs abilities (next).
- codegen: an ability / item without onSwitchIn gets its onStart as the SwitchIn handler (PS getCallback).
- Abilities batch 1 (effects/abilities.py): Drought / Drizzle / Sand Stream / Snow Warning, Electric / Grassy /
  Psychic / Misty Surge, Cloud Nine (abilityState.ending), Intimidate, Pressure (DeductPP), Unnerve, Frisk;
  abilityState projection (declared fields), ability End at live switch-out and at faint, '-ability' / '-item'
  events. impl species = core + every v0 species with an implemented ability (90). 300 random impl battles and
  targeted corpora (each ability, Psychic / Misty Terrain, Hurricane, Quick Guard) 100%.
- MUL-hook framework (effects/modifiers.py): BasePower / ModifyAtk / SpA / Def / SpD / STAB / Damage / Accuracy
  as pure registered mods over a static candidate list (findEventHandlers prefixes, sides, field), comparePriority
  sort, SPEED_TIE_HANDLER_MUL records + drawn fold order, chain vs replace handlers, final modify. Screens, terrain
  BasePower and sand/snow defence migrated onto it. Found: getDamage's attack event is always ModifyAtk/SpA by
  category on the user (Body Press / Foul Play only change the stat read, PS :1757-1762).
- Abilities batch 2 (effects/ability_mods.py, 26): Technician, Tough Claws, Iron Fist, Sharpness, Mega Launcher,
  Strong Jaw, Reckless, Punk Rock, Sand Force, Huge / Pure Power, Hustle, Guts (+ no burn halving), Blaze /
  Torrent / Overgrow / Swarm, Thick Fat, Solar Power, Multiscale, Filter, Solid Rock, Friend Guard, Sniper,
  Adaptability, Compound Eyes. impl pool 203 species. 300 random + 26 targeted + regression corpora 100%.
- Abilities batch 3 (effects/ability_blocks.py, 21, undispatched masked passes): TryBoost (Clear Body, White Smoke,
  Hyper Cutter, Keen Eye (+ignoreEvasion), Big Pecks, Flower Veil, Intimidate blockers Inner Focus / Own Tempo /
  Oblivious / Scrappy, Guard Dog's inverted boost + DragOut), SetStatus (Limber, Insomnia, Vital Spirit, Immunity,
  Water Veil, Water Bubble, Purifying Salt, Leaf Guard, Sweet / Flower Veil; speed-ordered with each other, before
  the terrains), TryAddVolatile (Inner Focus, Own Tempo), Immunity (Magma Armor), Own Tempo's Hit message, Scrappy
  ignoreImmunity. impl pool 254 species. Water Veil unverified (no placeable species).
- Abilities batch 4 (effects/ability_reactions.py, 29, undispatched): TryHit immunities (Volt / Water Absorb, Earth
  Eater, Sap Sipper, Motor Drive, Flash Fire (+volatile, mods, accuracy = true), Soundproof, Bulletproof,
  Telepathy, Good as Gold), DamagingHit (Rough Skin first; Static, Flame Body, Poison Point, Effect Spore, Spicy
  Spray, Stamina, Weak Armor, Gooey, Justified, Rattled, the attacker's Poison Touch; ABILITY_ONHIT_PROC draws),
  AfterEachBoost (Defiant, Competitive), AfterBoost (Rattled), Speed Boost, [C] Regenerator / Natural Cure
  (SwitchOut), Anger Point (-setboost), Steadfast. impl pool 294 species. Spicy Spray / Water Veil unverified
  (no placeable species). Found: has_volatile was not vectorised over refs (latent); field handlers run before an
  ability holder whose cached speed is <= 0 (Trick Room negative speeds); Protect's '-activate' vs Dragon Darts
  depends on getSmartTargets.
- Items batch 1 (effects/items.py): the 18 type-boost items (one table-driven family, modifiers.mod_family,
  ADR-106), Life Orb (ModifyDamage + AfterMoveSecondarySelf recoil), Expert Belt, Leftovers, Rocky Helmet,
  Focus Sash (Damage event + useItem, '-enditem' before '-damage'), weather rocks / Light Clay / Terrain Extender.
  useItem resets itemState.effectOrder; Struggle is '???' for the modifier events. 300 random + 10 targeted +
  76 regression corpora 100% (except the ADR-105 cases).
- Kernel count 191.6k after batches 3-4 (167.5k events off; was 136k). Ablations: no single culprit — cost grows
  ~0.6-1.2k ops per implemented entity with masked per-entity code. ADR-106 (proposed): table-driven handler
  families (O(1) per family). Reaction boosts / on-hit statuses already batched into one call per target set.
- Found via the wider pool: Gigaton Hammer's cantusetwice disable; Dragon Darts — any step failure and Protect's
  TryHit clear smartTarget, a single remaining target takes every hit, spreadHit is fixed before the steps.
- Regression suite regenerated with the current reducer (/tmp/wx/mkreg.sh pattern: same seeds, current stage).
  Empty zstd frames are skipped by the trace reader.
- ADR-105 (proposed, needs human approval): Population Bomb's fractional accuracy numerator (67.5) is truncated by
  the harness (67) while the engine logs the equivalent integer bound (68) -> 1 DRAW_ARGS bucket in 300 battles.
- fieldEvent tie fix: a tie group is detected at its first *live* member (a fainted holder sorted first hid it).
- Replays use a persistent XLA cache (.jax_cache, keyed by HLO): repeated corpora skip recompilation.
- harness/debug.py: per-decision PS vs engine projection / events side by side for one battle.
- Kernel budget (impl stage, bench/hlo_kernels.py): 82.6k after Protect, 102.7k after weather/terrain, 315k after
  ability batches 1-2, now 135.8k. Cause found by ablation: a lax.switch under vmap runs every branch on the
  whole state, so handlers dispatched inside other dispatches (Weather event inside the weather residual inside
  the fieldEvent switch; ability SwitchIn branches; ability End per faint slot) multiply. Fixed by running fixed-
  data handler families undispatched as masked passes (weather.each_weather / residual_each, abilities.switch_in_vec
  / end_vec) and compacting the modifier fold. Rule going forward: no new dispatch inside a dispatched handler.
- Kernel-like ops/step (impl, before weather): 76,818 (15,231 core-only). Planned: a hook-machinery optimisation pass once more
  of v0 lands (centralised in effects/hooks.py, fieldevent.py, conditions.py).

## Last T2/T3 run
local conformance runs (not formal gate runs), PS d601443, contract_version 1:
| corpus | battles | CBR | TMR | MPR | DPR | errors |
|---|---|---|---|---|---|---|
| core seed 1 (random) | 500 | 100% | 100% | 100% (no legal recorded) | 100% | 0 |
| core seed 101 (random, --legal on) | 2000 | 100% | 100% | 100% | 100% | 0 |
| core seed 102 (random-weighted, --legal on) | 2000 | 100% | 100% | 100% | 100% | 0 |

## Top 5 buckets
none yet (no engine)

## Watch items
- [CONFIRMED, ADR-103] SPEED_TIE_SELECT ties whose members include inactive/fainted actors are not recorded (contract §4 filter); seen in
  ~0.8–2% of battles (`final.diag.unrecordedSelectTies`). If these move state (e.g. instaswitch order replacing two
  fainted mons), it will surface as STATE buckets and needs a contract ADR.
- G-core coverage requires ≥ 50 SPEED_TIE_FIELD_EVENT hits, but the core stage has no residual/SwitchIn handlers, so
  that tie kind cannot occur (0 in 4.5k core battles). Needs a waiver in docs/coverage/waivers.json (protected → human).
- PS draws Population Bomb's per-hit accuracy with a float numerator under accuracy/evasion stages; the engine logs
  ceil(); DRAW_ARGS would differ if it ever occurs (not seen yet). May need a normalisation ADR.
- SPEED_TIE_HANDLER_MUL ordinal (`mulhook_ord`) counts every MUL-hook runEvent call; the engine will have to count
  identically once hooks exist (M3.1). Brittle; watch.
- Kernel count 12x over the 09 §1.1a budget (see above); CPU throughput 2,930 env-steps/s at B=256 without events,
  ~1,650 with (noisy: another user process was loading the machine during the event benchmarks).
- Trace size ≈ 95 KB/battle uncompressed, ~7 KB zstd (requests dominate); design estimate was 35 KB.

- ADR-101 (oracle answers the don't-care QUEUE_INSERT_TIE canonically) — needs human review; see docs/decisions.

## Blocked (docs/blocked/)
none

## Pending escalations
- (2026-10-24) ADR-106 proposed: kernel count grows linearly with content (~192k now, budget 10k); table-driven
  handler families recommended; the ADR-104 budget needs revisiting with the human.
- (2026-10-24) ADR-105 proposed: canonicalise fractional randomChance numerators to ceil in draw-arg comparison.
- (2026-10-23) contract_version 2 implemented (ADR-101 b + ADR-103). ADR-104 option 4 (micro-steps) implemented:
  ≈15.4k kernel-like ops per decision (from 119.8k); new phase CONT. G-core re-run under v2 on the micro-step engine:
  PASS (50,000 battles, 100%, contract verified). L40S: ~0.94M decision-steps/s at B=65k (events on). M3 content started (see below).
- (2026-10-23) G-core ACKNOWLEDGED by the human; ADR-102 and ADR-103 ACCEPTED. G-core re-evaluated: PASS
  M3 content unblocked. contract-hashes.md written on the human's instruction; ADR-101 option (b) accepted
  (record QUEUE_INSERT_TIE; with ADR-103 in contract_version 2).
- ADR-104: option 3 (target-axis vectorisation) done: 119,789 -> 74,684 kernels/step (events off). Waiting for the
  human's L40S run of bench/gpu_check.sh before deciding option 4 (micro-step rewrite).

## Contract hashes (must match docs/decisions/contract-hashes.md)
not yet created (harness/contract/* is written in M2.8)

## Next 3 tasks
1. M3 content in 05 §4 order: core volatiles (confusion, flinch, partial trap, lockedmove, mustrecharge,
   twoturnmove, choicelock), Protect family + Fake Out, field (weather / terrain), speed control, switching...
   Previously listed infrastructure: M3.8 coverage-steering deficit scores + crafted templates,
   kernel-count reduction (09 §1.1a: target-axis vmap J5, hit-loop
   restructuring), observe_privileged, M5.5 PPO smoke trainer
2. After acknowledgement: M3.2 handler classification, then M3.3-M3.7 content (conditions, statuses, Protect, Mega,
   mid-turn switch family) using effects/hooks.py
3. MUL-hook tie records (hooks.py TODO) with the first MUL content
