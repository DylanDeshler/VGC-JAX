# Contract: DrawKey, draw kinds, tie records, don't-care list (NORMATIVE; protected by ADR-029)

`contract_version: 1`. Any change to this file bumps the version, needs an ADR and human approval, and invalidates traces, fixtures and the oracle cache per ADR-030.

This file is the **only** definition of the forcing and draw-comparison contract between the oracle (PS) and the engine. 03 §5, 04 §8 and 07 §4 reference it and do not restate it. The kind inventory is derived from `research/poc/rng_callsites.json` (88 PS call sites at `d601443`; 52 reachable).

---

## 1. Concepts
- **Draw:** a random decision PS makes through `battle.prng` (`random`, `randomChance`, `sample`), outside speed-sort shuffles.
- **Tie record:** a semantic replacement for a `speedSort` shuffle. Shuffle draws are **never** compared as draws (§4).
- **Forced kind:** compared and forced. **Don't-care kind:** dropped by the harness on the PS side; the engine never draws it. **Degenerate draw:** only one possible outcome; dropped on both sides.

## 2. Hash and packing (engine side)
- `w0, w1 = jax.random.bits(fold_in(key, step_counter), (2,), uint32)`.
- **Packed words:**
  - `c0 = turn(11) | step_tag(3) | kind(8) | actor(4) | target(4)` (30 bits);
  - `c1 = hit(4) | ord(4) | sub(8) | subject(12) | action_ord(2)` (30 bits);
  - `c2 = step_actor(4) | step_subject(12)` (16 bits).
- `hash32 = fmix32( fmix32( fmix32(w0 ^ c0) ^ w1 ^ (c1 * 0x85EBCA6B) ) ^ (c2 * 0xC2B2AE35) )`, where `fmix32` is the MurmurHash3 finaliser. All uint32.
- `−1` fields are encoded as all-ones in their bit width.

## 3. DrawKey fields

| Field | Width | Meaning |
|---|---|---|
| `kind` | 8 | DrawKind (§5 table) |
| `turn` | 11 | `battle.turn` at draw time |
| `step_tag` | 3 | the running queue action's class (§3.1) |
| `step_actor` | 4 | side·4 + b of the running action's pokemon; −1 for COMMIT and RESIDUAL |
| `step_subject` | 12 | the running action's move id (EffectId) or action-kind code |
| `actor` | 4 | side·4 + b (per kind, §5) |
| `target` | 4 | side·4 + b, or −1 |
| `subject` | 12 | EffectId: the move for move-driven kinds, the handler's effect for handler-driven kinds |
| `hit` | 4 | 1-based hit index; 0 if n/a |
| `ord` | 4 | secondary index / occurrence ordinal / tie-group index / RANDOM_TARGET phase |
| `sub` | 8 | tie ordinal (§4); 0 otherwise |
| `action_ord` | 2 | `step_actor`'s move-action ordinal this turn (0, then 1 for an Instruct-created action) |

Mons are identified by **side·4 + brought index**, not by slot, so benched or fainted sources (e.g. a Future Sight attacker) are representable. The oracle maps PS Pokemon objects to (side, brought index) through the team-preview order.

### 3.1 `step_tag` mapping (PS `action.choice` → tag)
| PS action | step_tag |
|---|---|
| (choice resolution: `commitChoices`, `resolveAction` at commit, `insertChoice`/`changeAction` resolve) | COMMIT (0) |
| `switch`, `instaswitch` | A_SWITCH (1) |
| `megaEvo` | A_MEGA (2) |
| `beforeTurn`, `beforeTurnMove`, `revivalblessing`, `priorityChargeMove`, `start`, `team` | A_OTHER (3) |
| `move` (any order) | B_MOVE (4) |
| `residual` | RESIDUAL (5) |
| `runSwitch` (incl. start) | SWITCHIN (6) |

Inline switch-ins (drags inside a move's tail, Red Card) **inherit** the enclosing action's tag. Draws inside `resolveAction` triggered mid-action (Encore `changeAction`, Instruct) use COMMIT, with `step_actor`/`step_subject` of the *enclosing* action.

## 4. Tie records
- **Raw shuffle draws are excluded.** Shuffle draws inside `speedSort` (inventory kinds SPEED_TIE_ACTION_ORDER, SPEED_TIE_SWITCHIN_ORDER, SPEED_TIE_FIELD_EVENT, SPEED_TIE_EACH_EVENT, SPEED_TIE_HANDLER_ORDER, SPEED_TIE_ABILITY) are dropped from draw comparison.
- **The oracle emits tie records instead:** `{kind, key: (turn, step_tag, sub, ord), tied: [members], value}`.
- **Comparison.** Records are compared **1:1 by key**, with equal `tied` sets, and forced by `value`.

| Tie kind | Emitted when | `sub` | `ord` | Members (`tied`) | `value` |
|---|---|---|---|---|---|
| `SPEED_TIE_SELECT` | a selectable action (switch, instaswitch, megaEvo, beforeTurnMove, priorityChargeMove, revivalblessing, move) is executed and ≥ 2 queue entries tied it on the full comparePriority key **whose pokemon is active and not fainted** | `select_ord`: executed selectable actions this turn (early-returning actions don't advance it) | 0 | side·4+b of the tied entries' pokemon, sorted | winner (side·4+b) |
| `SPEED_TIE_SWITCHIN` | every `runSwitch` sort (batched, single, drag, start) over **all slot occupants incl. fainted**, by `pokemon.speed`; one record per tie group of size ≥ 2 | `runswitch_ord`: runSwitch executions this turn | group index within that sort | members sorted by slot (p1a, p1b, p2a, p2b) | permutation index (lexicographic rank over members) |
| `SPEED_TIE_FIELD_EVENT` | every `fieldEvent` (Residual, SwitchIn) sort; one record per tie group of size ≥ 2, **omitting handlers that will be skipped** (fainted holder unless slot condition) | `fieldevent_ord`: fieldEvent calls this turn | group index | members ordered field < side0 < side1 < p1a < p1b < p2a < p2b, then effect id | permutation index |
| `SPEED_TIE_WEATHER_EACH` | `eachEvent('Weather')` (weather chip) ties | `eachevent_ord`: eachEvent('Weather') calls this turn | group index | slot order | permutation index |
| `SPEED_TIE_HANDLER_MUL` | a `runEvent` handler-sort tie group of size ≥ 2 on a **MUL-class hook** (BasePower, ModifyAtk/Def/SpA/SpD/Spe, ModifyDamage, ModifyAccuracy, ModifyCritRatio, ModifySTAB) | `mulhook_ord`: MUL-hook runEvent calls this turn | group index | (holder side·4+b or field/side code, effect id) sorted | permutation index |
| `SPEED_TIE_ABILITY` | Magician's sort of speed-tied hit targets (`data/abilities.ts:2483`) | `select_ord` of the running move | group index | side·4+b sorted | permutation index |

**Ordinal storage.** The ordinals live in `TurnScratch` (03 §4.6), persist across SWITCH steps within a turn, and reset in `end_turn`.

**Engine side:**
- at each such sort, compute the same tie groups with the same member order;
- natural mode draws a permutation by hashing each member with the record key;
- forced mode applies `value`.

## 5. Draw kinds (forced, compared)

| DrawKind | Inventory kind(s) / PS site(s) | Value domain | actor | target | subject | hit | ord |
|---|---|---|---|---|---|---|---|
| ACCURACY | ACCURACY: `battle-actions.ts:800`, `ch/scripts.ts:507` (per-hit) | bool | attacker | defender | move | hit | 0 |
| CRIT | CRIT: `battle-actions.ts:1695` | bool | attacker | defender | move | hit | 0 |
| DAMAGE_ROLL | DAMAGE_ROLL: `battle.ts:2409` via modifyDamage | int 0–15 | attacker | defender | move | hit | 0 |
| CONFUSION_SELFHIT_DAMAGE_ROLL | `battle.ts:2409` via getConfusionDamage | int 0–15 | confused mon | confused mon | `confusion` | 0 | 0 |
| SECONDARY | `battle-actions.ts:1399` with numeric `chance` | int < 100 (raw) | attacker | defender | move | hit | secondary index |
| SECONDARY_STATUS_PICK | Tri Attack `moves.ts:19853`, [C] Dire Claw `ch/moves.ts:223` | index | attacker | defender | move | hit | secondary index |
| SELF_EFFECT_ROLL (explicit chance only) | `battle-actions.ts:1381` | int < 100 | user | −1 | move | hit | 0 |
| MULTIHIT_COUNT | `ch/scripts.ts:439` | index into the 2–5 table | attacker | −1 | move | 0 | 0 |
| RANDOM_TARGET | `side.ts:372` via getRandomTarget; `battle.ts:2523`; `conditions.ts:305` — **only** draws at execution `getTarget` (incl. randomNormal), retarget after a faint, resolve-time draws for non-randomNormal moves without targetLoc (Encore/Instruct), twoturnmove retarget | target side·4+b | user | −1 | move | 0 | phase: 0 resolve, 1 getTarget, 2 other |
| FRAC_PRIO | QUICK_CLAW `items.ts:5017`, PRIORITY_PROC Quick Draw `abilities.ts:3739` | bool | mon | −1 | item / ability | 0 | 0 |
| PARALYSIS | `ch/conditions.ts:5` | bool | mon | −1 | `par` | 0 | 0 |
| SLEEP_DURATION | `ch/conditions.ts:23` | index into [2,3,3] | mon | −1 | `slp` | 0 | 0 |
| FREEZE_THAW | `ch/conditions.ts:49` | bool | mon | −1 | `frz` | 0 | 0 |
| CONFUSION_DURATION | `conditions.ts:174` | int | mon | −1 | `confusion` | 0 | 0 |
| CONFUSION_SELFHIT | `conditions.ts:187` | bool | mon | −1 | `confusion` | 0 | 0 |
| PARTIAL_TRAP_DURATION | `conditions.ts:227` | int | trapped mon | −1 | `partiallytrapped` | 0 | 0 |
| LOCKEDMOVE_DURATION | `conditions.ts:265` | int | mon | −1 | `lockedmove` | 0 | 0 |
| PROTECT_STALL | `conditions.ts:456` | bool | user | −1 | move | 0 | 0 |
| ALLY_SWITCH_STALL | `moves.ts:343` | bool | user | −1 | `allyswitch` | 0 | 0 |
| INFATUATION | `moves.ts:745` | bool | mon | −1 | `attract` | 0 | 0 |
| FORCED_SWITCH_PICK | `battle.ts:1586` | brought index | dragged-out mon | −1 | running action's move id | 0 | 0 |
| ABILITY_ONHIT_PROC | Static, Flame Body, Poison Point, Cute Charm, Cursed Body, Effect Spore, Poison Touch | bool / int | holder | other mon | ability | hit | 0 |
| ABILITY_RESIDUAL_PROC | Shed Skin, [C] Healer, Harvest | bool | holder | −1 | ability | 0 | 0 |
| ABILITY_TRACE_TARGET | `abilities.ts:5145` | index | holder | −1 | `trace` | 0 | 0 |
| ABILITY_FOREWARN_PICK | `abilities.ts:1526` | index | holder | −1 | `forewarn` | 0 | 0 |
| ABILITY_PICKUP_TARGET | `abilities.ts:3272` | index | holder | −1 | `pickup` | 0 | 0 |
| ABILITY_MOODY_STAT | `abilities.ts:2715, 2726` | index | holder | −1 | `moody` | 0 | 0 raise / 1 lower |
| ITEM_FOCUS_BAND | `items.ts:2273` (every damage instance) | bool | holder | damage source or −1 | the damaging effect id | move hit or 0 | occurrence ordinal within (step, holder, subject) |
| MOVE_FICKLE_BEAM | `moves.ts:5231` | bool | user | target | move | hit | 0 |
| MOVE_SHELLSIDEARM_TIE | `moves.ts:16232` | bool | user | target | move | 0 | 0 |
| MOVE_ACUPRESSURE_STAT | `moves.ts:155` | index | user | target | move | 0 | 0 |
| MOVE_CALL_PICK | Sleep Talk `moves.ts:16887` | index | user | −1 | move | 0 | 0 |

**Value translation.** The harness converts PS `sample` indices into the engine domain: list index → side·4+b, brought index, or table index.

## 6. Don't-care kinds (PS side dropped; engine never draws)
| Kind | Definition | Evidence |
|---|---|---|
| RANDOM_TARGET_NOMINAL | `randomFoe` draws for spread/side moves' nominal target | RNG §6.3 (0% state impact) |
| RANDOM_TARGET_NOEFFECT | draws under the `getActionSpeed` frame (commit and every re-sort); resolve-time draws for randomNormal moves (execution redraws) | RF #14; `sim/battle.ts:2453-2504, 2672-2716` |
| SELF_EFFECT_ROLL (chance undefined) | `selfDrops` unconditional roll | RNG §6.3 |
| SECONDARY (chance undefined) | secondary with no chance | RNG inventory note |
| SPEED_TIE_EACH_EVENT (except Weather) | eachEvent ties, e.g. `Update` | RNG §6.3 |
| SPEED_TIE_HANDLER_ORDER (non-MUL hooks) | runEvent ties on GATE/REPL/side-effect hooks | RNG §6.3; MV §7.2 |
| QUEUE_INSERT_TIE | `insertChoice` random index among equal keys | RNG §6.3 |
| TEAM_SORT | preview sort ties | TL §11.5 |
| GENDER | construction gender draw (teams always set gender) | RNG §2.1 |

**Re-validation.** Weekly (07 §4.3), leave each don't-care kind free in PS over 10k battles while forcing everything else. A non-zero state impact means the kind must be promoted. Promotion is a contract change.

## 7. Degenerate draws (dropped on both sides)
- `randomChance(n, d)` with n ≥ d or n ≤ 0;
- `sample` over 1 element;
- `random(1)`, `random(a, a+1)`.

The engine skips them when its own arguments are degenerate. If the engine's arguments are non-degenerate where PS's were degenerate (or the reverse), the result is a `DRAW_MISMATCH`, which is intended: it localises an argument bug.

## 8. Unreachable kinds in Reg M-C at d601443 (assert; any occurrence is a `HARNESS`/`UNMAPPED` error)
- **G-Max and Dynamax:** GMAX*, SPEED_TIE_DYNAMAX_END.
- **Other generations:** QUICK_CLAW_GEN2/3; gen<5 and Loaded Dice multihit; base par/slp/frz/Healer/Dire Claw (overridden by [C]).
- **No legal user:** MOVE_MAGNITUDE, MOVE_PRESENT, MOVE_PSYWAVE, MOVE_CONVERSION2_TYPE, ITEM_STARF_STAT, Toxic Chain, Neutralizing Gas end-sort, Metronome / Assist call picks.
- **Rule not in format:** RULE_CATEGORY_SWAP.

## 9. Comparison semantics (07 §1.3)
For each step k → k+1:
1. multiset(PS forced-kind draws, keyed) == multiset(engine draw-log entries, keyed);
2. for equal keys, `args` are equal (e.g. `randomChance(acc′, 100)` numerators);
3. tie records match 1:1 by key with equal `tied`;
4. no engine `ERR_FORCED_MISSING`.

**Failure categories:**
- `DRAW_MISMATCH`: an unmatched key or tie record;
- `DRAW_ARGS`: same key, different arguments.
