# Review: semantic fidelity to PS and internal consistency (design 03 / 04 / 07)

**Scope.** This review checks the design package in `design/` (mainly 03, 04, 07, plus the files that restate the same contracts) against three sources: the research notes TL, MV, CH and RNG, and the PS source at commit `d601443` (`/tmp/psrepo-move/ps`, built `dist/`).

**Package edits during the review.** Several files changed while the review was running: 01, 02, 03, 05–11 and CLAUDE.md between 19:46 and 19:49. I re-read 03 and 07 after those edits. 04 has been unchanged since 19:26.

**Runtime checks.** Claims marked *verified* were run against PS with `PS_DIST=/tmp/psrepo-move/ps/dist node /tmp/review-fid/t1.js <case>`. That is a scratch script outside the package; the cases are `activeturns`, `revival2`, `revival4`, `round` and `instructcounter`.

**Format.** Items are ordered by severity, then by impact. Each item gives the problem, the evidence and a fix. "Exact text" means a drop-in replacement.

---

## BLOCKER

### 1. BLOCKER — 04 §12, 01 ADR-009, 04 §2.1: NA = 8 is too small for legal play
- **Problem.** At turn start, Block A executes:
  - `beforeTurn`;
  - then, per slot, either (switch + that switch's own runSwitch) or (satellite + mega).

  Start-of-turn switches aren't batched, so each switch has its own runSwitch execution. Three combinations each reach **9** light actions:
  - all four actives switch: 1 + 4×2;
  - three switch and one mover has a satellite and a mega;
  - two switch and two movers each have a satellite, with one mega per side.

  Each of these sets the overflow flag, but the gate requires an error rate of 0. Under the `random` policy this happens in roughly 3×10⁻⁵ of turns, i.e. hundreds of battles in a 1M-battle gate corpus.
- **Evidence.**
  - TL §2.8 (unbatched start-of-turn switches).
  - `sim/battle-queue.ts:375-409` (insertChoice puts runSwitch ahead of the remaining switches).
  - `sim/battle-actions.ts:208-212`.
  - TL §11.4 (N_ACT = 13 includes moves).
- **Fix.**
  - 04 §12 row, exact text: `| NA (Block A iterations) | 10 | turn start: beforeTurn + 4 × (switch + runSwitch) = 9; resume after a Block-A pause: ≤ 3 × 2 remaining + instaswitch + batched runSwitch = 8; +1 margin |`.
  - ADR-009: change "Bound 8" to "Bound 10".

### 2. BLOCKER — 00 §3, 01 ADR-012 and ADR-029, CLAUDE.md, 10 M0.4 vs 03 §5, 04 §5.5 and §8.1–8.4, 07 §3 and §4.1: the normative DrawKey file is missing, and the restated versions disagree
- **Problem.** ADR-012 makes `contract/drawkey.md` normative and says "every other document references it rather than restating it". The file isn't in the package. Meanwhile the restatements conflict:
  - **step_tag values.** 03 §5 gives 5: {COMMIT, A, B, RESIDUAL, SWITCHIN}. 04 §8.4 and 07 §4.1 give 7: {COMMIT, A_SWITCH, A_MEGA, A_OTHER, B_MOVE, RESIDUAL, SWITCHIN}.
  - **ADR-012 coordinates.** `(turn, phase_tag, actor_slot, actor_action_ordinal, target_slot, kind, hit, ord)` lacks `subject` and `sub`.
  - **04 §8.1 packing.** It packs an `extra (4)` field that no key definition contains.
  - **SPEED_TIE_SELECT key.** 04 §5.5 uses `(turn, block, selection_index)`. 07 §4.1 uses `(turn, step_tag, sub = executed-action ordinal in turn)`.
  - **Step identity.** 07 §3's example key has `step_actor` and `step_subject`. 07 §4.1, 03 §5 and 04 §8.2 don't.
  - **FRAC_PRIO key.** 04 §5.2 uses `(turn, COMMIT, actor)` with no `subject`. 07 includes subject, which is needed for Quick Draw plus Quick Claw on the same mon.

  M0.4 (the oracle trace writer) can't be implemented without guessing.
- **Fix.**
  - Add `design/contract/drawkey.md` as the only definition. It should hold the key (with item 13's field changes), the action-kind → step_tag table, the RNG-inventory-kind ↔ DrawKind table (item 12), and the tie-record schema (item 12).
  - In 03 §5, 04 §5.2, §5.5, §8.1, §8.2, §8.4, 07 §4.1 and ADR-012, replace the restated fields with "see `contract/drawkey.md`".
  - 04 §8.1 keeps only the bit widths, derived from the contract.

---

## MAJOR

### 3. MAJOR — 04 §7.1 and §7.3: the move-pipeline step order differs from PS (state and draw-set divergences)
- **Problems.**
  - **(a) Drain.** Drain is applied per hit inside `spreadDamage`, right after that target's damage. 04 §7.3.7 applies it "after the loop".
  - **(b) Per-hit effects.** `runMoveEffects`, `selfDrops`, `secondaries`, `forceSwitch`, `DamagingHit` and `AfterHit` all run **per hit, inside `spreadMoveHit`, in that order**. The design runs DamagingHit inside the loop but effects, secondaries and AfterHit after it. So Rocky Helmet, Static or Justified fire before the move's own onHit and secondaries, and a Drain Punch user can faint to Rocky Helmet before it heals.
  - **(c) Substitute.** `TryPrimaryHit` (Substitute) is step 0, before `getSpreadDamage`.
  - **(d) Target before gates.** `getTarget`, including its RANDOM_TARGET retarget draw, runs **before** the BeforeMove gates. A fully paralysed Outrage user has already drawn a target.
  - **(e) Pressure.** Pressure PP is taken in `useMoveInner`, after ModifyType/ModifyMove and `getMoveTargets`; Expanding Force's target change alters the Pressure set. Base PP is taken in runMove, before useMove.
- **Evidence.**
  - `sim/battle-actions.ts:267-365` (runMove) and `:429-610` (useMoveInner).
  - `data/mods/champions/scripts.ts:314-423` (spreadMoveHit) and `:426-588` (hit loop).
  - `sim/battle.ts:2180-2190` (drain inside spreadDamage).
  - MV §2.1, §2.2, §2.6.
- **Fix.** Replace 04 §7.1 steps 1–9 and §7.3 steps 7–9 with the PS sequence below.
  - **runAction.** Skip if the actor is fainted or inactive.
  - **runMove.** `active_move_actions += 1` → getTarget (may draw RANDOM_TARGET) → OverrideAction → BeforeMove gates (H1) → beforeMoveCallback → `last_damage = 0` → base PP (skipped if locked) → moveUsed.
  - **useMove.** ModifyTarget → ModifyType/ModifyMove (H3) → getMoveTargets (retarget; RedirectTarget H4) → Pressure (H5) → TryMove → tryMoveHit | trySpreadMoveHit steps 0–6.
  - **Hit loop, per hit.**
    1. multiaccuracy re-roll (hits ≥ 2);
    2. spreadMoveHit: move TryHit on targets[0] → TryPrimaryHit/Substitute (H12) → getSpreadDamage per target (CRIT, BasePower, …, DAMAGE_ROLL) → spreadDamage per target (Damage H21 → apply → drain, rounded) → runMoveEffects (H22) → selfDrops → secondaries (H25) → forceSwitch (H27) → DamagingHit (H26, left-to-right) → AfterHit per target (runs even if the attacker fainted [C]) → attacker EmergencyExit;
    3. Update.
  - **After the loop.** resolveFaints → recoil (rounded) → gotAttacked / timesAttacked → Update → AfterMoveSecondary (not suppressed by Sheer Force [C]) → per-target EmergencyExit.
  - **Back in useMove.** AfterMoveSecondarySelf (Life Orb, Shell Bell; suppressed by Sheer Force) → attacker EmergencyExit.
  - **Back in runMove.** AfterMove → resolveFaints + checkWin.

### 4. MAJOR — 04 §7.1 step 9: `activeMoveActions` is incremented at the start of runMove, not after the move
- **Problem.** PS increments `activeMoveActions` as the first statement of `runMove`. So it also counts actions that fail in BeforeMove (sleep, flinch, full paralysis) or for lack of PP.
  - Fake Out's inherited `onTry` tests `activeMoveActions > 1`, which assumes the increment has already happened.
  - [C] `onDisableMove` disables Fake Out once `activeMoveActions > 0`.
  - With the design's ordering, a lead fully paralysed on turn 1 is still offered Fake Out on turn 2, but PS rejects it. That is a MASK divergence, and the onTry threshold is off by one.
- **Evidence.** `sim/battle-actions.ts:274`; `data/mods/champions/moves.ts:349-356`; TL §8.4.
- **Fix.**
  - 04 §7.1 step 1, exact text: "1. Skip silently if the actor is fainted or inactive (runAction). Otherwise `active_move_actions += 1` first, before getTarget and the BeforeMove gates (PS runMove:274)."
  - Remove the increment from step 9.

### 5. MAJOR — 04 §7.3 step 7 and §7.8: recoil and drain use `Math.round`, not trunc
- **Problem.** "Recoil and drain (trunc semantics per PS)" and "Struggle recoil … trunc rounding per PS" are both wrong. Every recoil or drain amount with a fractional part ≥ 0.5 comes out 1 HP different.
- **Evidence.**
  - `sim/battle-actions.ts:1437-1440`: `Math.round(baseMaxhp/4)`, `Math.round(damageDealt*recoil[0]/recoil[1])`, `Math.round(maxhp/2)`.
  - `sim/battle.ts:2188`: drain uses `Math.round`.
  - MV §2.8.
- **Fix**, exact text:
  - "Recoil = max(1, round(total·n/d)) via `damage` (Rock Head blocks it).
  - Struggle = max(1, round(baseMaxhp/4)) via `directDamage` (no Damage event).
  - Drain = round(dmg·n/d), per target per hit.
  - `round` is JS `Math.round`, i.e. round-half-up for positive values: `(2·x·n + d) // (2·d)`."
  - Add odd-value golden tests.

### 6. MAJOR — 04 §7.1 step 2: the BeforeMove gate list is not in PS order
- **Problem.** The design calls the list "in PS order", but PS orders gates by `onBeforeMovePriority`, descending:

  | Priority | Gate |
  |---|---|
  | 11 | mustrecharge |
  | 10 | slp, frz |
  | 9 | Truant |
  | 8 | flinch |
  | 7 | Disable |
  | 6 | Gravity, Heal Block, Throat Chop |
  | 5 | Taunt |
  | 3 | confusion |
  | 2 | attract |
  | 1 | par |
  | 0 | choicelock |

  The design puts Truant last, puts Taunt/Disable/Gravity/Heal Block/Throat Chop after attract, and puts paralysis before attract. Example: a taunted, confused, paralysed mon picks a status move. In PS it fails at Taunt with no CONFUSION_SELFHIT and no PARALYSIS draw. The design's order makes both draws (DRAW_MISMATCH) and can produce a self-hit.
- **Evidence.**
  - `data/conditions.ts:39, 66, 96, 179, 201, 367`.
  - `data/abilities.ts:5190`.
  - `data/moves.ts:742, 3700, 7822, 8310, 19000, 19406`.
- **Fix**, exact text: "Gates run in the generated `effect_event_order` for BeforeMove; do not hand-order them. For reference: recharge 11, sleep/freeze 10, Truant 9, flinch 8, Disable 7, Gravity/Heal Block/Throat Chop 6, Taunt 5, confusion 3 (CONFUSION_SELFHIT), attract 2 (INFATUATION), paralysis 1 (PARALYSIS), choice lock 0."

### 7. MAJOR — 04 §11 end_turn step 2: one-turn conditions end in the residual walk, not in end_turn
- **Problem.** 04 says end_turn ends "one-turn side conditions (Wide Guard, Quick Guard), the Protect family's volatile end" and resets "the Helping Hand count". In PS these are duration-1 effects:
  - The residual duration tick removes them, sorted last.
  - Their `End` runs inside the walk, with resolveFaints after it.
  - So they are already gone at the end-of-turn SWITCH decision.

  The engine as written still holds them at that decision. That is a STATE divergence on every replacement after a turn with Protect, Wide Guard, Helping Hand or Follow Me.
- **Evidence.** MV §4.2 ("one-turn effects … removed in the residual of the turn they were set"); TL §8.2 last row; `sim/battle.ts:520-528`.
- **Fix.**
  - Delete those three phrases from 04 §11.
  - Add to 04 §9.4: "Duration-only effects (Protect family, stall, Helping Hand, Follow Me, Rage Powder, Wide/Quick Guard, flinch, Roost, lockedmove/twoturnmove/mustrecharge, …) are residual candidates with order = last (2³²) and end at their residual position."

### 8. MAJOR — 04 §2.1 Block A: no post_action after light actions
- **Problem.** Block A calls only `run_light_action`. PS runs the runAction tail after **every** action except `team`, `pass` and early returns. The tail does:
  - phazing;
  - resolveFaints + checkWin (hazards can KO an incoming mon);
  - Update (e.g. a Sitrus Berry after hazard damage on a start-of-turn switch-in);
  - Emergency Exit for the first runSwitch pokemon;
  - the switch request and pause.

  Without the tail, Emergency Exit on a start-of-turn switch-in never pauses, because 04's pause rule is triggered from post_action. Also, the beforeTurn action's tail runs an Update pass at the start of every turn.
- **Evidence.** `sim/battle.ts:2861-2986`; TL §6.3 and §11.4.
- **Fix.**
  - 04 §2.1 Block A body: `run_light_action(e); post_action(e)` unless e returned early.
  - post_action's re-sort step is a no-op in Block A.

### 9. MAJOR — 04 §6.4: wrong comparator for several events; effectOrder applies only to SwitchIn and RedirectTarget
- **Problems.**
  - **(a) Left-to-right events.** `Invulnerability`, `TryHit`, `DamagingHit` and `EntryHazard` are sorted by `compareLeftToRightOrder`: (order, priority, **target index**), stable, with no speed key and no shuffle. A spread move hitting a Static holder and a Flame Body holder resolves them in target order. A speed sort can give the attacker brn instead of par.
  - **(b) priorityEvent.** RedirectTarget, WeatherModifyDamage and LockMove use `compareRedirectOrder` with a stable sort: (priority, speed, holder `abilityState.effectOrder`), then collection order. That is deterministic, so "canonical order" is **not** a don't-care. Example: two equal-speed Lightning Rod holders (Raichu, Manectric, Rhyperior, Pincurchin and Mega Sceptile are legal) redirect to the older ability state.
  - **(c) effectOrder.** effectOrder is set only for callbacks ending in `SwitchIn` or `RedirectTarget`; elsewhere, including Residual, it is 0. Using `seq` as a universal last key removes ties that PS shuffles, e.g. both sides' Tailwind ending, or `protect` + `stall` on one mon at residual. The engine then misses the SPEED_TIE_FIELD_EVENT records that PS produces.
  - **(d) Inactive targets.** When the target is inactive and the source isn't active, runEvent gathers only field and battle handlers.
- **Evidence.** `sim/battle.ts:409-431, 796-802, 1001-1007, 1067-1080`; MV §1.3, §1.7, §1.8.
- **Fix.** Replace the "Key" and "Ties" bullets of 04 §6.4 with:
  - "(i) Invulnerability / TryHit / DamagingHit / EntryHazard: (order↑, priority↓, target index↑), stable, no RNG.
  - (ii) priorityEvent: (priority↓, speed↓, holder ability effectOrder↑), stable in collection order, no RNG.
  - (iii) All others: (order↑, priority↓, speed↓, subOrder↑, effectOrder↑), with effectOrder = state seq only for `*SwitchIn` / `*RedirectTarget` callbacks and 0 otherwise. Exact ties under (iii) are PS shuffles."
  - Add rule (d) to §6.2.

### 10. MAJOR — 03 §4.3 `speed_cached`, 04 §5.4 and §10.3: PS writes `pokemon.speed` in more places than updateSpeed
- **Problem.** "Updated only at PS's updateSpeed points" misses `setSpecies`, which sets `speed = storedStats.spe`: the raw stat, not negated under Trick Room, with no ModifySpe. setSpecies runs at:
  - construction;
  - every `clearVolatile` (switch-out and faint);
  - every `formeChange`: Mega Evolution, Stance Change, Mimikyu-Busted, Zero to Hero, Hunger Switch, Forecast;
  - Transform.

  Consequences:
  - After a Mega in Block A, the Mega's cached speed is its raw new stat until the next updateSpeed.
  - A dragged-in mon (no insertChoice) enters with its raw stored speed, and that value feeds runSwitch's speedOrder and SPEED_TIE_SWITCHIN.
  - Fainted mons still in a slot are part of that sort (`getAllActive(true)`), at raw speed.

  Two further details:
  - The SWITCH-phase commit also calls updateSpeed.
  - The instaswitch sort key for a **fainted** outgoing mon is a fresh `getActionSpeed()` on an inactive mon: boosts are cleared and only field/battle ModifySpe handlers apply. It is not `speed_cached`.
- **Evidence.**
  - `sim/pokemon.ts:509, 1408-1440` (line 1439), `1572`.
  - `data/mods/champions/scripts.ts:174`.
  - `sim/battle-actions.ts:208-235`.
  - `sim/battle.ts:3057`; TL §2.8.
- **Fix.**
  - 03 row, exact text: "`speed_cached` | i16 | `pokemon.speed` | written by updateSpeed (commit of MOVE **and** SWITCH, before each Block-B selection, start of residual, insertChoice for that mon) and by setSpecies (= raw stored Spe: init, clearVolatile on switch-out/faint, formeChange incl. Mega, Transform). Never written by dragIn."
  - Mirror this in 04 §5.4.
  - 04 §10.3: "instaswitch key = getActionSpeed of the outgoing mon at commit (for a fainted mon: stored Spe, negated under TR)".

### 11. MAJOR — 03 §4.3 `faint_queued`, 04 §11: faint order isn't represented, and it decides the winner
- **Problem.** PS keeps an ordered `faintQueue`, pushed in the order `faint()` is called. On a simultaneous wipe, the side of the **last shifted entry** wins, even if that entry was skipped as already fainted. A per-mon bool can't reproduce this. Orders that matter include:
  - Explosion's user first, then its targets;
  - spread KOs in target order;
  - recoil after targets;
  - Destiny Bond;
  - weather chip in eachEvent order.

  Two related details are also missing:
  - `faint()` sets `switchFlag = false`, so a KO'd U-turn user doesn't pause.
  - `faint()` is a no-op if the mon is already fainted or queued.
- **Evidence.** `sim/pokemon.ts:1602-1614`; `sim/battle.ts:2561-2670`. `checkWin(faintData)` receives the last shifted entry.
- **Fix.**
  - 03 BattleMeta: add "`faint_queue` | [8] i8 (side·4+b, −1) + `faint_queue_len` i8 | `battle.faintQueue` | FIFO".
  - 04 §11, exact text: "faint(mon): no-op if fainted or faint_queued; else hp = 0, switch_flag = 0, faint_queued = 1, push. resolveFaints pops in order; on a double wipe the winner is the side of the last popped entry."

### 12. MAJOR — 07 §1.3, §4.3 and §4.4, 04 §5.5 and §10.2: semantic tie records are under-specified and don't match how PS sorts
- **Problems.**
  - **(a) Shuffle draws.** Nothing says that raw `speedSort` shuffle draws (RNG kinds SPEED_TIE_ACTION_ORDER, SPEED_TIE_SWITCHIN_ORDER, SPEED_TIE_FIELD_EVENT) leave the draw multiset and are replaced by `ties` records. PS shuffles at every commit, including order-200 moves that get re-sorted later, and at every re-sort. Without that rule, DPR fails on every tie.
  - **(b) Selection ties.** PS keeps a fainted actor's entries in the queue and sorts them; getActionSpeed recomputes their speed. Their action then returns early with no tail. The contract doesn't say whether such entries count as `tied`, or whether they consume an ordinal.
  - **(c) runSwitch sorts.** runSwitch sorts **all** slot occupants, including fainted ones, at **every** runSwitch: single start-of-turn switches and drags too, not only batches as 04 §10.2 implies. One sort can contain several tie groups, which `sub = SwitchIn-batch ordinal` can't tell apart.
  - **(d) Residual ties.**
    - PS collects handlers of fainted-in-slot holders and then skips them.
    - Side and field holders (speed 0) have no slot for the "(holder slot, effect id)" member order.
    - SwitchIn `fieldEvent` ties (04 §6.4) have no ordinal; the only one defined is "within this residual walk".
  - **(e) Persistence.** The ordinals must persist across SWITCH steps; 03 §4.6 has no field for them.
  - **(f) Magician.** Magician (Delphox and Klefki, both legal) shuffles speed-tied hit targets (RNG kind SPEED_TIE_ABILITY, `data/abilities.ts:2483`). It is neither forced nor don't-care in 04 §8.3 or 07 §4.4.
- **Evidence.** `sim/battle.ts:434-465, 2975-2984, 3076`; `sim/battle-actions.ts:228-235`; `sim/battle.ts:489-575`; RNG §2.2.
- **Fix** (in `contract/drawkey.md` and 07 §1.3/§4.3):
  1. Shuffle draws inside speedSort are never compared as draws. Each sort's tie groups become tie records, compared 1:1 by key with equal `tied` sets, and forced by winner or permutation.
  2. **SELECT.** `tied` = entries with equal (order, priority, speed) whose pokemon is active and not fainted. An action that returns early emits nothing and doesn't advance the ordinal. The ordinal counts executed selectable actions this turn and is stored in `TurnScratch.select_ord`.
  3. **SWITCHIN.** Every runSwitch (batched, single, drag, start) emits one record per tie group of its sort over all slot occupants, including fainted ones (by `speed_cached`). Key: sub = runSwitch ordinal this turn, ord = group index. Members are ordered by slot.
  4. **FIELD_EVENT.** Key: sub = fieldEvent ordinal this turn (Residual and every SwitchIn fieldEvent), ord = group index. Omit handlers that will be skipped (fainted holder, unless it is a slot condition). Member order: field < side 0 < side 1 < p1a, p1b, p2a, p2b; then effect id.
  5. Add SPEED_TIE_ABILITY (Magician) as a forced tie kind.
  6. Add `select_ord`, `runswitch_ord` and `fieldevent_ord` to 03 §4.6.

### 13. MAJOR — 07 §4.1 per-kind rules: key fields that can't be computed identically on both sides, or that collide
- **Problems.**
  - **(a) No step identity.** The PoC needed the running queue action in the key: exact-log reproduction rose from 45% to 96% when it was added (RNG §6.2). Without it, handler-driven keys collide. Example: CONFUSION_DURATION (actor = receiver, subject = confusion) for a mon that is confused, cured by a Persim or Lum Berry, and confused again by another move in the same turn. Also, `action_ord` is ambiguous for handler draws: is it the holder's ordinal or the attacker's?
  - **(b) Focus Band.** ITEM_FOCUS_BAND draws on **every** damage instance, residual included (psn/brn/sand/Leech Seed/Salt Cure). In one residual walk, poison and sand on the same holder produce identical keys.
  - **(c) Forced-switch subject.** FORCED_SWITCH_PICK's "subject = move / item causing it" can't be derived: the tail's phazing loop only sees `forceSwitchFlag`, which 03 stores as a bool.
  - **(d) Missing row.** CONFUSION_SELFHIT_DAMAGE_ROLL has no row.
  - **(e) Inactive actors.** Actor and target are 3-bit slots, but Future Sight's CRIT and DAMAGE_ROLL belong to a source that may be benched or fainted. The PoC labels such mons `p1:species`.
  - **(f) Actor definition.** 04 §8.4 says actor = "the acting slot for the current queue entry"; 07 says actor = "attacker". They differ for a Magic Bounce reflection.
  - **(g) step_tag mapping.** The mapping from PS action kinds to step_tag is unspecified. Do drag-in SwitchIn draws inside a move's tail get B_MOVE or SWITCHIN? Which tag do beforeTurn, revival and the start action's runSwitch get?
- **Evidence.** `research/poc/lib/context.js:12-16, 114-118`; RNG §2.3 #7 and §6.2; `data/items.ts:2273`.
- **Fix** (in `contract/drawkey.md`):
  - Add `step_actor` (side·4+b of the running action's pokemon; −1 for COMMIT and RESIDUAL) and `step_subject` (its move id or action kind).
  - `action_ord` = step_actor's move-action ordinal.
  - actor/target = side·4+b (4 bits), where actor = the mon executing the move (the bouncer for Magic Bounce).
  - ITEM_FOCUS_BAND: subject = damaging effect id, target = source or −1, hit = move hit or 0.
  - FORCED_SWITCH_PICK: subject = the running action's move id.
  - Add CONFUSION_SELFHIT_DAMAGE_ROLL: actor = target = the confused mon, subject = confusion.
  - Table `action.choice → step_tag`:

    | PS action | step_tag |
    |---|---|
    | switch, instaswitch | A_SWITCH |
    | megaEvo | A_MEGA |
    | beforeTurn, beforeTurnMove, revivalblessing, priorityChargeMove | A_OTHER |
    | runSwitch (including start) | SWITCHIN |
    | move (any order) | B_MOVE |
    | residual | RESIDUAL |
    | none | COMMIT |

    Inline switch-ins (drags) inherit the enclosing action's tag.

### 14. MAJOR — 04 §5.2, §5.6 and §7.2, 07 §4.1 and §4.4: RANDOM_TARGET draws inside getActionSpeed have no effect, but the contract treats them as forced
- **Problem.** `getActionSpeed` calls `getTarget` for every queued move, at commit and at **every** re-sort.
  - For `randomNormal` moves (Outrage, Thrash, Petal Dance, Raging Fury, Struggle, Ghost-type Curse), `getTarget` always calls `getRandomTarget` → `sample(living foes)`. That is a non-degenerate 2-way draw per queued move per re-sort, and its result is discarded: no legal ModifyPriority handler reads the target.
  - The resolve-time draw for a randomNormal move also has no effect, because execution draws again.
  - The PoC classifies all of these as RANDOM_TARGET (single-target move). The don't-care list has only RANDOM_TARGET_NOMINAL (spread and side moves).
  - The engine never makes these draws, so every turn with a locked Outrage user facing two foes is a DRAW_MISMATCH.
  - 04 §7.2 also doesn't say that randomNormal **always** draws at execution.
- **Evidence.**
  - `sim/battle.ts:2672-2716, 2453-2504` (the `move.target !== 'randomNormal'` guard), `2975-2984`.
  - `sim/battle-queue.ts:268-275`.
  - `research/poc/lib/classify.js:20-24`.
  - TL §9.4 "PRNG note" calls these no-effect draws. MV §2.9 row l lists only spread moves.
- **Fix.**
  - 07 §4.4: add don't-care kind `RANDOM_TARGET_NOEFFECT` = (draws under the getActionSpeed frame) ∪ (resolve-time draws for randomNormal moves).
  - 04 §5.2: "randomNormal moves draw nothing at commit. Non-randomNormal moves with no targetLoc (Encore re-queue, Instruct with lastMoveTargetLoc 0) draw RANDOM_TARGET (ord = resolve) at resolve."
  - 04 §7.2: "randomNormal moves draw RANDOM_TARGET (ord = getTarget) at every execution, before the BeforeMove gates."

### 15. MAJOR — 04 §2.1, §5.5 and §10.3, ADR-009: a prioritized (order-3) move in the paused remainder breaks the Block A/B split (*verified*)
- **Problem.** PS's order after a SWITCH commit differs from the design's:
  - PS puts the new, sorted instaswitches **in front of** the unsorted remainder, without ever comparing the two groups.
  - It then inserts each runSwitch by `comparePriority`. That places the runSwitch **after** any order-3 move already in the remainder.
  - The re-sort in the instaswitch's tail keeps order 3 ahead of 101.
  - Net PS order: instaswitch → prioritized move → runSwitch → other moves.

  The design runs instaswitch + runSwitch in Block A, then the move in Block B. A pure key argmax would even run the move before the instaswitch.

  This is reachable with Round, which is legal. The first Round prioritizes the partner's Round. If the first Round triggers Eject Button or Emergency Exit, the partner's order-3 Round is left in the remainder.
- **Evidence.**
  - `sim/battle.ts:3063-3077` (sort the new actions, then push the old queue).
  - `sim/battle-queue.ts:375-409`.
  - Round's `prioritizeAction` (`data/moves.ts:15514`).
  - The `round` case prints: `|switch|p2a: Incineroar` → `|move|p1b: Milotic|Round|…|[from] move: Round` → `|-ability|p2a: Incineroar|Intimidate`. Competitive's +2 lands after the Round.
- **Fix.**
  - 03 §4.5: add `prepend [Q] bool`. It is set on entries created by a SWITCH-phase commit. The selection key becomes (prepend↓, order↑, priority↓, speed↓). The re-sort before a move clears all prepend flags.
  - 04 §2.1: replace "Every order-<200 entry precedes every move entry" with "…except that a runSwitch inserted after an instaswitch sorts behind order-3 moves already queued".
  - Allow a Block-B iteration to execute a RUN_SWITCH entry when it is the selected head, through a masked light branch. Set NB = 8.

### 16. MAJOR — 04 §10 and §2.1 Block C: Revival Blessing re-entry into an active slot runs after residual when no move remains (*verified*)
- **Problem.** For a revived **in-slot** ally, the `revivalblessing` action re-adds it with `queue.addChoice`, which resolves and pushes to the **end**, after RESIDUAL, with no sort. The entry moves to the front only through the next re-sort, which happens only if the next action is a move.
  - If the ally had already moved this turn, only RESIDUAL remains. PS then runs residual first, then the instaswitch and the runSwitch.
  - In that case the residual tail skips checkFainted, Update and the residual Emergency Exit, because `nextAction` is an instaswitch.
  - The instaswitch's switchIn has `oldActive === incoming`. It runs SwitchOut (no handlers fire, because the target is inactive), ability and item End, `cancelAction` on itself (dropping its stale move entry) and clearVolatile.

  The design's Block A would run the instaswitch immediately, since it is order 3.
- **Evidence.**
  - `sim/battle.ts:2821-2838, 2909, 2925`; `sim/battle-queue.ts:307-313`.
  - `revival4` prints `|-heal|p1: Milotic|85/170|[from] move: Revival Blessing`, then `|upkeep`, then `|switch|p1b: Milotic`, then `|turn|2`.
  - `revival2` (the ally hadn't moved yet) shows the re-sort placing the switch before residual.
- **Fix.** Add a "Revival Blessing re-entry" subsection to 04 §10:
  - "The REVIVAL action appends an INSTASWITCH after RESIDUAL (`prepend = 0`; only a later re-sort moves it).
  - If the head after the REVIVAL tail is a move, the usual re-sort brings the instaswitch to the front.
  - Otherwise Block C runs residual (whose tail skips checkFainted, Update and residual Emergency Exit), then INSTASWITCH + RUN_SWITCH, then the empty-queue tail."
  - Add this scenario to the 07 §9.2 templates.

### 17. MAJOR — 04 §5.6 (Encore, Instruct): both go through `resolveAction(…, midTurn = false)` (*verified*)
- **Problems.**
  - **(a) Encore re-rolls.** Encore's `changeAction` → `insertChoice` re-rolls fractional priority (a FRAC_PRIO draw for Quick Claw or Quick Draw) and calls `updateSpeed` on the encored mon. The design lists only the target re-roll.
  - **(b) Encore satellites.** resolveAction adds satellites. Encore into Focus Punch or Chilly Reception inserts an order-107 priorityChargeMove ahead of the remaining moves. Encore into Counter or Mirror Coat inserts a beforeTurnMove.
  - **(c) Instruct takes element 0.** Instruct uses `resolveAction(...)[0]`. Counter, Mirror Coat and Chilly Reception have no `failinstruct` flag, so for them element 0 is the **satellite**: the instructed action is the satellite, and the move itself never runs.
- **Evidence.**
  - `sim/battle-queue.ts:206-249` (FractionalPriority at :249), `301-305`, `375-386`.
  - `data/mods/champions/moves.ts:309-340`.
  - `data/moves.ts:9643-9671, 2999-3000, 11997-11998, 2404-2405`.
  - `instructcounter` prints `-singleturn … Instruct` and no Counter.
- **Fix.** Replace two rows of 04 §5.6:
  - **Instruct:** "resolve [satellite?, move] with a FRAC_PRIO re-roll; write element 0 as an order-3 entry."
  - **[C] Encore:** "cancelAction(p); updateSpeed(p); resolve with FRAC_PRIO re-roll, satellites, RANDOM_TARGET at resolve and recomputed priority; insert by key (QUEUE_INSERT_TIE is don't-care)."

### 18. MAJOR — 03 §4.3: state PS tracks that the schema lacks
- **Missing items.**
  - **`syrupTriggered`.** Supersweet Syrup is legal on Hydrapple. This flag is a Pokemon field that persists across switches, and abilityState is re-created on every switch-in, so `ability_state` can't hold it. (`bondTriggered` isn't needed: Battle Bond requires the species `greninjabond`, which isn't legal.)
  - **Ability and item stamps.** `abilityState.effectOrder` and `itemState.effectOrder` are re-stamped at switchIn, setAbility and setItem. compareRedirectOrder needs them (item 9b).
  - **Status `fnt`.** The status enum has no `fnt`. `checkFainted` sets `status = 'fnt'` on the fainted actives of sides that can switch, and the projection compares status at the end-of-turn SWITCH decision. switchIn resets a swapped-out fainted mon's status to ''. Fainting itself doesn't clear status.
- **Evidence.**
  - `data/abilities.ts:4713-4716`.
  - `sim/battle.ts:418-424, 2540-2550, 3380-3390`.
  - `sim/battle-actions.ts:146, 210-211`.
- **Fix.**
  - 03 §4.3 add row: "`syrup_triggered` | bool | `syrupTriggered` | never reset".
  - 03 §4.3 add row: "`ability_seq`, `item_seq` | i16 | `abilityState/itemState.effectOrder` | stamped at switchIn (non-empty id), setAbility, setItem".
  - Add `fnt` to the ConditionId statuses: "set only by checkFainted; cleared when switchIn swaps the fainted mon out".
  - Note that `baseMaxhp` ≡ `maxhp`, since there's no Dynamax.

### 19. MAJOR — 03 §4.2 `active` and §6 invariant 2: fainted mons stay in their slot
- **Problem.** The schema says `active` is "−1 empty", and invariant 2 says `active ≥ 0 ⇒ is_active`. Both contradict PS, which keeps a fainted mon in `side.active[pos]` (with `isActive = false`) until a switchIn replaces it. Several mechanics depend on that reference:
  - targeting a fainted ally returns it, so the move fails rather than retargeting;
  - Revival Blessing's in-slot path tests `target.position < active.length`;
  - residual collects slot conditions through it;
  - switchIn's outgoing half acts on it.

  A revived mon that hasn't been switched in yet has `fainted = false` and `isActive = false`.
- **Evidence.** TL §7.3; `sim/battle.ts:2480-2502, 2821-2838, 489-515`; `sim/battle-actions.ts:89-154`.
- **Fix.**
  - `active`: "brought index occupying the slot, including a fainted mon; −1 only before `start`".
  - Invariant 2: "`active[s,k] = b ⇒ position[b] == k`; `is_active[b] ⇒ ¬fainted[b] ∧ b ∈ active[s]`; no mon occupies two slots".

### 20. MAJOR — 04 §4.3 (SWITCH phase), 07 §6: SWITCH legality misses two PS rules
- **Problems.**
  - **(a) Revival Blessing slot.** PS computes `forcedSwitches = min(#flagged, canSwitch)`, counting the revival slot as flagged.
    - With at least one live bench mon, the revival slot **must** pick a fainted target.
    - With none, PS accepts both reviving any fainted mon (including the fainted ally in the other slot) and `pass`.
    - The revival choice doesn't add to `switchIns`.

    The design leaves `pass` legality undefined here.
  - **(b) Queued switch targets.** Bench mons targeted by a still-queued switch are excluded from both `canSwitch` and the choice, because `switchIns` is seeded from `queue.getSwitches`. This is reachable: in a start-of-turn double switch, an Emergency Exit after hazard damage can pause Block A while the second switch is still queued.
- **Evidence.**
  - `sim/side.ts:915-1017, 1118-1144, 1334-1361`.
  - `sim/battle.ts:1580-1603`.
- **Fix**, exact text for 04 §4.3:
  - "B = live bench mons not targeted by a queued switch or instaswitch.
  - Non-revival flagged slots choose from {24+b : b ∈ B} ∪ {pass}, with exactly min(#flagged, |B|) switches overall. The revival slot counts in #flagged.
  - Revival slot: {24+b : b fainted, any position} ∪ ({pass} iff |B| < #flagged)."

### 21. MAJOR — 04 §4.3, 07 §6, 08 §3, ADR-018: action-encoding conflicts
- **Problems.**
  - **(a) Locked mons.** 04 §4.3 says "move slot of the locked move, target 0", while 07 §6 says "→ `move 1`". Recharge has no move slot at all.
  - **(b) Self target.** For `adjacentAllyOrSelf` (Acupressure, [C] Milk Drink), 04 maps self to t = 0, 08 says "2 ally/self", and ADR-018 has no self. PS requires an explicit target for this type, so self needs a defined index.
  - **(c) Flattening.** The flattening of actions 0–23 (m, t, mega) is defined nowhere.
  - **(d) Dead clause.** "Each only if that slot exists at choice time" means nothing in doubles.
- **Evidence.** `sim/side.ts:552-849`; `sim/battle-actions.ts:3` (CHOOSABLE_TARGETS); TL §4.3–4.4; CH §6.1.
- **Fix.** Use one text in ADR-018, 08 §3 and 04 §4.3:
  - "`a = 6·m + 2·t + mega` (a < 24).
  - t → PS location: 0 → +1, 1 → +2, 2 → ally. For adjacentAllyOrSelf, t = 0 → self and t = 2 → ally. Non-choosable target types allow t = 0 only.
  - Locked (lockedmove, twoturnmove, mustrecharge) and Struggle: the only move action is a = 0 (`move 1`, no target, no mega)."
  - Delete the "slot exists" clause.

### 22. MAJOR — 04 §3, §11; TL §8.4 (erratum): battle start must run the full end_turn, and end_turn's reset list is incomplete (*verified*)
- **Problem.**
  - **Start.** 04 §3 ends with "turn = 1, phase = MOVE". In PS, start drains into `endTurn()`, which:
    - increments `activeTurns` for non-fainted actives (leads have activeTurns = 1 during turn 1, so Speed Boost fires at the end of turn 1);
    - sets `newlySwitched = false`;
    - computes disables and trapping.

    TL §8.4's "a lead has activeTurns = 0 during turn 1" is wrong.
  - **end_turn omissions.** 04 §11 also omits:
    - `move_last_turn_result ← move_this_turn_result` and `move_this_turn_result = undefined`;
    - `newly_switched = false`;
    - `last_successful_move_this_turn = null`;
    - the `turn != 1` guard on the used-item, stats-raised, stats-lowered and hurt resets;
    - attackedBy pruning;
    - the fact that the per-mon loop covers fainted-in-slot mons (only activeTurns skips them).
- **Evidence.** `sim/battle.ts:1640-1790`. `activeturns` prints `Garchomp:activeTurns=1,newlySwitched=false` at turn 1, and bench mons show `newlySwitched=true` (from clearVolatile at construction, `sim/pokemon.ts:525, 1572`).
- **Fix.**
  - 04 §3 last step: "then the empty-queue tail and `end_turn()`, which sets turn = 1".
  - Replace 04 §11 step 2 with the PS list in PS order.
  - `init` sets `newly_switched = True` for all mons.

### 23. MAJOR — 03 §4.3 `attacked_by`: bound and semantics
- **Problem.**
  - PS `gotAttacked` **appends** one entry per move per target, with no per-source dedupe.
  - Entries persist across turns while the source stays active: endTurn only clears `thisTurn` or drops entries whose source left.
  - So a mon on the field for 3 turns, hit by both foes each turn, holds 6 entries, and K_ATT = 4 overflows in routine play.
  - The schema also lacks `damageValue` (number vs false/undefined), which `getLastDamagedBy` filters on. Counter, Mirror Coat, Metal Burst and Comeuppance all depend on it.
- **Evidence.** `sim/pokemon.ts:25-32, 938-963`; `data/mods/champions/scripts.ts:164, 559-566`; `sim/battle.ts:1720-1727`.
- **Fix**, exact row: "`attacked_by` | ring [K_ATT=8, 6] i16: source (side·4+b), damage, damage_is_number, this_turn, move, slot | append-only, consumers read the newest entries | evicting a this_turn entry sets an overflow flag; evicting older entries is silent".

### 24. MAJOR — 04 §8.5, 07 §4.4: two "don't-care" tie kinds can change state
- **Problems.**
  - **(a) Weather chip.** `eachEvent('Weather')` deals sand chip in speed order, with ties shuffled (SPEED_TIE_EACH_EVENT). If both sides' last mons faint to the same chip, faint order decides the winner (item 11). Speed ties are common in mirror matches.
  - **(b) Modifier chains.** `runEvent` ties between modifier contributors (for example an attacker's Life Orb and a defender's resist berry at equal holder speed, plus a screen as a third factor) change the chained modifier by rounding. MV §7.2 itself says "by at most 1 HP", which is a STATE difference.

  The weekly 0%-impact experiment can only fail to observe these cases; it can't show they don't matter.
- **Evidence.** The `sandstorm` condition (`onFieldResidual` → `eachEvent('Weather')`); `sim/battle.ts:470-481`; MV §1.10 (72 of 343 triples differ).
- **Fix.**
  - "SPEED_TIE_EACH_EVENT is don't-care except for eachEvent('Weather'), which is forced. Its tie record is keyed (turn, RESIDUAL, sub = eachEvent ordinal, ord = group).
  - SPEED_TIE_HANDLER_ORDER is don't-care except when two or more tied handlers are modifier contributors in a chain of three or more factors; those are forced."
  - Or record both as waived risks in `docs/coverage/waivers.json`. Either way, don't call them state-irrelevant.

---

## MINOR

### 25. MINOR — 04 §9.4: residual-walk wording and example order
- **Problems.**
  - **(a) Wording.** "Decrement … end on 0; otherwise run its handler" can be read as "handlers with a duration never run". In PS the callback runs when the decremented duration is still above 0 (e.g. weather damage, partial trap).
  - **(b) Missing rules.**
    - Skip a handler whose holder is fainted, unless it is a slot condition.
    - Skip stale handlers.
    - Per-mon copies of field/side residual handlers (Grassy Terrain heal) run at that mon's speed and don't tick the duration.
    - Collection walks every non-null slot occupant, fainted ones included.
  - **(c) Example list errors.**
    - Healer [C], Hydration and Shed Skin are order 5 (sub 3), before Leftovers (5, sub 4), not in the order-28 group.
    - Black Sludge, Grip Claw (§7.6), Triage (§5.3), Ring Target (§7.3) and Storm Drain aren't legal.
- **Evidence.** `sim/battle.ts:489-575`; MV §1.6 and §4.3; TL §8.2; CH §5.
- **Fix.**
  - Step 4, first bullet: "if the handler has `end` and a duration: duration −= 1; at 0 call end and skip the callback; otherwise run the callback".
  - Add the rules in (b).
  - Regenerate the example list from `residual_table`, or drop it.

### 26. MINOR — 04 §7.4 and §7.3 step 5: damage and accuracy pseudo-code details
- **Problems.**
  - **(a) Crit before BasePower.** PS draws CRIT and applies the CriticalHit veto **before** the BasePower runEvent. The design's "0 ⇒ no damage" check comes before the crit draw, so it would drop a draw that PS makes.
  - **(b) Crit table index.** `crit_table[min(stage,3)]` with `[24, 8, 2, 1]` is off by one if "stage" means critRatio, whose default is 1.
  - **(c) Parental Bond.** The hit-2 `modify(·, 1024)` (the else branch of spread, applied right after +2) is missing.
  - **(d) Accuracy wording.** "Always-hit moves still draw randomChance(100,100)" is wrong. `accuracy: true` moves draw nothing. Numeric-accuracy moves always draw `randomChance(acc′, 100)`, which is degenerate only when the final acc′ ≥ 100.
- **Evidence.** `sim/battle-actions.ts:1678-1706, 752-816`; `data/mods/champions/scripts.ts:203-213`; MV §6.
- **Fix.**
  - Order: "… → basePowerCallback → critRatio = clamp(ModifyCritRatio(move.critRatio), 0, 4); if willCrit is undefined and critRatio > 0, CRIT = randomChance(1, [24,8,2,1][critRatio−1]) → CriticalHit veto → BasePower → …".
  - Spread step: `base = spread ? modify(base,3072) : pbond_hit2 ? modify(base,1024) : base`.
  - Reword step 5 as in (d).

### 27. MINOR — 04 §7.7 and §7.9: post_action and switch-flag details
- **Additions needed.**
  - **(a) Drags.** Drags happen in the tail's phazing loop, not "immediately". They run after the whole move, in p1a, p1b, p2a, p2b order, each with an inline runSwitch.
  - **(b) Skipped Update.** Update and the residual Emergency Exit loop are skipped when the next action is an instaswitch, and after `start`.
  - **(c) Eject Button guard.** [C] Eject Button is skipped if any active already has `switchFlag === true` (set by an earlier Eject Button or Emergency Exit in the same action). A pivot's string-valued flag doesn't block it.
  - **(d) No bench.** When a flagged side has `canSwitch == 0`, the tail clears its flags, except a Revival Blessing slot's.
- **Evidence.** `sim/battle.ts:2861-2973`; `data/mods/champions/items.ts:262-277`.
- **Fix.** Add (a)–(d) to §7.7 and §7.9.

### 28. MINOR — 04 §12, 03 §4.5: QMAX justification is inconsistent and leaves no headroom
- **Problem.** 04 §12 says satellites are "carried as flags", but 03 §4.5 stores BEFORE_TURN_MOVE, MEGA and PRIORITY_CHARGE as separate entries. With separate entries, a MOVE commit can hold 4 moves + 4 satellites + 2 megas + beforeTurn + residual = 12, which is exactly QMAX. Encore into a satellite-bearing move adds one more (item 17).
- **Evidence.** TL §11.4 (18 with separate entries, 10 with flags).
- **Fix.** "QMAX = 14 (12 at commit, +1 Encore satellite, +1 margin)".

### 29. MINOR — 04 §12: K_RES = 48 is asserted, not derived
- **Problem.** Residual candidates include every duration-only volatile, per-mon field-handler copies, slot conditions per position, and fainted-in-slot holders. A legal worst case exceeds 48; a typical turn is about 20.
- **Fix.** Derive K_RES from the census in datagen (the maximum number of residual-collectible effects per holder × 4, plus side and field), keep the overflow flag, and record the value in 05.

### 30. MINOR — 03 §4.2, §4.3, §4.6: encodings left to guesswork
- **Problems.**
  - `fainted_this/last_turn` is "brought index + 1, or a bool". Retaliate is the only consumer and tests truthiness (`data/moves.ts:15002`).
  - `hurt_this_turn` needs a null sentinel.
  - `move_*_result` is called "tri-state" but has four values.
  - `switch_flag` (§4.3) and `pending_switch_request` (§4.6) duplicate each other with different codes.
- **Fix.**
  - `fainted_this_turn`, `fainted_last_turn`: bool.
  - `hurt_this_turn`: −1 = null.
  - `move_*_result`: 0 undefined, 1 null, 2 false, 3 true.
  - Derive `pending_switch_request` from `switch_flag`, or drop it.

### 31. MINOR — 04 §7.1 step 2 and §7.6: FREEZE_THAW draw condition
- **Problem.** "Draws only while the counter > 0" is ambiguous.
- **Evidence.** `data/mods/champions/conditions.ts:46-55`.
- **Fix**, exact text: "If the move has `defrost` (Burn Up only for Fire types): no-op. Else time −= 1; if time ≤ 0, thaw with no draw; else draw FREEZE_THAW = randomChance(1,4)."

### 32. MINOR — 07 §5.1 and §5.3: projection identity and value transforms
- **Problems.**
  - **(a) Unstable id.** In Champions the Pokemon id `side:baseSpeciesId` changes when a mon Mega Evolves, because baseSpecies becomes the Mega.
  - **(b) Non-integer values.** Some effect-state values aren't i16-exact: Wish `hp = maxhp/2` can be x.5, Helping Hand's `multiplier` is 1.5^k, and Wish also has `startingTurn`. Some values are Pokemon references.
- **Evidence.** CH §1.5 and §8 step 5; `research/poc/lib/project.js:46`; `data/moves.ts:20923-20924, 8592-8596`.
- **Fix.**
  - id = `side:sheet index`.
  - Each projection-map entry gets (engine field, transform ∈ {identity, ×2, count of ×1.5, ref → side·4+b, bool}).
