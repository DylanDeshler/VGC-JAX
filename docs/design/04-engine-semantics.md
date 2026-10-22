# 04 — Engine semantics (normative)

This document specifies **what the engine computes and in what order**. PS at the pinned commit is the ground truth. Where this document and PS disagree, PS wins: fix this document by ADR.

Detailed PS walkthroughs, with `file:line` citations, live in:
- `research/ps-turn-loop-spec.md` (**TL**);
- `research/ps-move-damage-event-spec.md` (**MV**);
- `research/champions-mechanics-delta.md` (**CH**);
- `research/rng-harness-poc.md` (**RNG**);
- `research/review-fidelity.md` (**RF**): this revision incorporates all of its findings, several of which were verified by running PS.

Every engine function must carry a docstring line `PS: <file>:<line-range>` naming what it mirrors.

> **Erratum in research notes.** TL §8.4 says leads have `activeTurns = 0` during turn 1. That is wrong. Battle start runs `endTurn()`, so leads have `activeTurns = 1` on turn 1 (RF #22, verified). This document is authoritative.

---

## 1. Principles

1. **Mirror, then optimise.** First write each function as a direct structural transcription of the PS function, at single-battle scope with JAX control flow. Optimise only after it passes conformance, and only with a benchmark showing the need.
2. **Every branch computed, then masked.** No Python control flow on traced values.
   - Use `jnp.where` / `lax.select` for scalar choices.
   - Use per-hook `lax.switch` for bespoke handlers.
   - Use `lax.fori_loop` for bounded loops.
   - Use `lax.cond` only on values that stay unbatched (rare; prefer masks).
3. **Bounded everything.** Each loop has a static bound and an overflow error flag (§12).
4. **Canonicalise only randomness proven state-irrelevant** (`contract/drawkey.md` §5).

---

## 2. `step` — top-level control

```
step(state, actions[2], key, cfg, dex, forced=None) -> (state', StepOut):
  if state.phase == DONE: return state, terminal StepOut (all-true masks, zero reward)  # pgx convention
  words = derive_words(key, state.step_counter)                         # §8
  a0, a1 = sanitize(actions, legal_masks(state))                        # illegal → first legal + ERR_ILLEGAL_ACTION
  match state.phase (masked; all computed):
    PREVIEW: apply_preview(a0, a1)          # §3
    MOVE:    commit_move_choices(a0, a1)    # §5.2
    SWITCH:  commit_switch_choices(a0, a1)  # §10.3
  run_until_decision()                                                  # §2.1
  state.step_counter += 1
  return state, make_stepout(state)
```

### 2.1 `run_until_decision`
```
# Block A: light actions, bound NA = 10
fori_loop(NA): if not paused/done and head is a light entry (§5.5 selection):
                  e = select_next(queue); ret = run_light_action(e)
                  if not ret.early_return: post_action(e)              # PS runAction tail runs after light actions too (RF #8)
# Block B: moves (and runSwitch entries that sort behind prioritized moves), bound NB = 8
fori_loop(NB): if not paused/done and any valid non-residual entry:
                  if head would be a move: rekey_moves(queue)          # §5.4 (TL §5.7): recompute speed & priority of ALL move entries
                  e = select_next(queue)                               # §5.5
                  match e.kind (masked): MOVE → run_move_action(e) | RUN_SWITCH/INSTASWITCH → run_light_action(e)
                  post_action(e)                                       # §7.9
# Block C: end of turn, at most once
if not paused/done and head is RESIDUAL:
   residual()                                   # §9.4
   post_residual()                              # tail: checkFainted, Update, Emergency Exit (skipped if next is instaswitch; §10.4)
   if a revival INSTASWITCH was appended after residual: run it + its runSwitch now, then the empty-queue tail (§10.4)
if not paused/done and queue empty and no pending switch:
   end_turn()                                   # §11.3
```

**Why the blocks are exact.**
- PS selects by `comparePriority` over the queue. Order-<200 entries precede moves.
- PS re-sorts only when the next action is a move (TL §5.7).
- There are two exceptions, both handled by letting Block B run light entries:
  - **Prioritized move after a pause** (RF #15, verified). After a SWITCH-phase commit, PS **prepends** the new instaswitches before the unsorted remainder. It then inserts each instaswitch's runSwitch by key, which places that runSwitch **behind** any order-3 (prioritized) move already in the remainder, such as the partner's Round. Net order: instaswitch → prioritized move → runSwitch → other moves. The `prepend` flag in the selection key (§5.5) reproduces this.
  - **Revival re-entry** (§10.4).

**Pause semantics.**
- When `post_action` finds any `switch_flag` on a side that can switch:
  - set `phase = SWITCH` and `mid_turn = True`;
  - skip all remaining iterations (masked);
  - skip PS's re-sort tail.
- A side whose flags can't be honoured (no switchable bench) has those flags cleared by the tail, except a Revival Blessing slot (RF #27d).

---

## 3. Team preview and battle start (TL §2)

1. **Preview action.** `PREVIEW` actions index `PREVIEW_ORDERS[360][4]`, the ordered 4-tuples of sheet indices.
2. **Apply** (PS `chooseTeam` + `runAction('team')`):
   - `brought_sheet_idx[s] = order`, `ps_order[s] = [0,1,2,3]`, `pokemon_left = 4`;
   - build `Mons[s, b]` from the sheet:
     - stats from the Champions formula (§4.1);
     - `can_mega = dex.mega_of(species, item) != 0`;
     - PP from `move_core.pp_max`;
     - ability and item from the sheet;
     - `newly_switched = True` (PS clearVolatile at construction; RF #22);
     - `speed_cached` = the raw stored Spe (setSpecies).
3. **Start** (`runAction('start')`):
   - leads b = 0, 1 enter slots a and b on both sides;
   - **one batched SwitchIn resolution** over all entering mons (§10.2);
   - then the empty-queue tail;
   - then **`end_turn()`**, which sets `turn = 1`, gives leads `active_turns = 1` and `newly_switched = False`, and computes disables and trapping (RF #22, verified);
   - `phase = MOVE`.
4. **Gender.** `Team.gender` is always set, so construction makes no PRNG draw.

---

## 4. Stats, legality masks, requests

### 4.1 Champions stats (CH §1.3)
```
HP = base_hp + sp_hp + 75
X  = base_x + sp_x + 20 ;  nature+: trunc(trunc(X*110, 16)/100) ; nature−: trunc(trunc(X*90, 16)/100)
```
- Level 50 enters damage (`2·50/5 + 2 = 22`) and Seismic Toss / Night Shade, but not stats.
- Mega and forme changes recompute stats from SP and nature. HP is unchanged when base HP is equal.
- `baseMaxhp ≡ maxhp`, because there is no Dynamax.

### 4.2 Effective stat (`getStat`, MV §3.2, §5.2)
- **Stages:** integer floor: `stat*(2+n)/2` for n ≥ 0, otherwise `stat*2/(2−n)`.
- **Chain:** then the `ModifyX` hook chain (MUL), then `finalModify`.
- **Speed:**
  - paralysis `floor(spe*50/100)` after `finalModify` (MV §7.4 #6);
  - cap at 10000.
- **Crits** ignore the attacker's negative offensive stages and the defender's positive defensive stages.

### 4.3 Action encoding and legal masks (`battle/requests.py`; TL §4.9; ADR-018)

**Encoding (single source of truth; ADR-018 and 08 §3 reference this):**
- Move actions: `a = 6·m + 2·t + mega` (a < 24), with m ∈ 0..3, t ∈ {0,1,2}, mega ∈ {0,1}.
- Target t → PS signed location:
  - 0 → +1 (foe slot a);
  - 1 → +2 (foe slot b);
  - 2 → the ally's location.
  - For `adjacentAllyOrSelf`: t = 0 → **self**, t = 2 → ally.
  - Non-choosable target types allow only t = 0.
- `24 + b` = switch to (or revive) brought index b. `28` = pass.
- **Locked** (lockedmove, twoturnmove, mustrecharge) **and Struggle:** the only move action is `a = 0`, which maps to `move 1` with no target and no mega.

**MOVE phase, per side, per slot k with occupant P:**
1. **P fainted** (still in the slot, §10.1) or the slot is empty: only `28`.
2. **P locked:** only `0`. No switch.
3. **Otherwise:**
   - **Usable moves:** `j` with `pp > 0` and `move_disabled[j] == 0`. Hidden disables count as disabled.
   - **Struggle:** if no move is usable, the only move action is `0`.
   - **Targets:** from the move's target type after Curse ([C], non-Ghost → self) and Pollen Puff / [C] Milk Drink adjustments:

     | Target type | Legal t |
     |---|---|
     | `normal`, `any` | {0, 1, 2} |
     | `adjacentFoe` | {0, 1} |
     | `adjacentAlly` | {2} |
     | `adjacentAllyOrSelf` | {0, 2} |
     | others | {0} |

     Confirm against PS acceptance (07 §6).
   - **Mega:** allowed iff `P.can_mega ∧ ¬mega_used[side]`, and the move isn't locked or Struggle.
   - **Switch** `24 + b`: requires `¬trapped(P)` (true trapping, including hidden) and that b is a live bench mon not targeted by a queued switch.
4. **Joint constraints (`pair_ok[29,29]`):** not both mega; not both switching to the same b. Nothing else.

**SWITCH phase (RF #20):**
- **B** = live bench mons not targeted by a queued switch or instaswitch (PS seeds `switchIns` from `queue.getSwitches`).
- **#flagged** counts every flagged slot, including a Revival Blessing slot.
- **Non-revival flagged slots** choose from `{24+b : b ∈ B} ∪ {28}`, with exactly `min(#flagged, |B|)` switches overall, all distinct.
- **Revival slot** chooses from `{24+b : b fainted, any position, including a fainted ally still in the other slot} ∪ ({28} iff |B| < #flagged)`. The revival choice does not count toward switch distinctness.
- **Unflagged slots:** `28`.

**PREVIEW:** all 360 are legal. The wrapper may restrict them (08 §3).

**Disabled recomputation** at request time (PS `endTurn` → `DisableMove`; TL §8.4):
- choice lock, Encore, Taunt, Disable, Torment, Throat Chop, Heal Block, Gravity, Imprison, `cantusetwice`;
- [C] Fake Out / First Impression once `active_move_actions > 0` (incremented at the **start** of runMove; §7.1);
- [C] Belch and Stuff Cheeks are always selectable.

The mask must equal PS *acceptance* (07 §6).

---

## 5. Queue (TL §5)

### 5.1 Order classes
| Order | Class |
|---|---|
| 3 | instaswitch; prioritized (After You / Instruct / Round) |
| 4 | beforeTurn |
| 5 | beforeTurnMove (Counter, Mirror Coat) |
| 6 | revivalblessing |
| 101 | runSwitch |
| 103 | switch |
| 104 | megaEvo |
| 107 | priorityChargeMove (Focus Punch, Beak Blast, Chilly Reception) |
| 200 | move |
| 201 | quashed move |
| 300 | residual |

### 5.2 `commit_move_choices` = `commitChoices` + `resolveAction(midTurn=false)` (TL §5.2–5.3)
1. `update_speed()` for all non-fainted actives (§5.4).
2. For each non-pass slot action, create entries `[beforeTurnMove | priorityChargeMove]`, `[megaEvo if mega]`, `move`.
   - Roll **fractional priority once** (`FRAC_PRIO`: Quick Claw 1/5; Quick Draw 3/10 for non-Status moves; Stall −0.1).
   - `randomNormal` moves draw **nothing** at commit. Their execution-time draw is the one that matters (RF #14).
3. Switch entries take the outgoing mon's speed. megaEvo takes the evolver's commit-time speed.
4. Add the beforeTurn and residual entries.
5. Order-<200 entries keep commit-time keys.
6. All new entries have `prepend = False`.

### 5.3 Priority
- `priority10 = 10 * modified_priority + frac10`.
- `modified_priority` = base move priority (never compounded) → `ModifyPriority`: Grassy Glide (singleEvent), Prankster, Gale Wings (full HP **now**). Triage is not legal.
- Write it back as `move.priority` for Quick Guard and Psychic Terrain checks.

### 5.4 Speed and `speed_cached` (RF #10)
- `getActionSpeed(mon)` = `getStat('spe')` (boosts → ModifySpe chain), **negated under Trick Room [C]**, with no wrap.
- **`speed_cached` (PS `pokemon.speed`) is written by:**
  - **updateSpeed:** commit of MOVE **and** SWITCH phases; before each Block-B re-key; start of residual; `insertChoice` for that mon;
  - **setSpecies** (= the raw stored Spe, not negated, no ModifySpe): init; clearVolatile on switch-out and faint; every formeChange including Mega, Stance Change, Busted, Zero to Hero, Hunger Switch; Transform.
  - dragIn never writes it.
- Handler ordering (§6.4) and switch-in ordering (§10.2) read `speed_cached`.
- **Re-key** (before each Block-B move selection): recompute `speed` and `priority10` for every valid move entry, keeping `frac10`. Clear all `prepend` flags.

### 5.5 Selection
- Choose `argmax` over valid entries of `(prepend, −order, priority10, speed)`.
- **Ties** on the full key: the tie set contains only entries whose actor is active and not fainted. Resolve with the forced or natural `SPEED_TIE_SELECT` per `contract/drawkey.md` §4.
- Entries whose actor is fainted or inactive are selected as normal, but **return early**: no state change, no tail, no ordinal advance.

### 5.6 Mutations (TL §5.6, RF #17)
| PS | Engine |
|---|---|
| `prioritizeAction` (After You, Round, Instruct-created) | set `order = 3` |
| Quash | set `order = 201` |
| Instruct | `resolveAction(...)` → [satellite?, move] with a `FRAC_PRIO` re-roll; write **element 0** as an order-3 entry. For Counter, Mirror Coat and Chilly Reception, element 0 is the satellite, so the move itself never runs (verified) |
| [C] Encore `changeAction` | `cancelAction(p)`; `update_speed(p)`; resolve the encored move with a `FRAC_PRIO` re-roll, satellites (Focus Punch / Chilly Reception → order-107 entry; Counter / Mirror Coat → order-5 entry), `RANDOM_TARGET` at resolve (ord = resolve) and recomputed priority; insert by key (`QUEUE_INSERT_TIE` is don't-care) |
| `cancelAction(p)` | invalidate p's entries except runSwitch |
| SWITCH-phase commit | new instaswitch entries get `prepend = True` (§10.3) |

---

## 6. Hook framework (`effects/hooks.py`; MV §1, §1.13, §7.2)

### 6.1 Hook points
Implement MV §7.2 H1–H32. Each call site calls:
```
result = run_hook(H, ctx, target, source, relay, mode)    # mode ∈ {RUN_EVENT, SINGLE_EVENT, PRIORITY_EVENT}
```
- `ctx` is the event-context pytree: the ActiveMove's mutable copy (bp, type, category, flags, hit, crit, spread flag, ignore flags, …), the target set, the source and the effect.
- Hooks return `(relay', ctx', state')`.

### 6.2 Holder enumeration (static shape)
- **Candidates per `runEvent(E, target t, source s)`** (MV §1.13):
  - t's status / volatiles / ability / item / slot conditions (`on`);
  - each living ally of t **including t** (`onAlly`, `onAny`);
  - each living foe (`onFoe`, `onAny`);
  - s (`onSource`);
  - side conditions (`on` for t's side, `onFoe` for the other, `onAny` both);
  - field.
- **Inactive target** (RF #9d): if t is inactive and s isn't active, gather only field (and format) handlers.
- **Build** a fixed candidate tensor `[K_H]`:
  1. from each mon's compact active-effect list, `nonzero(unpack_bits(vol_bits), size=K_VOL_ACTIVE)`;
  2. filtered by static `has_hook` bitmaps;
  3. compacted by `nonzero(size=K_H)`.
  
  Overflow sets a flag.

### 6.3 Rows vs bespoke
- **MUL rows:** `m4096` + a predicate from a closed library (type, category, flag, HP fraction, weather, terrain, grounded, `typeMod > 0`, crit, has type, move ∈ set, first turn, ally present, …). Fold with `chain_modify` in sorted order.
- **GATE rows:** a predicate + a result code {pass, false, null, NOT_FAIL, 0} + an optional side effect from a closed set.
- **REPL rows:** a constant replacement.
- **DURATION rows:** duration-only.
- **Bespoke:** `lax.switch(local_idx[H, effect], BRANCHES_H, ctx, state, dex)`. Branch 0 is identity.
- **Classification:** `tools/datagen/handler_classification.json` (05 §3).

### 6.4 Ordering (RF #9)
Three comparators, as in PS (`sim/battle.ts:409-431, 796-802, 1001-1007`):
1. **Left-to-right events** (Invulnerability, TryHit, DamagingHit, EntryHazard): `(order ↑, priority ↓, target index ↑)`. Stable, no RNG.
2. **priorityEvent** (RedirectTarget, WeatherModifyDamage, LockMove, …): `(priority ↓, speed ↓, holder ability effectOrder ↑)`. Stable in collection order, no RNG.
   - This is deterministic and **not** a don't-care. Example: two equal-speed Lightning Rod holders redirect to the older ability state (`ability_seq`).
3. **All other events:** `(order ↑, priority ↓, speed ↓, subOrder ↑, effectOrder ↑)`.
   - effectOrder = the state `seq` **only** for `*SwitchIn` / `*RedirectTarget` callbacks; 0 otherwise.
   - Exact ties are PS shuffles. Handle them per `contract/drawkey.md` §5:
     - fieldEvent (Residual, SwitchIn) ties: forced kind `SPEED_TIE_FIELD_EVENT`;
     - eachEvent('Weather') ties: forced;
     - MUL-chain ties with ≥ 2 tied contributors in a chain of ≥ 3 factors: forced `SPEED_TIE_HANDLER_ORDER`;
     - other runEvent / eachEvent ties: canonical (don't-care).

The static parts come from `dex.effect_event_order`. Speed is `speed_cached`.

### 6.5 Folds
- **MUL:** `mod = 4096`, then `chain` over the sorted contributors. `relay = modify(relay, mod)` only if relay is a non-negative integer. Handlers that `return this.modify(...)` are REPL at their position (MV §7.4 #5).
- **GATE:** the first falsy result stops the fold. `false`, `null`, `''` and `0` are distinct (MV §1.9).
- **PRIORITY_EVENT:** the first non-undefined value wins.

### 6.6 Suppression
- Mold Breaker and family: a breakable ability on a target ≠ attacker, while the move ignores abilities;
- `ignoringAbility` (Gastro Acid), `ignoringItem` (Klutz, Embargo, Magic Room);
- Air Lock / Cloud Nine;
- fainted holders;
- `isStarted`.

See MV §1.11.

### 6.7 Compile hygiene
`lax.optimization_barrier((relay, ctx))` after each hook. Switches are built only from `EngineConfig.handlers`.

---

## 7. Move execution (MV §2; CH §1.9–1.10; RF #3–6)

### 7.1 `run_move_action(entry)`: runAction → runMove → useMove (PS order)
1. **runAction:** skip silently if the actor is fainted or inactive (early return).
2. **runMove** (`sim/battle-actions.ts:267-365`):
   1. **`active_move_actions += 1` first** (PS :274). This also counts actions that later fail.
   2. `getTarget(user, move, target_loc)`. **randomNormal moves always draw `RANDOM_TARGET` (ord = getTarget) here**, as do retargets from a fainted chosen foe. This happens **before** the BeforeMove gates.
   3. OverrideAction / LockMove (H2): lockedmove (`LOCKEDMOVE_DURATION` at start), twoturnmove, mustrecharge.
   4. **BeforeMove gates (H1)**, ordered by generated `effect_event_order`; do not hand-order them. Reference order by priority:

      | Priority | Gate |
      |---|---|
      | 11 | recharge |
      | 10 | sleep (decrement `time`, Champions sleep) / freeze (§7.6) |
      | 9 | Truant |
      | 8 | flinch |
      | 7 | Disable |
      | 6 | Gravity / Heal Block / Throat Chop |
      | 5 | Taunt |
      | 3 | confusion (`CONFUSION_SELFHIT` + confusion damage roll) |
      | 2 | attract (`INFATUATION`) |
      | 1 | paralysis (`PARALYSIS` 1/8 [C]) |
      | 0 | choice lock |

      A failure sets `move_this_turn_result` per PS and ends the action (the tail still runs).
   5. `beforeMoveCallback` (Focus Punch lost focus, etc.).
   6. `last_damage = 0`.
   7. Base PP deduction (skipped if locked).
   8. moveUsed: `last_move`, `move_this_turn`, `last_move_target_loc`.
3. **useMove / useMoveInner** (`:429-610`):
   1. ModifyTarget.
   2. ModifyType / ModifyMove (H3): the -ates including [C] Dragonize, Weather Ball, Terrain Pulse, Expanding Force target change, Sheer Force flag, ….
   3. `getMoveTargets` (retarget; RedirectTarget H4 for single-target moves only).
   4. **Pressure PP** (H5) over the final target set.
   5. TryMove / PrepareHit gates (H6):
      - Fake Out first-turn `onTry` (tests `active_move_actions > 1`);
      - Protect's `StallMove` (`PROTECT_STALL` with PS counter semantics) and `willAct` (Protect fails if the user acts last);
      - Follow Me doubles-only, Aurora Veil needs snow, ….
   6. Then `tryMoveHit` (self/side/field) or `trySpreadMoveHit` (§7.3).
4. **After useMove:** AfterMoveSecondarySelf (Life Orb recoil and Shell Bell, both suppressed by Sheer Force), then attacker EmergencyExit.
5. **Back in runMove:** AfterMove, then `resolveFaints` + checkWin.

### 7.2 Targeting (MV §2.3, TL §9)
- **Positions** are PS signed locations relative to the user. All positions are adjacent in doubles.
- **`getMoveTargets`:**
  - `allAdjacentFoes` → living foes in slot order;
  - `allAdjacent` → **ally first**, then foes;
  - single target → the location's occupant. A fainted foe means retarget (`RANDOM_TARGET`); a fainted ally means the move fails (the fainted ally is still in its slot, §10.1).
- **Spread flag** is decided from the target list **before** Protect, immunity and accuracy filtering (MV §7.4 #1).
- **Redirection:** Follow Me, Rage Powder (powder: Grass and Overcoat are immune), Lightning Rod. Stalwart, Propeller Tail and Snipe Shot ignore it. Equal-speed holders resolve by `ability_seq` (§6.4 rule 2).
- **Ally Switch:** swap `active[s,0] ↔ active[s,1]`, `position` and `ps_order` per `swapPosition`. The Gen 9 consecutive-use chance draws `ALLY_SWITCH_STALL`.

### 7.3 `trySpreadMoveHit` (vectorised over targets j ∈ {0,1,2}, masked)
**Steps before the hit loop** narrow `is_target`:
1. **Invulnerability (H7).**
2. **TryHit (H8, left-to-right).** Covers:
   - the Protect family and its punish effects;
   - Wide Guard, Quick Guard, Psychic Terrain;
   - absorb abilities, Magic Bounce (bespoke re-useMove), Good as Gold, Bulletproof, Soundproof, Telepathy, Queenly Majesty / Dazzling / Armor Tail.
   - **[C] Unseen Fist / Piercing Drill** break protection at ×0.25 with no punish effect.
3. **Type immunity (H9):** chart, Levitate, Air Balloon, Magnet Rise, Gravity, Iron Ball, Smack Down, Thousand Arrows, Scrappy / Mind's Eye.
4. **TryImmunity (H10):** powder vs Grass/Overcoat, Prankster vs Dark, move-specific immunities, OHKO rules.
5. **Accuracy (H11), for all targets:**
   - `accuracy: true` moves draw nothing;
   - numeric-accuracy moves draw `ACCURACY = randomChance(acc′, 100)`, which is degenerate only when the final acc′ ≥ 100;
   - acc′ applies stages (accuracy − evasion, clamped ±6, `[3+n]/3`) and the ModifyAccuracy / Accuracy hooks;
   - OHKO: 30 + level difference, plus the Sheer Cold rule.
6. **BreakProtect** (Feint), **StealBoosts** (Spectral Thief).

**Hit loop** (Champions `hitStepMoveHitLoop`):
- **Hit count:** `MULTIHIT_COUNT` for 2–5 hits (`sample([2×7, 3×7, 4×3, 5×3])`), or Parental Bond.
- **Per hit h:**
  1. For multiaccuracy moves, hits ≥ 2 re-roll `ACCURACY` (keyed by hit). A miss silently ends the loop.
  2. **spreadMoveHit** (Champions):
     1. the move's own TryHit on targets[0];
     2. **TryPrimaryHit / Substitute (H12)**: Substitute intercepts *before* damage;
     3. **getSpreadDamage** per target (§7.4: CRIT draw, BasePower, …, DAMAGE_ROLL);
     4. **spreadDamage** per target: Damage hook H21 (Focus Sash, Sturdy, Disguise; Focus Band draws `ITEM_FOCUS_BAND` on every damage instance), apply HP, **drain per target per hit**, rounded (§7.8);
     5. **runMoveEffects (H22):** boosts / status / volatile / side / field / weather / terrain; bespoke `onHit` (~72 status-move handlers);
     6. **selfDrops** (`SELF_EFFECT_ROLL` is don't-care when `self.chance` is undefined);
     7. **secondaries (H25)**, per target per secondary ord: `SECONDARY` = `random(100) < chance`. Chance undefined is don't-care; chance ≥ 100 still draws and is compared. Serene Grace doubles chance; Shield Dust filters. Tri Attack / [C] Dire Claw pick `SECONDARY_STATUS_PICK`;
     8. **forceSwitch (H27)**: sets `force_switch_flag` (drags happen in the action tail, §7.9);
     9. **DamagingHit (H26, left-to-right)**: Rocky Helmet, Rough Skin, Static / Flame Body / … (`ABILITY_ONHIT_PROC`), Justified, Stamina, …;
     10. **AfterHit per target**, which runs **even if the attacker fainted** [C];
     11. attacker EmergencyExit.
  3. Update.
  4. Stop on target faint as PS does.
- **After the loop:**
  - resolveFaints;
  - **recoil** (rounded, §7.8; Rock Head blocks it);
  - gotAttacked / `times_attacked`;
  - Update;
  - **AfterMoveSecondary**, which [C] **Sheer Force does not suppress**;
  - per-target EmergencyExit;
  - Eject Button: at most one per action, and [C] it doesn't cancel the attacker's pivot (§7.7).

### 7.4 Damage (MV §3; verified reference MV §3.6; RF #26)
```
if damageCallback: dmg = callback(...)            (Seismic Toss/Night Shade 50, Super Fang, Endeavor, Final Gambit, Counter/Mirror Coat/Metal Burst, …)
bp = basePowerCallback ? cb : move.bp
critRatio = clamp(ModifyCritRatio(move.critRatio), 0, 4)          # move.critRatio default 1
crit = willCrit ?? (critRatio > 0 ? CRIT draw randomChance(1, [24,8,2,1][critRatio-1]) : false)
crit = CriticalHit gate (Battle/Shell Armor veto AFTER the draw)
bp = BasePower hook (H14: MUL rows + bespoke); if bp == 0: no damage
atk, def = stats with stages (crit rules), ModifyAtk/SpA (H15), ModifyDef/SpD (H16), overrides (Body Press, Foul Play, Psyshock…)
base = trunc(trunc(trunc(trunc(2*50/5+2) * bp * atk / def) / 50)) + 2        # exact nesting per MV §3.6
base = spread ? modify(base, 3072) : (pbond_hit2 ? modify(base, 1024) : base)
base = WeatherModifyDamage (H17; priorityEvent; [C] Mega Sol user-sun)
base = crit ? trunc(base*3/2) : base
base = trunc(trunc(base * (100 - roll)) / 100)          # DAMAGE_ROLL ∈ 0..15
base = STAB modify(base, 6144 | ModifySTAB e.g. 8192)   # no Tera
base = type effectiveness: per factor ×2 or trunc(/2)   # Effectiveness hook H19
base = burn (physical, not Guts, not Facade): modify(base, 2048)
base = ModifyDamage chain (H20): screens 2732 doubles (crit ignores; Infiltrator), Life Orb 5324, Expert Belt, Multiscale, Filter/Solid Rock, Friend Guard, resist berries…
base = [C] protect bypass ? modify(base, 1024) : base
base = max(base, 1) ; base = trunc(base, 16)
```
Golden tests: `research/ps-move/damage-examples.js` (16 rolls × 4 examples) plus generated cases.

### 7.5 Boosts (MV §5)
`ChangeBoost` (Contrary, Simple) → `TryBoost` (Clear Body family, Mirror Armor, Flower Veil, Intimidate blockers) → clamp → `AfterEachBoost` (Defiant, Competitive, Rattled) → `AfterBoost`. Per-target sequential resolution (PS; ADR-019 flag).

### 7.6 Status and volatiles (MV §4; RF #31)
- **Admission gates:** Immunity / SetStatus / TryAddVolatile: Safeguard, Misty / Electric terrain, Insomnia, Vital Spirit, Limber, Own Tempo, Leaf Guard, Flower Veil, Sweet Veil, Purifying Salt, type immunities.
- **Reactions:** AfterSetStatus (Synchronize); Lum / Cheri via Update.
- **Sleep [C]:** `startTime = sample([2,3,3])` (`SLEEP_DURATION`) with PS `time` semantics.
- **Freeze [C]** at BeforeMove:
  - if the move has `defrost` (Burn Up only for Fire types): no-op;
  - else `time −= 1`;
  - if `time ≤ 0`: thaw with no draw;
  - else draw `FREEZE_THAW = randomChance(1,4)`.
- **Paralysis [C]:** 1/8.
- **Other volatiles:** toxic stage at residual; confusion `CONFUSION_DURATION`; partial trap `PARTIAL_TRAP_DURATION` (Binding Band; Grip Claw is not legal).

### 7.7 Switch flags (RF #27)
- **`selfSwitch`** (U-turn, Volt Switch, Flip Turn, Parting Shot only if its drop succeeded, Baton Pass, Shed Tail, Chilly Reception) sets `switch_flag` to the source move.
- **[C] Eject Button** sets `switch_flag = 1` on the holder. It is **skipped if any active already has `switch_flag == 1`** (from an earlier Eject Button or Emergency Exit in the same action). A pivot's move-valued flag doesn't block it. At most one per action, and it does not cancel the attacker's pivot.
- **[C] Emergency Exit** sets it when HP crosses 1/2, without clearing other flags.
- **Faint** sets `switch_flag = 0` (§11.1), so a KO'd U-turn user doesn't pause.
- **Red Card and `forceSwitch` moves** set `force_switch_flag`. The drag happens in the tail's phazing loop (§7.9).

### 7.8 Rounding of recoil, drain and Struggle (RF #5)
`round` is JS `Math.round` (round half up for positive values): `(2·x·n + d) // (2·d)`.
- **Recoil** = `max(1, round(total_damage·n/d))` via `damage`. Rock Head blocks it.
- **Struggle** = `max(1, round(baseMaxhp/4))` via `directDamage` (no Damage event).
- **Drain** = `round(dmg·n/d)` per target per hit.
- **Mind Blown / Steel Beam** = `round(maxhp/2)`.

Golden tests include odd values.

Two-turn moves (Solar Beam / Blade in sun, Electro Shot in rain, [C] Mega Sol forcing Electro Shot's charge, Meteor Beam, Phantom Force, Dig, Dive, Fly, Bounce) use `twoturnmove`.

### 7.9 `post_action` (PS runAction tail; TL §6.3; RF #8, #27)
Runs after every action that didn't return early (light actions included), in order:
1. **Phazing:** for each active in p1a, p1b, p2a, p2b order with `force_switch_flag`, `dragIn` (`FORCED_SWITCH_PICK`, subject = the running action's move), each with an inline runSwitch.
2. `resolveFaints` + `checkWin` (§11.1).
3. End-of-queue faint flags.
4. **Update** and the **Emergency Exit check** (only for the first runSwitch mon after a batched SwitchIn; TL §11.6 #12). Both are **skipped when the next queue entry is an instaswitch, and after `start`**.
5. **Switch requests:**
   - sides with flagged slots but no switchable bench get their flags cleared, except Revival Blessing;
   - if any flag remains, **pause**.
6. **Re-sort** if the next entry is a move. It happens at the next Block-B iteration via `rekey_moves`.

---

## 8. RNG (ADR-012; normative key in `contract/drawkey.md`)

- **Words.** Each step derives `w0, w1` from `fold_in(key, step_counter)`. Every random decision is `hash32(w0, w1, pack(DrawKey))`, where `hash32` = 2× MurmurHash3 `fmix32`, all uint32 (`contract/drawkey.md` §2). It is never positional.
- **Distributions:**
  - `randomChance(n,d)`: `hash % d < n`;
  - `random(a,b)`: `a + hash % (b−a)`;
  - `sample(list)`: `list[hash % len]`;
  - tie permutations: sort candidates by per-candidate hash.
- **Draw call contract:**
  - `value, state = draw(key_fields, dist_args, forced_table, log)`.
  - **Forced mode:** exact key match. A miss on a non-don't-care kind sets `ERR_FORCED_MISSING` and uses the natural value.
  - **Draw-log mode:** append `(key, dist_args, value, used_forced)`.
  - Both modes are compiled out when disabled.
- **Draw kinds, key fields per kind, value domains, tie records, step_tag mapping and the don't-care / degenerate lists:** all in `contract/drawkey.md`. They are not restated here.

---

## 9. Conditions (MV §4; TL §8)

### 9.1 Storage
Presence is a bitfield (03). State holds PS's `duration` and counters verbatim, `seq` (PS effectOrder at creation) and `source`.

### 9.2 Lifecycle
- **`add`:**
  - if present → Restart (Trick Room toggles off, Stockpile stacks, Spikes ≤ 3);
  - else admission gates → `duration` = default or `durationCallback` (rocks, Light Clay, Terrain Extender) → `seq = effect_seq++` → Start.
- **`remove`:** End, then clear.

### 9.3 Duration semantics
`duration` decrements only in the residual walk, at the effect's position. An effect set on turn T with duration N lasts through turn T+N−1. A weather's last turn has no chip.

**Duration-only one-turn effects end inside the residual walk** (sorted last), not in end_turn (RF #7). Examples:
- the Protect family, stall, Helping Hand, Follow Me / Rage Powder, Wide/Quick Guard, flinch, Roost;
- lockedmove / twoturnmove / mustrecharge where applicable.

So they are already gone at the end-of-turn SWITCH decision.

### 9.4 Residual walk (`fieldEvent('Residual')`; RF #25)
1. **Collect** candidates `(effect, holder, order, priority, subOrder, speed_cached, effectOrder = 0)` over:
   - field weather, terrain and pseudo-weathers;
   - both sides' side conditions, and slot conditions per position;
   - **every non-null slot occupant, including fainted ones**: status, volatiles, ability, item.
   
   Include per-mon copies of field/side residual handlers (e.g. Grassy Terrain heal, which runs at that mon's speed and does not tick the duration). Include duration-only effects at order = last.
2. **Compact** to `K_RES` (derived in datagen; overflow flag).
3. **Sort** with comparator (iii) of §6.4. Exact-key tie groups use `SPEED_TIE_FIELD_EVENT`.
4. **Iterate.** For each candidate:
   - **skip** if the holder is fainted (unless it is a slot condition) or the handler is stale;
   - **if the handler has `end` and a duration:** `duration −= 1`; at 0, call End and skip the callback; otherwise run the callback;
   - **otherwise** run the callback (row or bespoke);
   - after **each** handler: `resolveFaints` + `checkWin`.
5. **Static order.** Generate it from `dex.residual_table` (TL §8.2, MV §4.3). Do not hand-maintain example lists.
6. **Weather chip** (sandstorm) runs inside the weather handler via `eachEvent('Weather')`, in speed order. **Its ties are forced** (`contract/drawkey.md` §5), because faint order can decide the winner.

---

## 10. Switching (TL §2.7–2.8, §6.4)

### 10.1 Slot occupancy (RF #19)
A fainted mon **stays in its slot** (`active[s,k]` keeps it, `is_active = False`) until a switchIn replaces it. Targeting a fainted ally returns it, so the move fails. Residual collects its slot conditions. Revival Blessing's in-slot path relies on it.

### 10.2 `switch_in(side, slot, b, source)` and `runSwitch`
1. **Outgoing occupant:**
   - `cancelAction`;
   - SwitchOut hooks (Regenerator, [C] Natural Cure, Zero to Hero) — these fire only if it is active;
   - ability and item End;
   - `clearVolatile` (resets `times_attacked` [C]; Baton Pass / Shed Tail copy volatiles and boosts; setSpecies writes `speed_cached` = raw Spe);
   - a fainted outgoing mon's status `fnt` → `''` (RF #18).
2. **Party order:** swap per PS (the incoming mon takes the outgoing mon's array index).
3. **Incoming mon:** `active`, `position`, `is_active`, `active_turns = 0`, `active_move_actions = 0`, `newly_switched`, `previously_switched_in += 1`. Stamp `ability_seq`, `item_seq` (ability/item effectOrder).
4. **runSwitch:**
   - The speed sort runs over **all slot occupants, including fainted ones**, by `speed_cached` at **every** runSwitch: batched, single start-of-turn switch, drag, start. One record is emitted per tie group (`SPEED_TIE_SWITCHIN`, `contract/drawkey.md` §4).
   - Then entry hazards (Stealth Rock, Spikes, Toxic Spikes, Sticky Web) and SwitchIn handlers (Intimidate, weather/terrain setters, Unnerve, Trace, …) in PS order.
   - **Batched** (leads, EOT replacements, mid-turn instaswitches): all enter first, then one SwitchIn fieldEvent. **Not batched:** ordinary start-of-turn switches; each one's runSwitch runs before the next switch.

### 10.3 SWITCH-phase commit
1. `update_speed()` for actives.
2. Build instaswitch entries (order 3, `prepend = True`). Their speed is the **outgoing** mon's `getActionSpeed()` at commit. For a fainted outgoing mon that is the stored Spe negated under Trick Room; boosts are already cleared (RF #10).
3. They precede the untouched remainder (prepend flag; §5.5). Their runSwitch entries are inserted by key, so they sort **behind** any order-3 prioritized move left in the remainder (§2.1).
4. End-of-turn replacements follow the same path. After they complete, `checkFainted` runs again (hazard KOs), allowing up to 2 more SWITCH loops per side.

### 10.4 Revival Blessing re-entry (RF #16, verified)
- **Revive.** The REVIVAL action heals the chosen fainted mon (`|-heal|`).
- **In-slot target** (the fainted ally still occupying a slot): it **appends** an INSTASWITCH entry *after* RESIDUAL, with `prepend = False` (PS `addChoice` pushes without sorting).
  - **If the next head is a move,** the usual re-sort brings the instaswitch to the front, and it runs before the remaining moves.
  - **Otherwise** Block C runs the residual first. Its tail skips checkFainted, Update and the residual Emergency Exit, because the next action is an instaswitch. Then INSTASWITCH + RUN_SWITCH run. The switchIn has outgoing == incoming: SwitchOut fires no handlers because the target is inactive, then ability/item End, `cancelAction` on itself, clearVolatile. Then the empty-queue tail and end_turn.
- **Bench target:** the mon simply becomes non-fainted on the bench.

---

## 11. Faint, win, end of turn

### 11.1 Faint queue (RF #11)
- **`faint(mon)`:**
  - no-op if already fainted or queued;
  - else `hp = 0`, `switch_flag = 0`, `faint_queued = 1`, and push onto `faint_queue` (FIFO, 03 §4.1).
  - Fainting does not clear status.
- **`resolveFaints`** pops in order: `fainted = True`, `pokemon_left −= 1`, `total_fainted += 1`, `fainted_this_turn = True`, clearVolatile (setSpecies → `speed_cached` raw).
  - [C] A Mega persists.
  - The mon **remains in its slot**.
- **`checkFainted`** sets `status = fnt` on fainted actives of sides that can switch.
- **Win:** a side with 0 mons left loses. **On a double wipe, the side of the last popped entry wins**, even if that entry was skipped as already fainted.
- **Turn cap:** `cfg.max_turns` with `turn_cap_outcome` (08 §5). Conformance uses 1000 (PS's hard tie).

### 11.2 Mid-turn and end-of-turn switch requests
See §2.1 pause semantics and §10.3.

### 11.3 `end_turn` (PS `endTurn`, `sim/battle.ts:1640-1790`, in PS order; RF #22)
1. `turn += 1`.
2. **Per-mon loop** over **all slot occupants, including fainted ones**, with only `active_turns` skipping fainted mons:
   - `move_last_turn_result ← move_this_turn_result`, then `move_this_turn_result = undefined`;
   - `move_this_turn = none`;
   - `newly_switched = False`;
   - when `turn != 1`: reset `used_item_this_turn`, `stats_raised_this_turn`, `stats_lowered_this_turn`, `hurt_this_turn = −1`;
   - `active_turns += 1` if not fainted;
   - prune `attacked_by` (clear `this_turn`, drop entries whose source left);
   - recompute `move_disabled` (DisableMove) and `trapped` (TrapPokemon / MaybeTrapPokemon: Shadow Tag on Mega Gengar, Mean Look, Block, Jaw Lock, Octolock, Spirit Shackle, partial trap; [C] Run Away and Shed Shell exempt).
3. Side and battle: `fainted_last_turn ← fainted_this_turn`, `fainted_this_turn = False`, `last_successful_move_this_turn = none`. Turn scratch is reset, except the persistent ordinals noted in 03 §4.6.
4. Turn-limit checks (PS 1000 / `max_turns`).
5. `phase = MOVE`.

---

## 12. Static bounds (assert via error flags)

| Bound | Value | Justification |
|---|---|---|
| QMAX | 14 | 12 at commit (4 moves + 4 satellites + 2 megas + beforeTurn + residual) + 1 Encore satellite + 1 margin (RF #28) |
| NA | 10 | turn start: beforeTurn + 4 × (switch + runSwitch) = 9; resume: ≤ 8; + margin (RF #1) |
| NB | 8 | 4 primary moves + 2 Instruct + up to 2 runSwitch entries sorted behind prioritized moves (RF #15) |
| K_VOL_ACTIVE | 16 | volatiles per mon |
| K_RES | derived in datagen | max residual-collectible effects per holder × 4 + side + field; record the value in 05 |
| K_H | 16 | per-hook candidates |
| K_ATT | 8 (ring) | `attacked_by` (RF #23) |
| Faint queue | 8 | |
| Hit loop | 10 | Population Bomb |
| Public events per step | 32 | |
| Forced entries per step | KF = 160 | |

## 13. Mega Evolution (CH §8; TL §0 #5)
1. **Choice:** the `mega` bit. Legal per §4.3.
2. **Queue:** megaEvo entries (order 104) with commit-time speed. They run after switches and before any move, in start-of-turn speed order. Megas are not re-sorted between each other.
3. **`run_mega`** (a light action, followed by `post_action`):
   - `formeChange(mega, permanent)`: species, types, weight; stats recomputed; HP unchanged; `speed_cached` = raw Mega Spe (setSpecies);
   - `base_species = mega`;
   - ability = base_ability = the Mega ability: the old ability's End, then the **new ability's Start immediately**;
   - `can_mega = False` for the whole side, and `mega_used[side] = True`;
   - `move_this_turn_result = True`.
4. **Same turn:** Block B's re-key uses the new speed and abilities.
5. **Persistence:** the Mega survives switching and fainting [C].
6. **Projection identity:** a mon's id is `side:sheet_index`, not `baseSpecies`, which changes on Mega (07 §5).

## 14. Champions delta checklist
Every row of CH §0 (1–20) and CH §12 P0/P1 has a scenario test. They are tracked as catalogue `ENGINE_DELTA` rows (05 §2); entity-level deltas are tracked on the entity rows.

## 15. PS quirks that must be reproduced (consolidated)
- TL §11.6 (all 12 items);
- MV §7.4 (all 10);
- RNG §2.3 (all 7);
- RF items 3–32 as incorporated above;
- `move X mega` on a locked or Struggling mon is accepted and the mega ignored. The mask never offers it, but `ps_choice` tolerates it;
- a crafted state is not a fixpoint: normalise with one `Update` pass (07 §7).
