# PS move execution, damage and event system: implementation-grade spec for a JAX Champions doubles engine

*Research date: 2026-10-21. Target: `[Gen 9 Champions] VGC 2026 Reg M-C` (format id `gen9championsvgc2026regmc`, mod `champions`, doubles).*

**Oracle.** Pokémon Showdown commit `d601443` ("Add help for /setbot").
- The original checkout at `/tmp/psrepo/ps` disappeared partway through this work.
- I re-cloned the same commit (`d601443456c795f77c6e4f2b5b459167f2b10a3c`, clean `git status`) to `/tmp/psrepo-move/ps` and built it there.
  - Source: a local `git clone` of `/tmp/psrepo-delta/ps`, with `node_modules` copied from it. Only `@esbuild/darwin-arm64` was added, and only in my own copy. No other checkout was modified.
- All `file:line` citations are relative to the PS repo root and are identical at this commit.
- **Abbreviation:** `ch/` means `data/mods/champions/`.

**Scope.** Move execution, damage, the event system, conditions, stat stages and accuracy. Turn-level control flow (queue, requests, switching, fainting) is covered by a separate agent. I describe it here only where the move pipeline calls into it.

**Verification artefacts.** Everything is in `research/ps-move/`. Each script uses only the built `dist/` and never modifies PS.

| Script | What it does |
|---|---|
| `harness.js` | Builds a Reg M-C battle with fixed teams. Replaces `battle.prng` with a logging/forcing proxy (each draw is tagged with its calling PS function). Monkeypatches the event dispatchers to log every dispatch. |
| `damage-examples.js` | 4 worked damage examples. Each one runs real PS 16 times (forced damage roll r = 0..15) and compares against the explicit-integer reference in §3.6. **All 64 roll/target cells match.** |
| `intermediates.js` | Prints every intermediate value of the reference formula for those examples. |
| `verify-tables.js` | Checks the integer stat-stage formula against `Pokemon.calculateStat` (s = 1..2000, b = −6..6), and `modify()` against an integer closed form. Also shows that 3-term modifier chains are order-dependent. |
| `trace-turn.js`, `trace-more.js` | Full event-dispatch traces of real doubles turns. Outputs are `trace-turn1.txt`, `trace-start.txt` and `trace-turn-b{1,2,3}.txt`. |
| `rng-order.js`, `smoke.js` | RNG draw order for spread moves, secondaries and self-drops. |
| `hook-census.js` | Counts handlers per event over a dex subset. Writes `hook-census.json`. |

---

## 0. Executive summary (what matters for the JAX design)

1. **The event system has 5 dispatchers.**
   - `singleEvent` calls one effect's one callback.
   - `runEvent` gathers every handler on the target, its allies, its foes, the source, the side, the field and the format. It sorts them and folds a relay value through them, stopping on a falsy value.
   - `priorityEvent` is `runEvent` with `fastExit`: it stops on the first non-`undefined` value and uses a different sort.
   - `eachEvent` runs `runEvent` on every active Pokémon in speed order.
   - `fieldEvent` is the special Residual/SwitchIn walker. It also decrements every `duration` counter.
2. **Numeric modifiers are not applied by handlers directly.**
   - A handler calls `chainModify(x)`, which folds `x` into `event.modifier` in 4096-fixed-point.
   - `runEvent` applies `modify(relay, event.modifier)` once at the end, but only when the relay is a non-negative integer (`sim/battle.ts:936-940`).
   - Chaining is order-dependent for 3+ factors (verified). So a JAX engine must reproduce **handler order** (order → priority → speed → subOrder → effectOrder) for modifier chains, not just the set of modifiers.
3. **Damage formula.** The Champions `modifyDamage` (`ch/scripts.ts:196-312`) is numerically identical to base Gen 9 (`sim/battle-actions.ts:1778-1896`). The only difference is the `-supereffective`/`-resisted` log messages, which carry a 1/2 magnitude argument.

   The order is: `+2` → spread ×0.75 → `WeatherModifyDamage` chain → crit `trunc(×1.5)` → roll `trunc(trunc(d·(100−r))/100)` → STAB (`ModifySTAB`) → type (repeated ×2 or `trunc(/2)`) → burn ×0.5 → `ModifyDamage` chain (screens, Life Orb, Expert Belt, …) → protect-bypass ×0.25 → `0 → 1` → `trunc(d, 16)`.
4. **Champions-specific move-pipeline differences.** There are three overrides:
   - **`spreadMoveHit`.** Move `onAfterHit` (Rapid Spin, Knock Off, …) runs even if the attacker fainted. Base requires `pokemon.hp` (`ch/scripts.ts:411` vs `sim/battle-actions.ts:1180`).
   - **`hitStepMoveHitLoop`.** Sheer Force no longer suppresses `AfterMoveSecondary` and target `EmergencyExit` (Berserk, Pickpocket, Eject Button, Red Card, Emergency Exit …). Base wraps them in `!suppressingSecondaries()` (`sim/battle-actions.ts:1064` vs `ch/scripts.ts:573-585`). Parental Bond also doesn't print `-hitcount` when only one hit landed.
   - **`modifyDamage`.** Messages only.
   - **Elsewhere in Champions:**
     - Unseen Fist breaks Protect for contact moves but deals ×0.25 through it (`ch/abilities.ts:91-100` plus the `bypassProtect` branch of `modifyDamage`).
     - Full paralysis is 1/8 (`ch/conditions.ts:4-9`).
     - Sleep is `sample([2,3,3])` (`ch/conditions.ts:23`).
     - Freeze lasts at most 3 turns, with a 1/4 thaw chance per turn (`ch/conditions.ts:43-55`).
     - PP is `(pp/5+1)*4`, with base PP capped at 20 (`ch/scripts.ts:3-9,41-43`).
     - Trick Room speed is `-speed` with no 10000/13-bit wrap (`ch/scripts.ts:46-54`).
     - Fake Out is disabled after the first action (`ch/moves.ts:349-356`).
     - Stats are `base+SP+20` (HP `base+SP+75`) with nature ×1.1/×0.9 (`ch/scripts.ts:10-40`).
5. **The ruleset has no OHKO Clause and no Evasion Clause** (`ch/rulesets.ts:35-40`; resolved rule table in `research/champions/ruleset_resolved.json`). Sheer Cold, Fissure, Horn Drill and Guillotine, plus Double Team and Minimize, are reachable in the legal pool. So OHKO accuracy and evasion stages must be implemented.
6. **RNG draw order within one move is batched by step across targets** (verified in `rng-order.js`):
   - all accuracy rolls (target order);
   - then per target, crit roll and damage roll;
   - then the self-drop roll;
   - then per target, the secondary rolls.

   PS also makes **non-semantic draws** that a JAX engine should *not* replicate:
   - random target picks for spread moves inside `getActionSpeed`;
   - `selfDrops`' unconditional `random(100)`;
   - speed-tie shuffles inside handler sorts.

---

## 1. The event system

### 1.1 Vocabulary

- **Effect.** Any object that can carry handlers: Move/ActiveMove, Ability, Item, Condition (status, volatile, side/slot/field condition), Weather, Terrain, Species (`speciesState`), Format/Rule.
- **Handler.** An `on<Event>` property on an effect. It may be a function or a constant: a non-function value is returned as-is (`sim/battle.ts:647-651`, `915-917`).
  - Metadata properties are `on<Event>Order`, `on<Event>Priority` and `on<Event>SubOrder` (`sim/battle.ts:959-961`).
- **Holder.** Where the effect lives: a Pokémon (status, volatiles, ability, item, species, slot conditions), a Side (side conditions), the Field (pseudo-weather, weather, terrain) or the Battle (format, rules, `onEvent` registrations).
- **effectState.** A per-instance mutable dict bound to `this.effectState` while the handler runs (`sim/battle.ts:907-909`). Definition: `EffectState {id, effectOrder, duration?, ...}` (`sim/pokemon.ts:34-39`).
  - Pokémon: `statusState`, `volatiles[id]`, `abilityState`, `itemState`, `speciesState`, `side.slotConditions[pos][id]`.
  - Side: `sideConditions[id]`.
  - Field: `pseudoWeather[id]`, `weatherState`, `terrainState`.
  - In `runEvent`, `effectState.target` is overwritten with the holder before every call (`sim/battle.ts:909`).
- **relayVar.** The value threaded through a `runEvent`. It is the first handler argument when supplied; otherwise it is a hidden `true`.

### 1.2 `singleEvent(eventid, effect, state, target, source?, sourceEffect?, relayVar?, customCallback?)` — `sim/battle.ts:578-659`

Algorithm:
1. Guard: `eventDepth >= 8` throws (`583-590`). Log-spam guard (`591-597`).
2. `hasRelayVar = relayVar !== undefined`. Otherwise `relayVar = true` (`599-603`).
3. **Suppression checks.** Each returns `relayVar` unchanged without calling the handler.
   - The effect is a Status and the target's current status differs (stale status) (`605-608`).
   - `eventid==='SwitchIn'`, the effect is a breakable Ability, and `suppressingAbility(target)` holds (Mold Breaker during switch-in) (`609-613`).
   - The effect is an Item, the target is ignoring items, and the event is not `Start`/`TakeItem`/`SetAbility` (`614-618`).
   - The effect is an Ability, the target is ignoring abilities, and the event is not `End` (`619-622`).
   - The effect is Weather, weather is suppressed (Air Lock / Cloud Nine), and the event is not `FieldStart`/`FieldResidual`/`FieldEnd` (`623-629`).
4. `callback = customCallback || effect['on'+eventid]`. If it is `undefined`, return `relayVar` (`631-632`).
5. Save and set `this.effect`, `this.effectState = state || {}`, `this.event = {id, target, source, effect: sourceEffect}`, then `eventDepth++` (`634-641`).
6. `args = [target, source, sourceEffect]`, with relayVar prepended if `hasRelayVar` (`643-644`). Call the handler (`647-651`).
7. Restore the saved values and return `returnVal === undefined ? relayVar : returnVal` (`653-658`).

`singleEvent` does **not** apply `event.modifier`, and its `this.event` has no `modifier` field.
- A `chainModify` inside a singleEvent handler would compute `trunc(undefined*4096) = 0` and write the result to that throwaway event object.
- The value is discarded, so no data file relies on it.
- Numeric changes from singleEvent handlers must be **returned values**. An example is a move's own `onBasePower`, which is instead routed through `runEvent(..., onEffect=true)` so that its `chainModify` lands in the runEvent modifier.

### 1.3 `runEvent(eventid, target?, source?, sourceEffect?, relayVar?, onEffect?, fastExit?)` — `sim/battle.ts:765-944`

```
runEvent(eventid, target=battle, source, sourceEffect, relayVar, onEffect, fastExit):
  if eventDepth >= 8: throw
  effectSource = source if source is Pokemon else null
  handlers = findEventHandlers(target, eventid, effectSource)                     # 784
  if onEffect: handlers.unshift(resolvePriority({effect: sourceEffect,
                 callback: sourceEffect['on'+eventid], state:{}, effectHolder: target}))   # 785-794
  # sort (796-802)
  if eventid in {Invulnerability, TryHit, DamagingHit, EntryHazard}: sort by compareLeftToRightOrder
  elif fastExit:                                                     sort by compareRedirectOrder
  else:                                                              speedSort(handlers, comparePriority)  # may consume RNG
  if relayVar is undefined or null: relayVar = true; hasRelayVar = false             # 806-811
  else: args = [relayVar, target, source, sourceEffect]
  push this.event = {id, target, source, effect: sourceEffect, modifier: 1}          # 813-815
  targetRelayVars = relayVar if (target is array and relayVar is array) else [true]*len(target)   # 817-824
  for handler in handlers:
     if handler.index defined (array target):                                       # 826-835
        skip if !targetRelayVars[index] (except value 0 for DamagingHit)
        args[target-position] = handler.target; if hasRelayVar: args[0] = targetRelayVars[index]
     skip if Status effect and holder.status != effect.id                            # 839-842
     skip if breakable Ability and suppressingAbility(holder)        # Mold Breaker   843-848
     skip if Item and holder.ignoringItem() and event not in {Start, SwitchIn, TakeItem}   # 881-886
     skip if Ability and holder.ignoringAbility() and event != End                  # 887-895
     skip if (Weather effect or eventid=='Weather') and event not in {Residual, End} and field.suppressingWeather()  # 896-902
     returnVal = call(handler) with this.effect/effectState set (effectState.target = holder)    # 903-917
     if returnVal !== undefined:                                                    # 919-932
        relayVar = returnVal
        if !relayVar or fastExit:
           if array target: targetRelayVars[index] = relayVar; break if all falsy
           else: break
        if hasRelayVar: args[0] = relayVar
  pop event
  if relayVar is a number and relayVar == |floor(relayVar)|:                        # 936-940
      relayVar = modify(relayVar, event.modifier)      # applies the chained modifier ONCE
  return targetRelayVars if target is array else relayVar
```

Notes:
- **Dead code.** The Mold Breaker branch at `843-879` contains an `AttackingEvents` list for custom abilities. It is unreachable: the outer `if` already requires `flags.breakable`, and the inner `if (effect.flags['breakable'])` always `continue`s. Mold Breaker therefore suppresses **every** event of every breakable ability held by any holder (target, ally, foe, source), not only "attacking" events.
- **Final modify.** It runs for **any** numeric non-negative-integer relay, not only damage-ish events. Examples are priority values (`+1` → `modify(1, 1) = 1`) and PP-drop values. It is the identity whenever no handler called `chainModify`. Negative or fractional relays (e.g. priority −7, or fractional accuracy from `multiaccuracy`) skip it.
- **Array target** (used by `TryHit`, `DamagingHit`, `AfterMoveSecondary`, `TryHitSide` in FFA). `findEventHandlers` is called per Pokémon and each handler is tagged with `index`/`target` (`1051-1061`). The return is the per-target relay array.

### 1.4 `priorityEvent` = `runEvent(..., fastExit=true)` — `sim/battle.ts:950-955`

- It stops at the **first** non-`undefined` return, even a truthy one (`921`).
- It sorts by `compareRedirectOrder`: priority desc, speed desc, then `abilityState.effectOrder` asc. Order, subOrder and RNG are **not** used (`418-424`).
- Uses:
  - `RedirectTarget` (`sim/pokemon.ts:851`);
  - `WeatherModifyDamage` (`sim/battle-actions.ts:1800`, `ch/scripts.ts:216`). Weather handlers call `chainModify` and return `undefined`, so all of them run and the final modify applies.
  - `LockMove` (`sim/pokemon.ts:971`), `SemiLockMove`.

### 1.5 `eachEvent(eventid, effect?, relayVar?)` — `sim/battle.ts:470-481`

- Takes all active Pokémon, `speedSort`s them by `b.speed - a.speed` (ties shuffled → RNG), and calls `runEvent(eventid, pokemon, null, effect, relayVar)` on each.
- After `eachEvent('Weather')` (gen ≥ 7) it runs `eachEvent('Update')`.
- Used for `Update`, `BeforeTurn`, `Weather`, `WeatherChange` and `TerrainChange`.
- For these 5 event names, **prefixed handlers are not gathered** (`sim/battle.ts:1066`). Only plain `on<Event>` on the target and field/battle handlers run.

### 1.6 `fieldEvent(eventid, targets?)` — `sim/battle.ts:489-575` (only `Residual` and `SwitchIn`)

```
fieldEvent(eventid, targets):
  getKey = 'duration' if eventid=='Residual' else none
  H  = findFieldEventHandlers(field, 'onField'+eventid, getKey)                    # field-level, holder=field
  for side: H += findSideEventHandlers(side, 'onSide'+eventid, getKey)            # side-level, holder=side
     for active in side.active:
        if eventid=='SwitchIn': H += findPokemonEventHandlers(active, 'onAnySwitchIn')
        if targets and active not in targets: continue
        H += findPokemonEventHandlers(active, 'on'+eventid, getKey)               # status/volatiles/ability/item/species/slot
        H += findSideEventHandlers(side, 'on'+eventid, -, customHolder=active)    # e.g. pledge/G-Max side onResidual per mon
        H += findFieldEventHandlers(field, 'on'+eventid, -, customHolder=active)  # e.g. Grassy Terrain heal per mon
        H += findBattleEventHandlers('on'+eventid, getKey, active)
  speedSort(H, comparePriority)                                                    # 512, ties -> RNG shuffle
  while H:
     h = H.shift()
     if h.effectHolder.fainted and not slot condition: continue                   # 517-519
     if eventid=='Residual' and h.end and h.state.duration:                        # 520-528  DURATION TICK
         h.state.duration -= 1
         if h.state.duration == 0: h.end(...); continue          # removeVolatile / removeSideCondition / clearWeather ...
     skip if the state object is no longer installed where expected (removed mid-walk)  # 530-563
     evt = 'Side'+eventid if holder is Side; 'Field'+eventid if holder is Field     # 565-567
     if h.callback: singleEvent(evt, effect, h.state, h.effectHolder, null, null, undefined, h.callback)
     resolveFaints(); if ended: return                                             # 572-573
```

Key properties:
- **Duration-only effects** (e.g. Protect: `duration:1` with no `onResidual`) are still collected because of `(getKey && state[getKey])` in every collector (`sim/battle.ts:1116, 1125, 1133, 1140, 1157, 1178, 1203, 1212, 1220, 1237`). They sort with `order = false → 2^32` (last) unless the effect declares `onResidualOrder` / `onSideResidualOrder` / `onFieldResidualOrder`.
- **Customholder handlers have `end: null`** (`sim/battle.ts:1206, 1215, 1223, 1240`). The per-Pokémon copy of a field/side effect never decrements the duration; only the field/side-level copy does.
- **Faint resolution after every residual handler.** `resolveFaints()` runs after each residual handler (`572`).

### 1.7 Handler gathering

#### `findEventHandlers(target, eventName, source)` — `sim/battle.ts:1049-1109`

Let `P = prefixedHandlers = eventName ∉ {BeforeTurn, Update, Weather, WeatherChange, TerrainChange}` (`1066`).

**Target is a Pokémon** that is active (or the source is active) (`1067-1080`):

| Collected | From | Holder set |
|---|---|---|
| `on<E>` | target | the target itself |
| `onAlly<E>` and `onAny<E>` | every Pokémon in `target.alliesAndSelf()`, i.e. **including the target** | living allies only: `side.allies()` filters `hp>0`, `sim/side.ts:390-396` |
| `onFoe<E>` and `onAny<E>` | every Pokémon in `target.foes()` | living foes only |
| `onSource<E>` | the source Pokémon (if any) | regardless of whether it is active (`1081-1083`) |
| `on<E>` | the target's side (`findSideEventHandlers`) | side |
| `onFoe<E>` | the foe side | side |
| `onAny<E>` | both sides | side |
| `on<E>` | field (pseudo-weathers, weather, terrain) | field |
| `on<E>` | battle (format, rules, `onEvent`) | battle |

- The ally/foe/any/source rows apply only if `P`.
- After the Pokémon-level collection, `target = target.side`. The side-level rows come from the Side branch with `shouldBubbleDown = false` (`1084-1105`).
- If the target Pokémon is **inactive and the source is not active**, nothing Pokémon- or side-level is gathered. Only field and battle handlers run.

**Target is a Side** (`shouldBubbleDown = true`, e.g. `TryHitSide`, `SideConditionStart`). Every active Pokémon contributes:
- `on<E>` if on that side;
- `onFoe<E>` if on the other side;
- `onAny<E>` always.

Side conditions contribute as in the Pokémon case.

**Target is the Battle** (`runEvent('X')` with no target). Only field and battle handlers are gathered.

#### Per-holder collectors

| Collector | Order of collection | `end` used by the duration tick |
|---|---|---|
| `findPokemonEventHandlers` (`1111-1170`) | status (`statusState`), each volatile (`volatiles[id]`), ability (`abilityState`), item (`itemState`), species (`speciesState`), each slot condition at its position | `clearStatus`, `removeVolatile`, `clearAbility`, `clearItem`, noop, `removeSlotCondition` |
| `findSideEventHandlers` (`1230-1245`) | each side condition | `removeSideCondition` (null if customHolder) |
| `findFieldEventHandlers` (`1195-1228`) | each pseudo-weather, weather, terrain | `removePseudoWeather` / `clearWeather` / `clearTerrain` (null if customHolder) |
| `findBattleEventHandlers` (`1172-1193`) | format, `battle.events` (from `onEvent`) | none |

#### `getCallback` — `sim/battle.ts:1025-1047`

For `onSwitchIn` on a Pokémon: if an ability or item has no `onSwitchIn` and no `onAnySwitchIn`, its **`onStart` is used as the `onSwitchIn` handler**. This is how Intimidate and the weather/terrain setters fire on switch-in.

#### Move handlers

The active move's own handlers (`onTry`, `onTryHit`, `onBasePower`, `onModifyType`, …) are **not** gathered by `findEventHandlers`. They run as explicit `singleEvent(..., move, ...)` calls, or are prepended via `onEffect=true`:
- `BasePower` (`sim/battle-actions.ts:1704`);
- `ModifyTarget` (`sim/battle-actions.ts:480`).

#### Move conditions

`dex.conditions.getByID(id)` resolves:
- `data/conditions.ts` entries;
- else `Moves[id].condition` / `Abilities[id].condition` / `Items[id].condition`;
- else the `item:`/`ability:` pseudo-conditions (`sim/dex-conditions.ts:674-705`).

So `volatileStatus: 'protect'` uses `moves.protect.condition`.

### 1.8 `resolvePriority` and ordering — `sim/battle.ts:957-1023`

For each handler:
- `order = effect[cb+'Order'] || false`;
- `priority = effect[cb+'Priority'] || 0`;
- `subOrder = effect[cb+'SubOrder'] || 0`.

If `subOrder` is 0, a default is assigned by effect type (`962-999`):

| Effect | Default subOrder |
|---|---|
| (Z-Move) | 1 (comment only) |
| Condition: volatile/status-like | 2 |
| Slot condition | 3 |
| Side condition | 4 |
| Field condition / Weather / Terrain / Format / Rule | 5 |
| Ability Poison Touch, Perish Body | 6 |
| Ability | 7 |
| Item | 8 |
| Ability Stall | 9 |

Other fields:
- `effectOrder = state.effectOrder` **only** for callbacks ending in `SwitchIn` or `RedirectTarget` (`1001-1007`). Otherwise it is undefined (treated as 0).
- `speed = holder.speed` if the holder is a Pokémon (`1008-1010`). For Magic Bounce `onAllyTryHitSide` the unmodified stat is used (`1011-1014`). For `*SwitchIn` callbacks, a fractional tiebreak is subtracted from the pre-sorted `speedOrder` (`1015-1020`).
- Side/Field/Battle holders have `speed` undefined (treated as 0).
- `pokemon.speed` is refreshed by `updateSpeed()` → `getActionSpeed()`. In Champions that is `getStat('spe')`, negated under Trick Room (`ch/scripts.ts:46-54`). Base uses `trunc(10000 - spe, 13)` under Trick Room (`sim/pokemon.ts:662-670`).

#### The three comparators

| Comparator | Lines | Keys | Used by |
|---|---|---|---|
| `comparePriority` | `409-416` | order asc (false → 2^32, last), priority desc, speed desc, subOrder asc, effectOrder asc. **Ties are shuffled with `prng.shuffle`** in `speedSort` (`434-465`) → RNG draws. | `runEvent` default; `fieldEvent`; `eachEvent` uses a speed-only comparator. |
| `compareRedirectOrder` | `418-424` | priority desc, speed desc, ability effectOrder asc; no shuffle. | `priorityEvent` / `fastExit` |
| `compareLeftToRightOrder` | `426-431` | order asc, priority desc, index asc (target position in the array), stable sort, no speed. | `Invulnerability`, `TryHit`, `DamagingHit`, `EntryHazard` |

#### `effectOrder` — `initEffectState`, `sim/battle.ts:3380-3390`

- It is a global counter incremented whenever a state with an `id` and an active/non-Pokémon target is created. Otherwise it is 0.
- It is used only for SwitchIn hazards, RedirectTarget ties (Lightning Rod vs Storm Drain vs Follow Me at equal speed/priority), and Dancer.

#### Practical ordering consequences (verified in `trace-turn1.txt`)

- **Residual interleaves per Pokémon.** Leftovers (order 5, subOrder 4) and Grassy Terrain heal (order 5, subOrder 2) run per Pokémon in speed order: Rillaboom's terrain heal, then Rillaboom's Leftovers, then the slower Incineroar's terrain heal. Speed sorts before subOrder.
- **BasePower chain.** Handlers run in descending `onBasePowerPriority`:
  - Technician 30, Sheer Force 21, type items (Charcoal / Mystic Water) 15, Gem 14, Helping Hand 10, terrains 6.
  - Ex3 verifies that Mystic Water (15) is chained before Helping Hand (10).

### 1.9 Return-value semantics

| Return | Effect inside `runEvent` | Conventional meaning at call sites |
|---|---|---|
| `undefined` | No change; continue (`919`) | "no opinion" |
| `null` | relay = null, stop (falsy) | Silent failure: caller prints no "But it failed!" (e.g. `getSpreadDamage` `1228-1236`, `trySpreadMoveHit` `655-661`) |
| `false` | relay = false, stop | Loud failure: caller prints `-fail` / `-immune` / `-miss` |
| `0` | relay = 0, stop (falsy). **`hitStep` filters keep 0 as success** (`targets.filter(... hitResults[i] or hitResults[i] === 0)`, `sim/battle-actions.ts:667`) | No Guard's `onAnyInvulnerability` returns 0 = "hit, stop asking" (`data/abilities.ts:2972-2975`). `HIT_SUBSTITUTE = 0` (`sim/battle.ts:274`). |
| `''` (`NOT_FAIL`, `sim/battle.ts:273`) | Falsy, stop | "Didn't hit, but don't count as a failure". Protect / Wide Guard / Quick Guard `onTryHit` return it (`data/moves.ts:13958+`, `20804+`, `14486+`). `hitStepTryHitEvent` keeps `''` distinct from `false` (`sim/battle-actions.ts:711-713`). `moveThisTurnResult = null` if nothing failed (`678`). |
| `true` | relay = true, continue (with `fastExit`: stop) | "Explicitly allowed". Accuracy `true` means always hit. |
| number | relay = number, continue (`fastExit`: stop) | Replacement value. Later handlers see the new number. The final modify still applies. |
| object/array | relay replaced (e.g. `ModifyBoost`, `ModifySecondaries`, `ChangeBoost`, `Type`, `RedirectTarget` returning a Pokémon) | |

`combineResults(left, right)` (`sim/battle-actions.ts:1615-1631`) merges per-target outcomes in `runMoveEffects`:
- Rank `undefined < '' < null < boolean < number`. The higher type wins.
- Two numbers add.
- A truthy left beats a falsy non-zero right.

### 1.10 Modifiers: `modify`, `chainModify`, `finalModify`, `chain`

All are 4096-based fixed point. `trunc(x) = x >>> 0` (uint32 floor), and `trunc(x, 16) = (x>>>0) % 65536` (`sim/dex.ts:365-368`).

```
modify(v, num, den=1):  m = trunc(num*4096/den);  return trunc((trunc(v*m) + 2047) / 4096)   # battle.ts:2348-2359
     # = round-half-DOWN of v*m/4096  (verified == floor((v*m+2047)/4096) for v<=5000, 12 common m)
chainModify(num, den=1): prev = trunc(event.modifier*4096); next = trunc(num*4096/den)
                         event.modifier = ((prev*next + 2048) >> 12) / 4096                    # battle.ts:2337-2346
     # = round-half-UP product in 4096 units
chain(a, b)  same arithmetic on explicit values                                                  # battle.ts:2321-2335
finalModify(v): v = modify(v, event.modifier); event.modifier = 1; return v                      # battle.ts:2397-2401
```

- `finalModify` is used by paralysis (`data/conditions.ts:33`): it applies all chained Speed modifiers first, then `floor(spe*50/100)`.
- Common constants:
  - 0.25 → 1024; 0.5 → 2048; 0.75 → 3072; 1.5 → 6144; 2 → 8192.
  - Light Screen / Reflect / Aurora Veil in doubles → 2732.
  - Bright Powder 3686; Wide Lens 4505; Expert Belt / type items / Muscle Band 4915.
  - Life Orb 5324; terrain / Sheer Force / Gem 5325.
- **Order dependence.** `(a*b+2048)>>12` is commutative for 2 terms, but 3+ terms are order-dependent. 72 of 343 ordered triples from {4915, 6144, 5325, 2732, 5324, 3072, 4505} differ (`verify-tables.js`).

### 1.11 Ability/item suppression and Mold Breaker

- **`suppressingAbility(target)`** (`sim/battle.ts:366-369`) is true when all of these hold:
  - `activePokemon.isActive`;
  - `activePokemon !== target` (gen ≥ 8);
  - `activeMove.ignoreAbility`;
  - the target is not holding Ability Shield.
- Mold Breaker sets `move.ignoreAbility = true` in `onModifyMove` (`data/abilities.ts:2690-2701`). `useMoveInner` also copies `ignoreAbility` from a calling move (`sim/battle-actions.ts:486-489`).
- **Effect.** Every handler of an ability with `flags.breakable` held by anyone other than the attacker is skipped in `runEvent` (`843-848`) and in `singleEvent('SwitchIn')` (`609-613`).
  - Examples: Levitate, Lightning Rod, Storm Drain, Multiscale, Friend Guard, Filter, Thick Fat, Inner Focus, Aura Guard, Eelevate.
  - `isGrounded` checks `suppressingAbility` for Levitate/Eelevate (`sim/pokemon.ts:2177`).
- **`ignoringAbility()`** (`sim/pokemon.ts:879-898`) is checked in this order:
  1. Inactive (gen ≥ 5) → ignored.
  2. `notransform` + transformed → ignored.
  3. `cantsuppress` → never ignored.
  4. Gastro Acid → ignored.
  5. Holding Ability Shield, or own ability is Neutralizing Gas → not ignored.
  6. Any active, non-gastro'd, non-transformed, non-ending Neutralizing Gas → ignored (unless this Pokémon is `commanding`).
- **`ignoringItem()`** (`sim/pokemon.ts:900-907`) is true for:
  - inactive (gen ≥ 5), Embargo, or Magic Room;
  - Klutz (unless `ignoreKlutz`).
  - Primal orbs are never ignored.
- **Weather suppression.** `field.suppressingWeather()` (`sim/field.ts:106-116`) is true when any active non-fainted Pokémon has an un-ignored ability with `suppressWeather`.

### 1.12 Every event that fires in a normal doubles turn, in order

These lists come from real traces (`trace-*.txt`). The turn shown is Fake Out, Rock Slide, rain-boosted Hurricane and Wood Hammer (flinched), followed by residual: 511 dispatches, 75 distinct (kind, event) pairs.

**Kinds:** S = `singleEvent`, R = `runEvent`, P = `priorityEvent`, E = `eachEvent`, F = `fieldEvent`.

**A. Battle start / switch-in** (`trace-start.txt`; owned by the turn-flow agent; listed for completeness).
- `S BattleStart` (per species) → `R BeforeSwitchIn` (each) → `F SwitchIn`.
- Inside `F SwitchIn`, in speed order: `S SwitchIn` (ability `onStart` via `getCallback`). That triggers `R ChangeBoost` → `R TryBoost` → `R AfterEachBoost` → `R AfterBoost` for Intimidate, and `S FieldStart` → `E WeatherChange`/`TerrainChange` for setters.
- Then `E Update`, `R EmergencyExit`, `R DisableMove`/`S DisableMove` (per move slot), `R TrapPokemon`, `R MaybeTrapPokemon`, `S FoeMaybeTrapPokemon`, `P LockMove`, `P SemiLockMove`, `R ModifySpecies`.

**B. Turn start / queue resolution** (turn-flow agent). Per action:
- `P LockMove`, `P SemiLockMove`;
- `R ModifyBoost` / `R ModifySpe` (speed);
- `R FractionalPriority`;
- `S ModifyPriority` (move) and `R ModifyPriority`;
- then `E BeforeTurn` and `E Update`.

**C. Per move action** (`runMove`). Indentation shows nesting; `[t]` means once per target.

```
R OverrideAction                     runMove 285        (skipped for Struggle / external)
R BeforeMove                         312   -> false/null: R MoveAborted 314, stop
  (deductPP; P LockMove via getLockedMove 337)
useMove -> useMoveInner:
R ModifyTarget (+move onModifyTarget via onEffect)        480
S ModifyType (move) ; S ModifyMove (move)                 494-495
R ModifyType ; R ModifyMove                               502-503
(getMoveTargets) P RedirectTarget                         pokemon.ts:851  (single-target moves only, doubles)
R DeductPP [per pressure target]                          540
S TryMove (move) ; R TryMove                              550-553
S UseMoveMessage                                          558
-- field/side moves: tryMoveHit --
   S Try ; S PrepareHit ; R PrepareHit ; R TryHitField | R TryHitSide ; moveHit -> spreadMoveHit (S TryHitField/TryHitSide ...)
     -> S SideStart / S FieldStart / S Start ; R SideConditionStart / R PseudoWeatherChange ...
-- pokemon-targeting moves: trySpreadMoveHit --
S Try (move) ; S PrepareHit (move) ; R PrepareHit         652-654   (Protect's onPrepareHit -> R StallMove)
R Invulnerability [t]                                     692
R TryHit [array]                                          706       (Protect/Wide Guard/Quick Guard/Psychic Terrain/absorb abilities)
R NegateImmunity [t] (+R Type)                            pokemon.ts:2270 via runImmunity
S TryImmunity [t]                                         735
R ModifyAccuracy [t] ; R ModifyBoost x2 ; R Accuracy [t]  774-798
(R HitProtect inside protect onTryHit via checkMoveBypassesProtect)   battle.ts:1316
hit loop (per hit):
  [multiaccuracy hit>1: R ModifyBoost, R ModifyAccuracy, R Accuracy]
  spreadMoveHit:
    S TryHit (move, targets[0] only)                       ch/scripts.ts:332
    R TryPrimaryHit [t]                                    (Substitute)
    getDamage [t]: R NegateImmunity ; R ModifyCritRatio ; R CriticalHit (if crit) ;
                   R BasePower (+move onBasePower) ; R ModifyBoost x2 ; R ModifyAtk|SpA ; R ModifyDef|SpD ;
                   P WeatherModifyDamage ; R ModifySTAB ; (S Effectiveness ; R Effectiveness) per defender type ;
                   R ModifyDamage
    spreadDamage [t]: R Damage ; (drain: R TryHeal, R Heal)
    runMoveEffects [t]: boosts -> R ChangeBoost, R TryBoost, R AfterEachBoost, R AfterBoost ;
                        status -> R Immunity, R SetStatus, S Start, R AfterSetStatus ;
                        volatile -> R Immunity, R TryAddVolatile, S Start | S Restart ;
                        S Hit (move) ; R Hit
    selfDrops -> moveHit(self) (nested spreadMoveHit)
    secondaries [t]: R ModifySecondaries ; per secondary roll -> moveHit (nested)
    forceSwitch [t]: R DragOut
    R DamagingHit [array] ; S AfterHit (move) [t] ; R EmergencyExit (attacker)
  E Update
(after loop) resolveFaints -> R BeforeFaint, R Faint, S End (ability/item...), R AfterFaint
  applyRecoilDamage -> R Damage, R EmergencyExit
E Update
S AfterMoveSecondary (move) ; R AfterMoveSecondary [array]   (Champions: not suppressed by Sheer Force)
R EmergencyExit [t]
back in useMoveInner: (move failed: S MoveFail, R EmergencyExit)
S AfterMoveSecondarySelf (move) ; R AfterMoveSecondarySelf    (Life Orb recoil; suppressed by Sheer Force)  600-607
R EmergencyExit (attacker)
runMove: S AfterMove (move) ; R AfterMove                      368-369
(runAction epilogue) E Update ; R EmergencyExit ; queue re-sort: R ModifySpe, S/R ModifyPriority ...
```

**D. End of turn** (`runAction('residual')`, `sim/battle.ts:2851-2858`).
- `F Residual`. Its children, in residual order:
  - `S FieldResidual` (weather upkeep → `E Weather` → `R Weather` → `E Update`);
  - `S Residual` per handler (terrain heal → `R TryHeal`, `R Heal`; Leftovers; burn → `R Damage`; …);
  - `S SideResidual`;
  - `S End` / `S SideEnd` / `S FieldEnd` when durations expire.
- Then `E Update` and `R EmergencyExit` (per active).
- Then `endTurn` (`sim/battle.ts:1640-1818`): `R DisableMove`, `S DisableMove` (per move slot), `R TrapPokemon`, `R MaybeTrapPokemon`, `S FoeMaybeTrapPokemon`.

**Seen in the other traces** (Protect, crit, KO, Will-O-Wisp, Tailwind, Icy Wind, Earthquake turns):
- `R StallMove`, `R HitProtect`, `S MoveFail`, `R CriticalHit`;
- `R BeforeFaint`, `R Faint`, `R AfterFaint`, `R ModifySpecies`;
- `R SetStatus`, `R AfterSetStatus`, `R Immunity`;
- `R UseItem`, `R TryEatItem`, `S Eat`, `R EatItem`, `R AfterUseItem` (Sitrus);
- `R TryHitSide`, `S TryHitSide`, `S SideStart`, `R SideConditionStart` (Tailwind);
- `R ChangeBoost`, `R TryBoost`, `R AfterEachBoost`, `R AfterBoost` (Icy Wind).

**Full catalogue.** The complete list of event names PS defines is the `EventMethods` interface, `sim/dex-conditions.ts:13-501`. It has 401 `on*` keys including prefixes and metadata, and about 100 base events.

### 1.13 Designing a static hook table from this

Treat each `runEvent(E, …)` call site as a **hook point** with a fixed signature.

1. **Holder enumeration is static in doubles.** For a hook on target `t` with source `s`, the candidate holders are:
   - `t`'s status/volatiles/ability/item/slot (`on`);
   - each living ally including `t` (`onAlly`, `onAny`);
   - each living foe (`onFoe`, `onAny`);
   - `s` (`onSource`);
   - both sides' conditions (`on` for `t`'s side, `onFoe` for the other, `onAny` both);
   - field pseudo-weathers, weather and terrain (`on`);
   - format (`on`).

   That is at most 4 Pokémon × (status + V volatile slots + ability + item) plus 2 sides × S side-condition slots plus field slots. **The handler candidate set per hook is a fixed-size tensor**, gated by masks (present / not suppressed / alive).
2. **Order.** Precompute per (effect, event, prefix) the static triple `(order, priority, subOrder)` from data. Speed is dynamic: the holder's current `pokemon.speed`.
   - Sort keys: `(order, -priority, -speed, subOrder, effectOrder)`.
   - In practice most hooks have ≤ 3 live handlers. A fixed small sorting network, or "evaluate in a static canonical order", is exact whenever priorities differ. Ties are random in PS: pick a canonical order and accept the RNG-only difference.
3. **Fold.** Each handler kind becomes one of:
   - **Multiplier contributor**: emits a 4096 modifier, possibly gated by a predicate.
   - **Replacer**: returns a new relay value (number, boost dict, type list, target slot).
   - **Gate**: returns false / null / `''` / 0 to stop.

   The fold is: `mod = chain(mod, m_i)` in sorted order, stop at the first gate, then `modify(relay, mod)` once.
4. **Suppression masks per handler.** Stale status, Mold Breaker (breakable ∧ holder ≠ attacker ∧ the move ignores abilities), `ignoringAbility`, `ignoringItem`, and Air Lock.

---
## 2. The move pipeline (doubles)

**Call graph:**

```
runAction('move')  → runMove  → useMove → useMoveInner
    ├─ tryMoveHit                 (targets 'all' / 'foeSide' / 'allySide' / 'allyTeam')
    └─ trySpreadMoveHit           (everything else)
         → hitStep*
         → hitStepMoveHitLoop     (Champions)
         → spreadMoveHit          (Champions)
```

- `runAction('move')` is at `sim/battle.ts:2758-2765`. It returns early if the user is inactive or fainted.
- Callers such as Dancer use `runMove`. Most callers (Sleep Talk, Magic Bounce, …) use `useMove` (`sim/battle-actions.ts:419-428`).

### 2.1 `runMove(move, pokemon, targetLoc, opts)` — `sim/battle-actions.ts:267-418`

| # | Step | Lines | Events | RNG |
|---|---|---|---|---|
| 1 | `pokemon.activeMoveActions++` (used by Fake Out in Champions: `ch/moves.ts:349-356` disables it once `activeMoveActions > 0`; base `onTry` fails if `> 1`) | 274 | | |
| 2 | `target = getTarget(pokemon, move, targetLoc, originalTarget)` (§2.3) | 280 | | `getRandomTarget` may `sample` (RNG) |
| 3 | `baseMove = getActiveMove(move)`. A deep clone with `hit = 0` (`sim/dex.ts:316-322`). The priority is remembered. | 281-283 | | |
| 4 | Unless Struggle, Z, Max or external: `OverrideAction`. If it returns a move, swap to it and re-pick a random target. | 284-292 | R OverrideAction | maybe |
| 5 | `setActiveMove(move, pokemon, target)` | 302 | | |
| 6 | `willTryMove = runEvent('BeforeMove', pokemon, target, move)`. If falsy: `MoveAborted`, `clearActiveMove(true)`, `moveThisTurnResult = willTryMove` (false = "failed", null = "no choice"), return. | 312-321 | R BeforeMove → R MoveAborted | sleep / freeze / paralysis / confusion / flinch handlers draw here |
| 7 | `cantusetwice` hint volatile | 324-326 | | |
| 8 | `move.beforeMoveCallback` (Focus Punch) | 328-334 | | |
| 9 | `pokemon.lastDamage = 0` | 335 | | |
| 10 | If not external: `lockedMove = getLockedMove()` (P LockMove). If not locked: `deductPP(baseMove, null, target)`. If that returns 0 and the move isn't Struggle: `cant … nopp`, fail. If locked: `sourceEffect = lockedmove`. Then `moveUsed(move, targetLoc)`. | 336-349 | P LockMove | |
| 11 | `moveDidSomething = useMove(baseMove, pokemon, {target, sourceEffect})` | 365 | | |
| 12 | `battle.lastSuccessfulMoveThisTurn` | 366 | | |
| 13 | `S AfterMove` (move), then `R AfterMove` | 368-369 | | |
| 14 | Dancer: dancers sorted by `storedStats.spe` ascending, then `abilityState.effectOrder`; each runs `runMove(..., externalMove)` | 376-402 | | |
| 15 | `resolveFaints()`, `checkWin()` | 411-412 | R BeforeFaint / Faint / AfterFaint | |

Notes on step 10 and the move-tracking fields:
- **`deductPP`** (`sim/pokemon.ts:909-923`) finds the move slot, sets `used = true`, and subtracts `amount` (default 1) clamped at 0. It returns the amount deducted, or 0 if there is no slot or no PP.
- **`moveUsed`** (`sim/pokemon.ts:925-936`) sets `lastMove = move`, `lastMoveTargetLoc = targetLoc` and `moveThisTurn = move.id`.
- **Champions PP.** Base PP > 20 is capped at 20 (`ch/scripts.ts:3-9`), and max PP is `(pp/5+1)*4` (`ch/scripts.ts:41-43`). The mapping is 5 → 8, 10 → 12, 15 → 16, 20 → 20.
- **`battle.lastMove`** is set in `clearActiveMove(failed=false)` (`sim/battle.ts:381-390`), which runs at the end of each `runAction` (`sim/battle.ts:2874`). Failed-before-use paths call `clearActiveMove(true)`, which doesn't set it.

### 2.2 `useMove` / `useMoveInner` — `sim/battle-actions.ts:429-610`

`useMove` resets `moveThisTurnResult = undefined`, runs the inner function, and stores the result unless the inner function already set `moveThisTurnResult` (`435-439`).

| # | Step | Lines | Events |
|---|---|---|---|
| 1 | `sourceEffect` defaults to `battle.effect` (Instruct and Custap are excluded) | 451-452 | |
| 2 | `move = getActiveMove(...)`; `pokemon.lastMoveUsed = move` | 454-455 | |
| 3 | Copy `priority` and `pranksterBoosted` from the outer active move | 474-477 | |
| 4 | `ModifyTarget` with relay `{target}`. The move's own `onModifyTarget` (Comeuppance, Metal Burst) is prepended via `onEffect`. | 478-482 | R ModifyTarget |
| 5 | If target is undefined: `getRandomTarget`. For target `self` / `allies`: `target = pokemon`. | 482-485 | |
| 6 | If called by another move: `move.sourceEffect`, and `move.ignoreAbility` is inherited | 486-489 | |
| 7 | `setActiveMove(move, pokemon, target)` | 492 | |
| 8 | `S ModifyType` (move), `S ModifyMove` (move). If `move.target` changed: `getRandomTarget`. | 494-501 | |
| 9 | `R ModifyType` then `R ModifyMove` (abilities and items: Mold Breaker, Sheer Force, Stalwart, choice lock, King's Rock, …). If the target type changed: re-target. | 502-507 | |
| 10 | Return false if the move is gone or the user fainted. Then the `move` protocol log line. With no target: `-fail`, return false. | 508-529 | |
| 11 | `{targets, pressureTargets} = getMoveTargets(move, target)` (§2.3). `target = last target`. | 531-534 | P RedirectTarget |
| 12 | **Pressure** (§2.8) | 536-548 | R DeductPP per pressure target |
| 13 | `S TryMove` (move), then `R TryMove` (Primordial Sea / Desolate Land, …). Return if falsy. | 550-556 | |
| 14 | `S UseMoveMessage` | 558 | |
| 15 | `move.ignoreImmunity ??= (category === 'Status')` | 560-562 | |
| 16 | `selfdestruct === 'always'`: `faint(pokemon)` *before* hitting (Explosion) | 564-566 | |
| 17 | Branch. For targets `all` / `foeSide` / `allySide` / `allyTeam`: `damage = tryMoveHit(targets, …)`. `moveResult = true` if the damage is truthy, 0 or undefined; `NOT_FAIL` gives `moveThisTurnResult = null`. Otherwise, if `targets` is empty: `-fail`, return false; else `moveResult = trySpreadMoveHit(targets, pokemon, move)`. | 568-583 | |
| 18 | `selfBoost` (Clangorous Soul Blaze) | 584 | |
| 19 | User at 0 HP: `faint` | 585-587 | |
| 20 | If `!moveResult`: `S MoveFail` (High Jump Kick crash, …) and attacker `R EmergencyExit`; return false | 589-596 | |
| 21 | Unless the user has Sheer Force on a Sheer-Force-boosted move (`suppressingSecondaries`, `sim/battle.ts:371-373`) and the move is not a future move: `S AfterMoveSecondarySelf` (move), `R AfterMoveSecondarySelf` (**Life Orb recoil**, Shell Bell, Magician), then attacker `R EmergencyExit` | 600-607 | |
| 22 | return true | 609 | |

### 2.3 Target resolution in doubles

#### Positions

- `getLocOf(target)` returns +1/+2 for foe slots a/b and −1/−2 for own slots a/b (`sim/pokemon.ts:805-810`). `getAtLoc` inverts it (`791-799`).
- `validTargetLoc` (`sim/battle.ts:2415-2447`). With `activePerHalf = 2`, everyone is adjacent to everyone (`sim/pokemon.ts:762-767`).
  - `normal` / `randomNormal` / `scripted`: any adjacent slot, ally or foe.
  - `adjacentAlly`: ally only.
  - `adjacentAllyOrSelf`.
  - `adjacentFoe`: foe only.
  - `any`: not self.

#### `getTarget(pokemon, move, targetLoc, originalTarget)` — `sim/battle.ts:2453-2504`

Run at action time (and inside `getActionSpeed` for priority computations).
1. **`tracksTarget`.** Applies to the move's flag, or to Stalwart / Propeller Tail, checked here because `ModifyMove` hasn't run yet (`2456-2463`). If the original target is still active, return it. This ignores both redirection and position swaps.
2. **`smartTarget`** (Dragon Darts): the chosen slot if alive, else random (`2467-2470`).
3. **Targeting your own slot** with an `adjacentAlly` / `any` / `normal` move returns null (fails), except mid-charge (`2472-2479`).
4. **Valid loc, not `randomNormal`** (`2480-2502`):
   - fainted foe → fall through to random;
   - fainted ally → return the fainted ally (no retarget; the move later fails);
   - living → return it.
5. Otherwise `getRandomTarget`.

#### `getRandomTarget` — `sim/battle.ts:2506-2538`

- For `self` / `all` / `allySide` / `allyTeam` / `adjacentAllyOrSelf`: `pokemon`.
- For `adjacentAlly`: `sample(adjacentAllies)` (RNG), or null.
- Doubles (`activePerHalf = 2`): `pokemon.side.randomFoe()` = `sample(living foes)` (RNG) `|| foe.active[0]`.

#### `getMoveTargets(move, target)` — `sim/pokemon.ts:812-877`

| move.target | Targets (in this order) |
|---|---|
| `all`, `foeSide`, `allySide`, `allyTeam` | allies-and-self (unless `foe*`), then `foes(true)` (unless `ally*`) |
| `allAdjacent` (Earthquake, Surf, Discharge) | adjacent allies, then adjacent foes. **The ally comes first.** |
| `allAdjacentFoes` (Rock Slide, Heat Wave, Icy Wind) | adjacent foes |
| `allies` | allies and self |
| anything else (single target) | see the steps below |

For single-target moves:
1. If the target is missing, or is a fainted **foe**: `getRandomTarget` (a retarget; RNG). Otherwise keep the target.
2. If `activePerHalf > 1 && !move.tracksTarget`:
   `target = priorityEvent('RedirectTarget', pokemon, pokemon, move, target)`.
3. `smartTarget` → `getSmartTargets` (2 targets).
4. If the final target is fainted (and the move isn't a future move): empty targets, so the move fails with `[notarget]`.
5. Log the retarget.

Pressure targets are the same list, except `foeSide` → none and the `mustpressure` flag → all foes (`867-876`).

#### Redirection handlers

Run through `priorityEvent`: first non-undefined wins; sorted by priority, then speed, then ability effectOrder.

| Effect | Handler | Lines | Condition |
|---|---|---|---|
| Follow Me volatile | `onFoeRedirectTarget`, priority 1 | `data/moves.ts:6042-6078` | holder not Sky-Dropped and `validTarget(holder, source, move.target)` |
| Rage Powder volatile | `onFoeRedirectTarget`, priority 1 | `data/moves.ts:14595-14630` | as Follow Me, **plus** `source.runStatusImmunity('powder')`: Grass types, Overcoat and Safety Goggles are immune |
| Lightning Rod | `onAnyRedirectTarget`, priority 0 | `data/abilities.ts:2344-2368` | `move.type === 'Electric'`, not a pledge, and the holder is a valid target treating `randomNormal`/`adjacentFoe` as `normal` |
| Storm Drain | `onAnyRedirectTarget` | `data/abilities.ts:4637`… | as Lightning Rod but for Water |

- Because Follow Me and Rage Powder have priority 1, they beat Lightning Rod and Storm Drain.
- Both are `onFoe…` (or `onAny` for the abilities), so the ability can redirect an **ally's** move too (Lightning Rod is `onAny`).
- Snipe Shot and Stalwart / Propeller Tail set `tracksTarget`, which skips `RedirectTarget` entirely (`sim/pokemon.ts:850`).
- The absorb part (Lightning Rod's `onTryHit` returning null after `+1 SpA`) happens later, in `hitStepTryHitEvent`.

#### Spread flag

`trySpreadMoveHit` sets `move.spreadHit = true` iff `targets.length > 1 && !smartTarget` (`sim/battle-actions.ts:613`). This happens **before** any Protect, immunity or accuracy filtering.
- So the ×0.75 applies even if only one target is finally hit.
- Verified in Ex1: Corviknight is immune to Earthquake, and both foes still take ×0.75.

### 2.4 `trySpreadMoveHit(targets, pokemon, move)` — `sim/battle-actions.ts:612-682`

1. Set `spreadHit` (613).
2. `S Try` (move, target = targets[0]) → `S PrepareHit` (move) → `R PrepareHit` (652-654).
   - Protect-family `onPrepareHit` does `queue.willAct() && runEvent('StallMove')` (`data/moves.ts:13958+`).
   - If falsy: `-fail` if false; return `hitResult === NOT_FAIL` (655-661).
3. For each step: `hitResults = step(targets)`. If it returns undefined, the step is skipped. Otherwise:
   - `targets = targets.filter((_, i) => hitResults[i] || hitResults[i] === 0)`;
   - `atLeastOneFailure ||= hitResults.includes(false)`;
   - if `smartTarget` and there was a failure, clear `smartTarget`;
   - stop if no targets remain (663-674).
4. `move.hitTargets = targets`. If no targets remain and nothing failed, `moveThisTurnResult = null` (all targets protected → "not a failure"). Return `targets.length > 0` (676-681).

Step order (gen ≥ 7; the swaps for gens ≤ 6 at 641-648 don't apply):

| Step | Function | Lines | Events | RNG | Fails how |
|---|---|---|---|---|---|
| 0 | `hitStepInvulnerabilityEvent` | 683-704 | `R Invulnerability` per target (left-to-right sort). Helping Hand skips it. A `commanding` target always misses. Gen ≥ 8 Toxic from a Poison type always hits. | — | `false` → `-miss` |
| 1 | `hitStepTryHitEvent` | 705-715 | `R TryHit` on the target **array**: Protect family, Wide Guard / Quick Guard (side, priority 4), Psychic Terrain (field, priority 4), Lightning Rod / Storm Drain / Volt Absorb / Sap Sipper / …, Magic Bounce | Protect's `checkMoveBypassesProtect` runs `R HitProtect` | `false` → `-fail` (only if no `true`). `''` = blocked silently. Other falsy → false. |
| 2 | `hitStepTypeImmunity` | 716-727 | `target.runImmunity(move, msg)` → `R NegateImmunity` (+ `R Type` via `getTypes`) | — | false → `-immune` |
| 3 | `hitStepTryImmunity` | 728-751 | Powder vs. Grass/Overcoat (`getImmunity('powder')`); `S TryImmunity` (move); Prankster-boosted vs. a Dark-type foe | — | false → `-immune` |
| 4 | `hitStepAccuracy` | 752-816 | `R ModifyAccuracy`, `R ModifyBoost` (×2), `R Accuracy` (§6) | **`randomChance(acc, 100)` per target in target order**, unless accuracy is `true` | miss → `-miss`, Blunder Policy |
| 5 | `hitStepBreakProtect` | 817-842 | If `move.breaksProtect` (Feint, …): remove protect volatiles and the Crafty Shield / Mat Block / Quick Guard / Wide Guard side conditions; delete `stall` | — | returns undefined |
| 6 | `hitStepStealBoosts` | 843-872 | Spectral Thief only | — | undefined |
| 7 | `hitStepMoveHitLoop` (Champions) | `ch/scripts.ts:426-588` | §2.5 | multihit count, crit, roll, secondaries | |

Notes on the Protect family:
- Each `onTryHit` begins with `if (checkMoveBypassesProtect(move, source, target)) return;` (`sim/battle.ts:1314-1323`). That check returns false (blocked) when `(category !== 'Status' || blockStatus) && move.flags.protect && runEvent('HitProtect', attacker, defender, move)`.
- **Champions Unseen Fist** (`ch/abilities.ts:91-100`) and Piercing Drill: `onHitProtect` returns false for contact moves and sets `getMoveHitData(move).bypassProtect = effect`. The move goes through Protect, and `modifyDamage` later applies `modify(d, 0.25)` plus `-ability` (`ch/scripts.ts:298-305`).
- Protect resets a 2-turn `lockedmove` (`data/moves.ts:13958+`).

### 2.5 `hitStepMoveHitLoop` (Champions) — `ch/scripts.ts:426-588`

Differences from base (`sim/battle-actions.ts:916-1079`), confirmed by `diff`:
- (a) `resolveFaints` is unconditional (no gen check; equivalent in gen 9). Champions line 545.
- (b) No `-hitcount` when Parental Bond only hit once (546-547).
- (c) **`afterMoveSecondaryEvent` and the per-target `EmergencyExit` loop are not wrapped in `if (!suppressingSecondaries())`** (573-585). In Champions, Sheer Force does not suppress Berserk, Anger Shell, Pickpocket, Eject Button, Red Card or Emergency Exit / Wimp Out triggers.

Algorithm:

```
damage[i] = 0 for each target; move.totalDamage = 0; pokemon.lastDamage = 0
targetHits = move.multihit || 1
if array [2,5]: targetHits = sample([2]*7 + [3]*7 + [4]*3 + [5]*3)          # 35/35/15/15 %   RNG: random(20)
                if targetHits < 4 and has Loaded Dice: targetHits = 5 - random(2) # (Loaded Dice is 'Past' in Champions)
elif array [a,b]: targetHits = random(a, b+1)
if targetHits == 10 and Loaded Dice: targetHits -= random(7)                     # Population Bomb
for hit in 1..targetHits:
   break if any damage[i] === false
   break if hit>1 and user asleep (unless sleepUsable)
   break if every target has 0 HP
   move.hit = hit; move.lastHit = (hit == targetHits)
   smartTarget (Dragon Darts) handling
   if multiaccuracy and hit>1 (Triple Axel, Population Bomb):
        float accuracy *= / /= [1,4/3,5/3,2,7/3,8/3,3][stage]  (accuracy & evasion stages, R ModifyBoost)
        R ModifyAccuracy; unless alwaysHit: R Accuracy; break unless randomChance(accuracy, 100)    # RNG per extra hit
   [moveDamageThisHit, targetsCopy] = spreadMoveHit(targetsCopy, pokemon, move, move)
   break if every entry is false; nullDamage = false
   damage[i] = (md === true || !md) ? 0 : md; move.totalDamage += damage[i]
   E Update
   if user fainted and single target: hit++; break
if hit == 1: return all false
if nullDamage: all false
resolveFaints({checkWin: !pokemon.hp})
-hitcount message
if move.totalDamage: applyRecoilDamage(totalDamage)                 # recoil once, on the TOTAL (Parental Bond: both hits)
target.gotAttacked(...), timesAttacked += hits
if no damage[i] is truthy or 0: return damage
E Update
afterMoveSecondaryEvent(targets): S AfterMoveSecondary (move, targets[0]); R AfterMoveSecondary [array]
for each target with hp: R EmergencyExit(target, hpBefore = hurtThisTurn + damage)
return damage
```

- **Parental Bond.** `onPrepareHit` sets `move.multihit = 2` and `multihitType = 'parentalbond'` (`data/abilities.ts:3171-3189`). This is skipped for spread moves, status moves and moves that already multihit.
  - Hit 2 takes `modify(d, 0.25)` right after `+2` (`ch/scripts.ts:208-213`).
  - Each hit draws its own crit and roll (verified in Ex4 RNG log: acc, crit1, roll1, crit2, roll2).
  - Recoil is computed on the summed damage (Ex4: `round(170·33/100) = 56`).

### 2.6 `spreadMoveHit` (Champions) — `ch/scripts.ts:314-423`

The only difference from base (`sim/battle-actions.ts:1080-1192`) is at line 411: `if (moveData.onAfterHit)` instead of `if (moveData.onAfterHit && pokemon.hp)`. Rapid Spin, Mortal Spin, Ceaseless Edge, Knock Off, Covet, Ice Spinner and others run their `onAfterHit` even if the user fainted (Rocky Helmet / Rough Skin).

It is called with `(targets, source, move, moveData = move | secondary | self, isSecondary, isSelf)`:

1. **TryHit** (single-target hardcode, `targets[0]` only) (327-340).
   - `S TryHitField` for target `all`, `S TryHitSide` for side targets, else `S TryHit` (move).
   - Examples: Helping Hand fails if the ally won't move; Substitute's self-check; Brick Break removes screens; Wide Guard's `onTry` is separate.
   - Falsy → `-fail` if false; return `[[false], targets]`.
2. **Step 0, Substitute.** If not secondary/self and not a side/field move: `damage[i] = R TryPrimaryHit(target)` (`sim/battle-actions.ts:1193-1202`).
   - Substitute (`data/moves.ts:18300+`) computes `getDamage`, hits the sub, applies recoil and drain, and returns `HIT_SUBSTITUTE = 0`. The target then becomes `null` (still "hit" for move-success purposes, but no further effects) (349-358).
   - For secondaries without `self`, `damage[i] = true`.
3. **Step 1, `getSpreadDamage`** (`sim/battle-actions.ts:1203-1240`). Per target, in order: `activeTarget = t`; `getDamage(source, t, moveData)` (§3).
   - `false` or `null` → `damage[i] = false`; `-fail` only if it was already false.
   - `undefined` → no damage, continue (status moves).
4. **Step 2, `battle.spreadDamage`** (`sim/battle.ts:2108-2210`). Per target:
   - non-numeric → pass through;
   - target at 0 HP → 0; inactive → false;
   - clamp ≥ 1 (if nonzero);
   - `R Damage(target, source, effect=move, dmg)` (Focus Sash `onDamage` at priority −40, `data/items.ts:2281-2297`; Disguise), then clamp ≥ 1 again. Multiscale is **not** here: it is an `onSourceModifyDamage` ×0.5 inside `modifyDamage` (`data/abilities.ts:2761-2767`);
   - `target.damage(d)`, which caps at the HP and queues a faint (`sim/pokemon.ts:1616-1626`);
   - `hurtThisTurn = hp`; `source.lastDamage`;
   - **drain:** `heal(round(d·drain[0]/drain[1]), source, target, 'drain')` → `R TryHeal`, `R Heal` (`2187-2190`).
5. **Step 3, `runMoveEffects`** (`sim/battle-actions.ts:1241-1372`). Per non-false target, in this order:
   - `boosts` → `battle.boost` (§5);
   - `heal` (fails at full HP; `round(baseMaxhp·num/den)`);
   - `status` → `trySetStatus`; a failure of a status move's primary status fails that target;
   - `forceStatus`;
   - `volatileStatus` → `addVolatile`;
   - `sideCondition`, `slotCondition`, `weather`, `terrain`, `pseudoWeather`;
   - `forceSwitch` → `canSwitch`;
   - `S HitField` / `S HitSide` / `S Hit` (move), then **`R Hit`** (not for self/secondary);
   - `selfdestruct === 'ifHit'`;
   - `selfSwitch` (U-turn / Volt Switch / Parting Shot sets `source.switchFlag` if something succeeded and the user can switch; 1366-1369).
   - A move that "did nothing" prints `-fail` (1358-1365).
6. **Step 4, `selfDrops`** (`sim/battle-actions.ts:1373-1391`). If `moveData.self` and not `move.selfDropped`: for each target that isn't `false`:
   - If not secondary and `self.boosts`: draw **`random(100)` unconditionally** and apply if `chance` is undefined or the roll is below `chance`. Then set `selfDropped` (not for multihit), so the drop happens once even for spread moves.
   - Otherwise `moveHit(self)` with no roll.
   - Examples: Close Combat, Draco Meteor, Superpower, Overheat (the roll is non-semantic: always succeeds).
7. **Step 5, `secondaries`** (`sim/battle-actions.ts:1392-1408`). Per non-false target:
   - `secondaries = R ModifySecondaries(target, source, moveData, list)` (Shield Dust, Covert Cloak filter);
   - then **for each secondary: `random(100)`** and apply via `moveHit(target, source, move, secondary, isSecondary=true)` if `chance` is undefined or the roll is below `chance`.
   - The `chance % 256` overflow applies only to gen ≤ 8.
   - Sheer Force has already deleted `move.secondaries` in `ModifyMove` (`data/abilities.ts:4203-4222`), so there are no rolls.
8. **Step 6, `forceSwitch`** (`sim/battle-actions.ts:1409-1425`). `R DragOut` → `forceSwitchFlag`.
9. **Damaged targets** (numeric damage, still a target). If not secondary/self:
   - `R DamagingHit(targets[], source, move, damages[])` (left-to-right): Rocky Helmet, Rough Skin, Static, Flame Body, Weakness Policy, Stamina, Justified, …;
   - `S AfterHit` (move) per target (**Champions: even if the attacker fainted**);
   - attacker `R EmergencyExit`.

### 2.7 `tryMoveHit` (field and side moves) — `sim/battle-actions.ts:879-915`

1. `setActiveMove`.
2. `S Try` → `S PrepareHit` → `R PrepareHit`.
3. `R TryHitField` (target `all`), or `R TryHitSide` (side; Magic Bounce's `onAllyTryHitSide`). FFA hazards are irrelevant here.
4. `moveHit` → `spreadMoveHit` with the representative target. `S TryHitSide` (move) → `runMoveEffects` → `sideCondition` / `pseudoWeather` / `weather` / `terrain` → `S SideStart` / `S FieldStart` → `R SideConditionStart` / `R PseudoWeatherChange` / `E WeatherChange` / `E TerrainChange`.
   - Example: Wide Guard's `onTry` requires `queue.willAct()`, and its `onHitSide` adds `stall` to the user.
5. There are no accuracy, crit or damage steps.

### 2.8 PP, Pressure, `moveThisTurn`, `lastMove`, Struggle, recoil, drain

- **Base PP deduction.** `runMove` step 10: one PP from `baseMove`'s slot unless locked or external.
- **Pressure** (`sim/battle-actions.ts:536-548`). Only when not called by another effect, or when called by a move with PP (`callerMoveForPressure`). For each pressure target, `ppDrop = R DeductPP(target, source=user, move)`.
  - Pressure's handler returns 1 unless the target is an ally of the source (`data/abilities.ts:3438-3450`).
  - In doubles a spread move against two Pressure foes costs 1 + 2 PP.
  - `extraPP` is deducted from the same slot.
  - The final modify applies but is the identity.
- **`moveThisTurn` / `lastMove`.**
  - `moveUsed` sets `lastMove`, `lastMoveTargetLoc` and `moveThisTurn` (`sim/pokemon.ts:925-936`).
  - `moveThisTurnResult` is true/false/null as set in `useMove` / `runMove`. In `endTurn` it is rolled into `moveLastTurnResult` and `moveThisTurn` is cleared (`sim/battle.ts:1689-1692`).
  - `lastMoveUsed` is set in `useMoveInner` (455) even for called moves.
- **Struggle** (`data/moves.ts:18207-18228`): 50 BP, `???` type via `onModifyMove`, so no STAB and neutral effectiveness. `randomNormal` target, with an RNG foe pick.
  - `runMove` skips `OverrideAction` and tolerates `deductPP` returning 0.
  - Recoil: `struggleRecoil` → `clampIntRange(round(baseMaxhp/4), 1)` via `directDamage` (no Damage event) (`sim/battle-actions.ts:1437, 1444-1446`).
- **Recoil** (`applyRecoilDamage`, `sim/battle-actions.ts:1435-1453`).
  - `recoil: [n, d]` → `max(1, round(totalDamage·n/d))` via `battle.damage(..., 'recoil')` → `R Damage`. Rock Head blocks it by returning `null` when `effect.id === 'recoil'` and the move isn't Struggle (`data/abilities.ts:3907-3913`).
  - Mind Blown / Chloroblast → `round(maxhp/2)`.
  - Then attacker `R EmergencyExit`.
  - Recoil from damage dealt to a Substitute is applied inside the Substitute handler.
- **Drain.** `round(d·n/den)` in `spreadDamage` (`sim/battle.ts:2187-2190`), healed with effect `drain` (`R TryHeal`: Big Root chains 5324/4096, `data/items.ts:488-492`; Liquid Ooze inverts). Substitute uses `ceil` (`data/moves.ts:18300+`).
- **Life Orb** (`data/items.ts:3422-3438`): `onModifyDamage` 5324/4096; `onAfterMoveSecondarySelf` deals `baseMaxhp/10` if `source !== target` and the category isn't Status. Both are skipped by Sheer Force (step 21 in §2.2).

### 2.9 RNG draws in move execution (order verified by `rng-order.js`, `smoke.js`, `damage-examples.js`)

Every draw goes through `battle.prng`:
- `random(n)` returns `floor(u32·n/2^32)`;
- `randomChance(n, d)` is `random(d) < n` (`sim/prng.ts:96-123`);
- `battle.randomChance` honours `forceRandomChance` (`sim/battle.ts:351-354`).

| # | Kind (engine event id) | Call site | Draw | When |
|---|---|---|---|---|
| a | BeforeMove gates: full paralysis (Champions 1/8), freeze thaw (1/4), confusion self-hit (33/100), sleep (none: a counter), Attract, … | `ch/conditions.ts:5,49`; `data/conditions.ts:187` | `random(8)`, `random(4)`, `random(100)` (+ `random(16)` roll for confusion damage) | per mover, before PP |
| b | Random target (retarget / `randomNormal` / Struggle / `adjacentAlly` with no choice) | `sim/battle.ts:2523, 2537`; `sim/side.ts:369-373` | `random(#living foes)` | `getTarget` / `getMoveTargets` |
| c | Protect chain | `data/conditions.ts:451-459` (`stall.onStallMove`) | `random(counter)`, counter = 3^k up to 729 | `PrepareHit`, consecutive use only |
| d | Accuracy | `sim/battle-actions.ts:800` | `random(100) < acc`, **per target, all targets first** | step 4 |
| e | Multihit count | `ch/scripts.ts:439, 447, 450` | `random(20)` (2–5), `random(a, b+1)` | hit loop start |
| f | Crit | `sim/battle-actions.ts:1694-1696` | `random(critMult[stage])` with critMult = [0, 24, 8, 2, 1]. No draw if `willCrit` is defined or the stage is 0. Stage ≥ 3 still draws (`random(2)`, `random(1)`). | per target, inside `getDamage` |
| g | Damage roll | `sim/battle.ts:2407-2410` | `random(16)`, multiplier `100 − r` | per target, right after its crit draw |
| h | Self-drop roll | `sim/battle-actions.ts:1381` | `random(100)`, unconditional | once per move (non-semantic for chance-less drops) |
| i | Secondary | `sim/battle-actions.ts:1399` | `random(100)` per secondary per target, **after all targets' damage** | after `runMoveEffects` |
| j | Status durations at infliction: sleep `sample([2,3,3])` (Champions), confusion `random(2,6)`, partial trap `random(5,7)`, `lockedmove` `random(2,4)` | `ch/conditions.ts:23`; `data/conditions.ts:174, 227, 265` | | inside `S Start` |
| k | Speed-tie shuffles | `sim/battle.ts:461` | `random(m, n)` | any `speedSort` with a tie: queue, `eachEvent`, `fieldEvent`, `runEvent` handler ties |
| l | Non-semantic target picks | `sim/battle.ts:2697` (`getActionSpeed` → `getTarget`), `sim/battle-queue.ts` `resolveAction` | `random(2)` | every queue (re)sort, for spread moves (whose target is irrelevant) |

**Observed sequence for a doubles turn** (Rock Slide → Draco Meteor → Icy Wind), from `rng-order.js`:
1. non-semantic target picks ×7;
2. Rock Slide: acc(t1), acc(t2), crit(t1), roll(t1), crit(t2), roll(t2), sec(t1), sec(t2);
3. target pick;
4. Draco Meteor: acc, crit, roll, selfDrop;
5. target picks ×2;
6. Icy Wind: acc, acc, crit, roll, crit, roll, sec, sec;
7. speed-tie shuffles in `Update` and `Residual`.

---
## 3. The damage formula, exactly as PS computes it

### 3.1 Integer primitives

| Name | Definition | Source |
|---|---|---|
| `trunc(x)` | `x >>> 0`: floor for 0 ≤ x < 2^32, then uint32 wrap | `sim/dex.ts:365-368` |
| `trunc(x, 16)` | `(x >>> 0) % 65536` | same |
| `clampIntRange(x, lo, hi)` | `floor(x)`, then clamp; non-numbers become 0 | `lib/utils.ts:320-326` |
| `modify(v, m)` | `trunc((trunc(v·m) + 2047) / 4096)`, with m an integer in 4096ths | `sim/battle.ts:2348-2359` |
| `chain(a, b)` | `(a·b + 2048) >> 12` | `sim/battle.ts:2321-2346` |
| Roll | `trunc(trunc(d·(100 − r)) / 100)`, r = `random(16)` ∈ [0, 15] | `sim/battle.ts:2407-2410` |

All intermediate products in practical VGC fit in int32. For example, `22·BP·A` ≤ 22 · 400 · 4000 ≈ 3.5e7. **int32 is sufficient in JAX.** The final 16-bit wrap only matters above 65535 damage, which never happens with level-50 stats.

### 3.2 Stats (Champions)

**Stored stats** (`ch/scripts.ts:10-40`; Reg M-C has no Level Clause Mod, so the non-level branch is used):

```
HP    = base + SP + 75
other = base + SP + 20
        if nature+: other = trunc( trunc(other*110, 16) / 100 )
        if nature-: other = trunc( trunc(other*90,  16) / 100 )
```

- SP is the `evs` field: 0..32 per stat, 66 total.
- There are no IVs.
- Example: Sneasler Adamant 32 Atk: 130 + 32 + 20 = 182 → trunc(200.2) = **200**.

**`calculateStat(stat, boost, modifier=1, statUser)`** (`sim/pokemon.ts:581-615`; used by damage):
1. Take the stored stat. Under Wonder Room, Def and SpD are swapped.
2. `boost = R ModifyBoost(statUser, {stat: boost})[stat]`. Unaware's `onAnyModifyBoost` zeroes it.
3. Clamp the boost to ±6 and apply `floor(stat·[1, 1.5, 2, 2.5, 3, 3.5, 4][b])` or `floor(stat / table[−b])`. The integer equivalent `floor(s·(2+b)/2)` / `floor(2s/(2−b))` was verified for s ≤ 2000.
4. `modify(stat, 1)`, which is the identity.

**`getStat`** (`sim/pokemon.ts:617-660`; used for speed and other non-damage reads):
- the same boost math (unless `unboosted`);
- then `R Modify<Stat>(this, null, null, stat)`;
- then cap Speed at 10000.

**Speed for ordering** (`ch/scripts.ts:46-54`): `getStat('spe')`, negated under Trick Room.
- Paralysis: `finalModify`, then `floor(spe·50/100)`, at priority −101 (`data/conditions.ts:30-38`).
- Tailwind: `chainModify(2)` (`data/moves.ts:18871+`).
- Choice Scarf: `chainModify(1.5)` (`data/items.ts:995-1017`).

### 3.3 `getDamage(source, target, move)` — `sim/battle-actions.ts:1639-1776`

```
if !target.runImmunity(move, msg): return false                         # 1656 (2nd immunity check)
if move.ohko: return target.maxhp                                       # 1660
if move.damageCallback: return it (Counter, Final Gambit, Super Fang...)  # 1661
if move.damage == 'level': return source.level (=50); elif move.damage: return it   # 1662-1666
bp = move.basePower; if basePowerCallback: bp = callback(source, target, move)      # 1670-1673
if !bp: return (bp === 0 ? undefined : bp)                              # 1674  (status moves -> undefined)
bp = max(1, floor(bp))
critRatio = clamp( R ModifyCritRatio(source, target, move, move.critRatio || 0), 0, 4 )   # 1678-1689; dex default critRatio = 1 (sim/dex-moves.ts:490)
crit = move.willCrit || false
if move.willCrit === undefined and critRatio: crit = randomChance(1, [0,24,8,2,1][critRatio])   # RNG
if crit: crit = R CriticalHit(target, null, move)                       # Battle Armor / Shell Armor -> false
bp = R BasePower(source, target, move, bp, onEffect=true)               # move.onBasePower + all handlers; final modify
if !bp: return 0; bp = max(1, bp)
(tera 60-BP floor: dead in Champions, canTerastallize = null, ch/scripts.ts:179-181)
attacker = target if move.overrideOffensivePokemon=='target' (Foul Play) else source
defender = source if move.overrideDefensivePokemon=='source' else target
atkStat = move.overrideOffensiveStat (Body Press 'def') or (Physical ? 'atk' : 'spa')
defStat = move.overrideDefensiveStat (Psyshock/Psystrike/Secret Sword 'def') or (Physical ? 'def' : 'spd')
aB = attacker.boosts[atkStat]; dB = defender.boosts[defStat]
if crit: ignore aB if aB<0; ignore dB if dB>0                            # 1740-1745
if move.ignoreOffensive: aB = 0; if move.ignoreDefensive (Sacred Sword, Darkest Lariat): dB = 0
A = attacker.calculateStat(atkStat, aB, 1, source)
D = defender.calculateStat(defStat, dB, 1, target)
A = R Modify{Atk|SpA by CATEGORY}(source, target, move, A)               # 1759-1762 (event name follows category, not atkStat)
D = R Modify{Def|SpD by defStat}(target, source, move, D)               # 1763
base = trunc(trunc(trunc(trunc(2*L/5+2) * bp * A) / D) / 50)            # 1772   (L=50 -> trunc(22))
return modifyDamage(base, source, target, move)
```

**Event argument conventions** (a frequent source of bugs). In these hooks, "target" in the handler name means the runEvent target, which is often the attacker.

| Event | runEvent target | runEvent source | Attacker's handlers | Defender's handlers |
|---|---|---|---|---|
| ModifyCritRatio | attacker | defender | `onModifyCritRatio` (Scope Lens, Super Luck) | `onSourceModifyCritRatio` |
| BasePower | attacker | defender | `onBasePower` (Technician, type items, Helping Hand volatile) | `onSourceBasePower` (Dry Skin vs. Fire, …) |
| ModifyAtk / ModifySpA | attacker | defender | `onModifyAtk` (Huge Power, Blaze, Guts); ally: `onAllyModifyAtk` (Flower Gift) | `onSourceModifyAtk` (Thick Fat, Heatproof, Water Bubble) |
| ModifyDef / ModifySpD | defender | attacker | `onSourceModifyDef/SpD` | `onModifyDef/SpD` (Assault Vest, Eviolite, Fur Coat, sandstorm SpD, snow Def) |
| WeatherModifyDamage (priorityEvent) | attacker | defender | | Weather conditions check `defender.effectiveWeather()` (Utility Umbrella) |
| ModifySTAB | attacker | defender | Adaptability | |
| Effectiveness | defender | **type string** | | Delta Stream, Disguise, Iron Ball |
| ModifyDamage | attacker | defender | `onModifyDamage` (Life Orb, Expert Belt) | `onSourceModifyDamage` (Multiscale, Filter / Solid Rock, resist berries, Fluffy) |

Screens and Friend Guard use `onAnyModifyDamage` and test `target !== source && effectState.target.hasAlly(target)`.

### 3.4 `modifyDamage` (Champions) — `ch/scripts.ts:196-312`

Numerically identical to `sim/battle-actions.ts:1778-1896`.

| # | Step | Code | Integer op |
|---|---|---|---|
| 1 | `d = base + 2` | 201 | add |
| 2 | Spread (if `move.spreadHit`) | 203-207 | `modify(d, 3072)` (0.5 only in FFA) |
| 2' | Parental Bond hit ≥ 2 (else-branch) | 208-213 | `modify(d, 1024)` |
| 3 | Weather: `priorityEvent('WeatherModifyDamage', attacker, defender, move, d)`. Rain: Water ×1.5, Fire ×0.5. Sun: inverse, plus Hydro Steam. Primordial Sea / Desolate Land boosts only. Mega Sol (Champions ability) treats the user's moves as in sun via `effectiveWeather`. | 216 | `modify(d, chainedM)` |
| 4 | Crit (if `moveHit.crit`) | 219-222 | `trunc(d · 1.5)`, float multiply then trunc (not `modify`) |
| 5 | Random | 225 | `trunc(trunc(d·(100−r))/100)` |
| 6 | STAB (type ≠ `???`): 1.5 if `forceSTAB` or `hasType` or pre-tera type, then `R ModifySTAB` (Adaptability: 2; 2.25 only with tera) | 232-263 | `modify(d, 6144)` or `modify(d, 8192)` |
| 7 | Type: `tm = clamp(runEffectiveness, −6, 6)`, stored in `moveHitData.typeMod` | 266-282 | `d *= 2` per +1; `d = trunc(d/2)` per −1 |
| 8 | Burn: `brn ∧ Physical ∧ ¬Guts ∧ move ≠ Facade` | 286-290 | `modify(d, 2048)` |
| 9 | Final: `R ModifyDamage(attacker, defender, move, d)`. Order comes from handler sort; most have priority 0, so speed then subOrder decides, which only matters when 3+ factors chain. | 296 | `modify(d, chainedM)` |
| 10 | `bypassProtect` (Champions Unseen Fist / Piercing Drill through Protect; Z/Max) | 298-305 | `modify(d, 1024)` |
| 11 | `if d == 0: return 1` | 308 | |
| 12 | `return trunc(d, 16)` | 311 | |

Common `ModifyDamage` multipliers (4096ths):
- Reflect / Light Screen / Aurora Veil in doubles: 2732 (×0.5 in singles). Skipped on crit or `infiltrates` (`data/moves.ts:14839-14879`, `830-877`). Aurora Veil doesn't stack with the matching screen.
- Life Orb 5324; Expert Belt 4915 if `typeMod > 0` (`data/items.ts:1913-1926`).
- Multiscale / Shadow Shield 2048.
- Filter / Solid Rock / Prism Armor 3072 on super-effective hits.
- Friend Guard 3072.
- Resist berries 2048 (consumed).

Then `spreadDamage` clamps to ≥ 1 and runs `R Damage` (Focus Sash). Champions has no other post-`modifyDamage` damage changes.

### 3.5 Type effectiveness and immunity

- **Type chart.** `data/typechart.ts`: `damageTaken[attackType]` ∈ {0 neutral, 1 weak, 2 resist, 3 immune}. Special keys `prankster`, `powder`, `par`, `brn`, `sandstorm`, … encode status and pseudo immunities.
- **`dex.getEffectiveness(move, type)`** (`sim/dex.ts:271-292`): 1 → +1, 2 → −1, else 0. Immunity (3) counts as **0** here; it is handled by `getImmunity` instead.
- **`runEffectiveness(move)`** (`sim/pokemon.ts:2229-2254`): `Σ_{type ∈ getTypes()}` of: `dex.getEffectiveness` → `S Effectiveness` (move: Freeze-Dry vs. Water = +1, Flying Press adds Flying) → `R Effectiveness(defender, type, move)` (Delta Stream). Tera Shell is irrelevant.
- **`runImmunity(move)`** (`sim/pokemon.ts:2257-2288`):
  - `move.ignoreImmunity === true` or `[type]` → not immune. Status moves default to `ignoreImmunity = true` (`sim/dex-moves.ts:505`); Scrappy sets `{Normal, Fighting}` (`data/abilities.ts:4080-4088`).
  - Type `???` → not immune.
  - `negate = !R NegateImmunity(target, type)`.
  - Ground → `isGrounded(negate)` (Flying type, Levitate unless Mold Breaker, Air Balloon, Magnet Rise; Gravity / Smack Down / Iron Ball / Ingrain ground; `sim/pokemon.ts:2169-2181`).
  - Otherwise `negate || dex.getImmunity(type, target)` (any type with code 3 → immune).
- **Immunity is checked twice**: in `hitStepTypeImmunity` (filters targets, logs `-immune`) and again in `getDamage` (1656).
- **Ability immunities are `TryHit`, not type immunity.** Volt Absorb, Water Absorb, Flash Fire, Sap Sipper, Lightning Rod and Storm Drain (`onTryHit` returning null/false) are evaluated in step 1, *before* type immunity. Levitate is the exception (`isGrounded`).

### 3.6 Reference implementation (explicit integer ops)

Verified bit-exact on 64 roll/target cells: `damage-examples.js` implements exactly this.

```python
U32 = 0xFFFFFFFF
def trunc(x):            return int(x) & U32 if x >= 0 else (int(x) & U32)   # x>>>0 ; inputs are non-negative in practice
def trunc16(x):          return trunc(x) % 65536
def modify(v, m):        return trunc((trunc(v * m) + 2047) // 4096)        # m: int, 4096 = 1.0
def chain(a, b):         return (a * b + 2048) >> 12
def chain_all(ms):       # ms in PS handler order (sorted: order, -priority, -speed, subOrder, effectOrder)
    m = 4096
    for x in ms: m = chain(m, x)
    return m
def stage(stat, b):      # stat stages for atk/def/spa/spd/spe
    b = max(-6, min(6, b))
    return (stat * (2 + b)) // 2 if b >= 0 else (stat * 2) // (2 - b)

def damage(L, bp, bp_mods, A_raw, a_boost, a_mods, D_raw, d_boost, d_mods,
           crit, spread, pbond_hit2, weather_mods, roll, stab_m, type_mod, burned, final_mods, bypass_protect):
    bp = max(1, modify(bp, chain_all(bp_mods)))                      # R BasePower (+ final modify)
    if crit and a_boost < 0: a_boost = 0
    if crit and d_boost > 0: d_boost = 0
    A = modify(stage(A_raw, a_boost), chain_all(a_mods))            # calculateStat + R ModifyAtk/SpA
    D = modify(stage(D_raw, d_boost), chain_all(d_mods))            # calculateStat + R ModifyDef/SpD
    d = trunc(trunc(trunc(trunc(2 * L // 5 + 2) * bp * A) // D) // 50)
    d += 2
    if spread:      d = modify(d, 3072)
    elif pbond_hit2: d = modify(d, 1024)
    d = modify(d, chain_all(weather_mods))
    if crit:        d = trunc(d * 3 // 2)                            # == trunc(d*1.5) for integer d
    d = trunc(trunc(d * (100 - roll)) // 100)                        # roll = random(16) in 0..15
    if stab_m:      d = modify(d, stab_m)                            # 6144 (STAB) or 8192 (Adaptability); skip for '???'
    t = max(-6, min(6, type_mod))
    for _ in range(max(t, 0)):  d *= 2
    for _ in range(max(-t, 0)): d = trunc(d // 2)
    if burned:      d = modify(d, 2048)                              # physical, not Guts, not Facade
    d = modify(d, chain_all(final_mods))                             # R ModifyDamage
    if bypass_protect: d = modify(d, 1024)
    if d == 0: return 1
    return trunc16(d)
```

Notes:
- `trunc(d*1.5)` equals `(3·d)//2` for non-negative integer `d`, since d·1.5 is exact in float64.
- `2*L//5+2` with L = 50 gives 22.
- The final `modify` inside `runEvent` happens only if the relay is a non-negative integer. It always is for these damage-path events.

### 3.7 Worked examples (verified by running PS)

Each example is a Reg M-C battle at level 50 with real Champions stats. The damage roll is forced to each r ∈ 0..15 in turn. Accuracy is forced to hit, crit forced as stated, and secondaries forced off. Reproduce with `node research/ps-move/damage-examples.js` (prints "ALL EXAMPLES VERIFIED") and `node research/ps-move/intermediates.js`.

**Example 1: spread Earthquake, ally immune, STAB, SE/NVE, Life Orb (doubles).**

Setup:
- P1a Garchomp @ Life Orb, Jolly, SP Atk 32 → Atk 182. Uses Earthquake (`allAdjacent`).
- P1b Corviknight (Flying: immune).
- P2a Incineroar, Careful, SP Def 20 → Def 130.
- P2b Rillaboom, Adamant, SP Def 2 → Def 112.

`getMoveTargets` gives [Corviknight, Incineroar, Rillaboom]. Three targets → `spreadHit = true`. Corviknight is removed at `hitStepTypeImmunity`, and the ×0.75 still applies to both foes.

| Target, roll | Trace |
|---|---|
| Incineroar r=0 | 22·100·182 = 400400 → /130 = 3080 → /50 = 61 → +2 = 63 → spread `modify(63, 3072)` = 47 → ×100/100 = 47 → STAB `modify(47, 6144)` = 70 → ×2 = 140 → Life Orb `modify(140, 5324)` = **182** |
| Incineroar r=15 | … 47 → ×85/100 = 39 → STAB 58 → ×2 = 116 → LO **151** |
| Rillaboom r=0 | 400400/112 = 3575 → /50 = 71 → 73 → 55 → 55 → 82 → `trunc(82/2)` = 41 → LO **53** |

PS output for r = 0..15:
- Incineroar: [182, 179, 179, 174, 174, 172, 172, 166, 166, 164, 164, 159, 159, 156, 156, 151]
- Rillaboom: [53, 52, 51, 51, 51, 51, 49, 49, 48, 48, 47, 47, 47, 45, 45, 44]

Both match the reference. PS log: `|move|p1a: Garchomp|Earthquake|p2b: Rillaboom|[spread] p2a,p2b`, `|-immune|p1b: Corviknight`, `|-supereffective|p2a: Incineroar|1`, `|-resisted|p2b: Rillaboom|1`. The trailing `|1` is the Champions message argument.

**Example 2: crit with negative attack / positive defense boosts ignored, burn, Reflect ignored, Life Orb, 4×.**

Setup:
- P1a Sneasler @ Life Orb, Adamant, SP Atk 32 → Atk 200. Burned, Atk −1. Uses Mach Punch (40 BP, Fighting STAB).
- P2a Kingambit (Dark/Steel → 4×), Impish, SP Def 32 → Def 189. Def +2. Reflect is up on its side.
- Crit forced.

| Roll | Trace |
|---|---|
| r=0 | boosts ignored → A = 200, D = 189 → 22·40·200 = 176000 → /189 = 931 → /50 = 18 → 20 → crit `trunc(20·1.5)` = 30 → 30 → STAB 45 → ×4 = 180 → burn `modify(180, 2048)` = 90 → Life Orb `modify(90, 5324)` = **117**. Reflect's handler returns early on crit, so no 2732. |
| r=15 | 30 → 25 → 37 → 148 → 74 → **96** |

PS: [117, 112, 112, 112, 109, 109, 109, 104, 104, 104, 104, 101, 101, 101, 96, 96]. Matches.

RNG log: `random(100)` acc (forced 0) → `random(24)` crit (forced 0) → `random(16)` roll.

**Example 3: rain + Helping Hand + Mystic Water (BasePower chain) + Light Screen in doubles, 2×, special.**

Setup:
- P1a Pelipper @ Mystic Water, Drizzle, Modest, SP SpA 32 → SpA 161. Uses Water Pulse (60 BP).
- P1b Whimsicott uses Helping Hand on Pelipper (priority +5 moves first).
- P2a Incineroar, Careful, SP SpD 32 → SpD 156. Light Screen is up on its side.

| Step | Value |
|---|---|
| BasePower handlers, sorted by priority | Mystic Water (`onBasePowerPriority` 15) then Helping Hand volatile (10) |
| Chain | `chain(4915, 6144)` = 7373 |
| BP | `modify(60, 7373)` = **108** |
| r=0 | 22·108·161 = 382536 → /156 = 2452 → /50 = 49 → 51 → rain `modify(51, 6144)` = 76 → 76 → STAB 114 → ×2 = 228 → Light Screen `modify(228, 2732)` = **152** |
| r=15 | 76 → 64 → 96 → 192 → **128** |

PS: [152, 149, 148, 145, 144, 144, 141, 140, 137, 137, 136, 133, 132, 132, 129, 128]. Matches.

**Example 4: Mega Kangaskhan Parental Bond Double-Edge (two hits, second ×0.25; recoil on the total).**

Setup:
- P1a Kangaskhan @ Kangaskhanite, mega-evolves this turn (Atk 194, Adamant, SP 32).
- Uses Double-Edge (120 BP) into P2a Garchomp (Def 115).

| Hit, roll | Trace |
|---|---|
| Hit 1, r=0 | 22·120·194 = 512160 → /115 = 4453 → /50 = 89 → 91 → 91 → STAB **136** |
| Hit 2, r=0 | 91 → `modify(91, 1024)` = 23 → 23 → STAB **34** |

PS (hit1+hit2) for r = 0..15: [136+34, 135+33, 133+33, 132+33, 130+33, 129+31, 127+31, 126+31, 124+31, 123+30, 121+30, 120+30, 120+30, 118+30, 117+28, 115+28]. Matches.

- Recoil: `round(170·33/100)` = 56. In the log, Kangaskhan goes 136 → 80 HP after the two Rough Skin chips.
- RNG order: acc → crit₁ → roll₁ → crit₂ → roll₂. There is one accuracy roll for the whole Parental Bond move, because it is not `multiaccuracy`.

---
## 4. Status, volatiles, and side/slot/field conditions

### 4.1 Storage and lifecycle

| Kind | Storage | Add | Remove | Restart (re-application while present) |
|---|---|---|---|---|
| Major status | `pokemon.status` (id), `statusState` | `setStatus` (`sim/pokemon.ts:1705-1771`): fail if same status / immune (`runStatusImmunity`: type chart + `R Immunity`) → `R SetStatus` → set → `duration`/`durationCallback` → `S Start` (can cancel) → `R AfterSetStatus` | `cureStatus` / `clearStatus` → `setStatus('')` | none (fails) |
| Volatile | `pokemon.volatiles[id]` | `addVolatile` (`sim/pokemon.ts:1990-2048`): if present → `S Restart` if defined, else false; `runStatusImmunity(id)`; `R TryAddVolatile`; create state `{id, target, source, sourceSlot, sourceEffect, duration}`; `durationCallback`; `S Start` (false → delete) | `removeVolatile` → `S End` then delete (`2056-2067`); all cleared on switch-out (`ch/scripts.ts:123-176`) | `onRestart` (Helping Hand stacks ×1.5, stall ×3, lockedmove) |
| Side condition | `side.sideConditions[id]` | `addSideCondition` (`sim/side.ts:413-443`): restart via `S SideRestart`; `durationCallback(side.active[0], source, effect)`; `S SideStart`; `R SideConditionStart` | `removeSideCondition` → `S SideEnd` (`456-462`) | `onSideRestart` (hazard layers) |
| Slot condition | `side.slotConditions[pos][id]` | `addSlotCondition` (`sim/side.ts:464-495`), `S Start` | `removeSlotCondition` → `S End` (`504-511`) | `onRestart` |
| Weather | `field.weather`, `weatherState` | `setWeather` (`sim/field.ts:39-89`): same weather from an ability fails in gen > 5; `R SetWeather`; `duration`/`durationCallback(source)` (5, or 8 with rocks); `S FieldStart`; `E WeatherChange` | `clearWeather` → `S FieldEnd`, `E WeatherChange` | none |
| Terrain | `field.terrain`, `terrainState` | `setTerrain` (`sim/field.ts:130-157`): same terrain fails; `S FieldStart`; `E TerrainChange` | `clearTerrain` | none |
| Pseudo-weather | `field.pseudoWeather[id]` | `addPseudoWeather` (`sim/field.ts:186-216`): `S FieldRestart` if present (Trick Room toggles off: `data/moves.ts:19936+`); `S FieldStart`; `R PseudoWeatherChange` | `removePseudoWeather` → `S FieldEnd` | `onFieldRestart` |

### 4.2 How `duration` counts down (the only automatic counter)

**The decrement point.** `duration` lives on the effect state and is decremented in exactly one place: `fieldEvent('Residual')` (`sim/battle.ts:520-528`), once per end-of-turn residual phase.
1. Handlers are sorted by residual order.
2. When the walker reaches a handler whose state has `duration` and an `end` function, it decrements first.
3. If the result is 0, it calls `end` (`removeVolatile`, `removeSideCondition`, `clearWeather`, …) **instead of** the residual callback.

**Consequences, verified with `durations.js`:**

| Effect | Set when | Ends (`-sideend` / `-fieldend` / `-weather none`) | Active turns |
|---|---|---|---|
| Tailwind (4) | turn 1 | end of turn 4 | 1–4 |
| Trick Room (5) | turn 1 | end of turn 5 | 1–5 |
| Reflect (5) | turn 2 | end of turn 6 | 2–6 |
| Drought sun (5) | lead switch-in (turn 0) | end of turn 5 | 1–5 |

The general rule: **an effect with duration N set during turn T is active through turn T+N−1 and removed in the residual phase of turn T+N−1.**

Further rules:
- **One-turn effects** (duration 1: Protect, Helping Hand, Follow Me, Rage Powder, Wide Guard, Quick Guard, flinch, Roost, Endure) are removed in the residual of the turn they were set.
- **Turn-0 setting.** Weather set at turn 0 (lead switch-in) is first decremented at the end of turn 1.
- **Weather in its last turn:** the end happens *before* the upkeep callback, so there is no damage tick that turn.
- **Order within the residual.** Since the decrement happens when the handler is reached, an expiring effect still applies to residual handlers sorted **before** it. Weather (field order 1) expires before Leftovers (order 5) etc. Screens (side order 26) and Trick Room / terrains (field order 27) expire near the end.
- **Mid-turn removals** (Defog, Brick Break, Feint removing Protect, Trick Room toggle, a weather override) delete the state immediately. Handlers already collected for the residual are skipped by the "state still installed?" check (`sim/battle.ts:530-563`).
- **Fainted holders.** Their Pokémon-held effects are skipped, but slot conditions are not (`517-519`). Clearing volatiles on switch-out removes their counters.
- **`durationCallback` signatures differ.**
  - Volatile: `(target, source, effect)`.
  - Side: `(side.active[0], source, effect)`.
  - Field: `(source, source, effect)` (`sim/field.ts:80,148,208`).
  - So "Light Clay on the setter" is checked on `source`.
- **Counters not driven by `duration`** are decremented by hand inside handlers:
  - sleep `statusState.time` (BeforeMove);
  - Champions freeze `time` (BeforeMove);
  - confusion `time` (BeforeMove);
  - toxic `stage` (Residual: +1, capped at 15);
  - `lockedmove.trueDuration` (Residual);
  - Perish Song (duration-driven, but prints its count in Residual);
  - stall `counter` (multiplied on Restart);
  - Encore / Taunt / Disable (duration-driven).

**JAX mapping.** Each condition becomes a fixed slot with a `turns_left` int8 (0 = absent) plus a few int8/int16 args (counter, stage, move id, source slot). The residual phase becomes:
1. compute all per-slot residual handler masks;
2. order them by the static (order, priority, subOrder) and the dynamic holder speed;
3. scan, decrementing `turns_left` and firing `end` when it hits 0.

Because decrement-then-end happens at the handler's position, the scan must interleave decrements with residual damage and healing in sorted order. A single "decrement everything first" pass is **not** equivalent. A weather that expires does not tick that turn; Leftovers ordered before a Pokémon's own expiring volatile still sees it.

### 4.3 Residual order table (VGC-relevant)

`onResidualOrder` (lower first). Ties are broken by priority, then holder speed, then subOrder.

| Order | Effect (subOrder) | Source |
|---|---|---|
| 1 | Weather upkeep + `E Weather` (sand damage), field-level | `data/conditions.ts:509` etc. |
| 3 | Future Sight / Doom Desire (slot) | `data/conditions.ts:395` |
| 4 | Wish (slot) | `data/moves.ts:20935` |
| 5 | Grassy Terrain heal (2), Healer / Hydration / Shed Skin (3), Leftovers / Black Sludge (4) | `data/moves.ts:7714`, `data/items.ts:3356, 550`, abilities |
| 6 | Aqua Ring | |
| 7 | Ingrain | |
| 8 | Leech Seed | `data/moves.ts:10218` |
| 9 | Poison / Toxic | `data/conditions.ts:133, 154` |
| 10 | Burn; berry re-check | `data/conditions.ts:15` |
| 11 | Nightmare | |
| 12 | Curse | |
| 13 | Partial trap (Bind, Whirlpool, …), Salt Cure | `data/conditions.ts:233`, `data/moves.ts:15642` |
| 14 | Octolock, Syrup Bomb | |
| 15 | Taunt (end) | `data/moves.ts:18988` |
| 16 | Encore | |
| 17 | Disable | |
| 18 | Magnet Rise | |
| 20 | Heal Block | |
| 21 | Embargo | |
| 22 | Throat Chop | |
| 23 | Yawn | |
| 24 | Perish Song | |
| 25 | Roost | |
| 26 (side) | Reflect (1), Light Screen (2), Safeguard (3), Mist (4), **Tailwind (5)**, Lucky Chant (6), pledges (7–9), Aurora Veil (10) | `data/moves.ts:14869` etc. |
| 27 (field) | **Trick Room (1)**, Gravity (2), Water Sport (3), Mud Sport (4), Wonder Room (5), Magic Room (6), terrains (7) | `data/moves.ts:19966` etc. |
| 28 | Speed Boost, Moody, Harvest, Pickup, Bad Dreams, Cud Chew, Slow Start (2); Flame / Toxic Orb, Sticky Barb (3); Uproar | |
| 29 | Hunger Switch, Opportunist, Zen Mode, Schooling, Shields Down, Power Construct, Eject Pack, Mirror Herb, White Herb | |
| none (→ last) | Duration-only volatiles: Protect, Helping Hand, Follow Me, flinch, stall, lockedmove (has `onResidual` without an order), choicelock | |

Note: `lockedmove.onResidual` and `stall` have no order, so they sort last.

### 4.4 Statuses (Champions values)

| Status | Mechanics | Source |
|---|---|---|
| `brn` | Residual 10: `damage(baseMaxhp/16)`; ×0.5 physical damage in `modifyDamage` (§3.4) | `data/conditions.ts:2-19` |
| `par` | Speed: `finalModify`, then `floor(·50/100)` (onModifySpe priority −101; Quick Feet exempt). **BeforeMove (priority 1): `randomChance(1, 8)` full para** (Champions) | `data/conditions.ts:20-46`; `ch/conditions.ts:2-10` |
| `slp` | **Start: `time = startTime = sample([2,3,3])`** (Champions). BeforeMove (priority 10): Early Bird extra −1; `time--`; if ≤ 0 wake and act; else `cant` (sleep-usable moves proceed). Effective: 1 skipped turn with p = 1/3, 2 with p = 2/3. | `ch/conditions.ts:11-30`; `data/conditions.ts:66-81` |
| `frz` | **Champions: Start `time = 3`; BeforeMove: (defrost moves bypass) `time--`; thaw if `time ≤ 0` or `randomChance(1, 4)`**, else `cant`. Fire-type damaging hit thaws (`onDamagingHit`); defrost move thaws in ModifyMove. | `ch/conditions.ts:31-56`; `data/conditions.ts:83-122` |
| `psn` | Residual 9: `baseMaxhp/8` | `data/conditions.ts:123-137` |
| `tox` | `stage = 0` at start and on switch-in. Residual 9: `stage = min(stage+1, 15)`; `damage(max(1, floor(baseMaxhp/16)) · stage)` | `data/conditions.ts:138-161` |

All `this.damage(x)` calls go through `spreadDamage`, which clamps `x` to ≥ 1 after `floor`. Fractional HP is floored, e.g. `183/16 = 11.4375 → 11` (`sim/battle.ts:2130, 2145`; `sim/pokemon.ts:1616-1626`).

### 4.5 Key volatiles and their counters

| Volatile | duration | Counter / args | Key handlers | Source |
|---|---|---|---|---|
| confusion | — | `time = random(2,6)` (Axe Kick: 3–5) | BeforeMove (priority 3): `time--`; 0 → remove and act; else `randomChance(33,100)` → 40-BP typeless self-hit with its own `random(16)` roll (`getConfusionDamage`, `sim/battle-actions.ts:1904-1916`) | `data/conditions.ts:162-197` |
| flinch | 1 | — | BeforeMove (priority 8): `cant`, `R Flinch` (Steadfast) | `198-207` |
| protect (Detect) / spikyshield / banefulbunker / kingsshield / obstruct | 1 | — | TryHit (priority 3): block unless `checkMoveBypassesProtect`; contact punishers | `data/moves.ts:13958-14003` etc. |
| stall | 2 | `counter` 3 → ×3 per Restart, max 729 | StallMove: `randomChance(1, counter)`; failure deletes stall. The 1st use has no stall volatile, so there is no draw. Verified: consecutive uses draw `random(3)`. | `data/conditions.ts:443-466` |
| helpinghand | 1 | `multiplier` 1.5, ×1.5 per Restart | BasePower priority 10: `chainModify(multiplier)` | `data/moves.ts:8576-8610` |
| followme / ragepowder | 1 | — | `onFoeRedirectTarget` priority 1 | §2.3 |
| lockedmove | 2 | `trueDuration = random(2,4)`, `move` | LockMove; Residual `trueDuration--`; ends → confusion if `trueDuration ≤ 1` | `data/conditions.ts:253-286` |
| twoturnmove | 2 | `move`, `targetLoc` | LockMove; Start runs `R PrepareHit` | `287-323` |
| mustrecharge | 2 | — | BeforeMove priority 11 → null | `364-378` |
| choicelock | — | `move` | BeforeMove: fail other moves; DisableMove | `324-363` |
| partiallytrapped | `random(5,7)` (8 with Grip Claw) | `boundDivisor` 8 (6 with Binding Band) | Residual 13: damage 1/8; TrapPokemon | `222-252` |
| substitute | — | `hp = floor(maxhp/4)` | TryPrimaryHit priority −1 | `data/moves.ts:18300+` |
| taunt / encore / disable | 3 / 3 / 5 | `move` (encore / disable) | Champions Encore rewrites the queued action instead of `onOverrideAction` (`ch/moves.ts:309-338`); Champions Disable `onBeforeMove` (`ch/moves.ts:228-239`) | |
| commanded / commanding | — | — | DragOut / Trap; commanding is invulnerable (`hitStepInvulnerabilityEvent`) | `data/conditions.ts:813-848` |

---

## 5. Stat stages

### 5.1 `battle.boost(boosts, target, source, effect, isSecondary, isSelf)` — `sim/battle.ts:2037-2106`

1. Default the target, source and effect from the current event.
2. Return 0 if the target has 0 HP. Return false if it is inactive, or if no foe Pokémon are left (gen > 5) (`2046-2048`).
3. `boost = R ChangeBoost(target, source, effect, {...boost})`. Contrary negates; Simple doubles (`2049`).
4. `boost = target.getCappedBoost(boost)`: per stat, `clamp(cur + Δ, −6, 6) − cur`; zero entries are dropped (`sim/pokemon.ts:1231-1240`).
5. `boost = R TryBoost(target, source, effect, {...boost})`. Clear Body, White Smoke, Hyper Cutter, Big Pecks, Mirror Armor, Clear Amulet, Mist, Guard Dog, Inner Focus / Oblivious / Own Tempo / Scrappy vs. Intimidate, … delete entries (`2051`).
6. For each remaining stat, `boostBy({stat: Δ})` applies the capped delta (`sim/pokemon.ts:1242-1251`).
   - If the delta is non-zero: message, then `R AfterEachBoost` (Defiant / Competitive react here, `+2`), `success = true`.
   - Otherwise a "won't go higher/lower" message (rules at `2094-2098`).
7. `R AfterBoost(target, source, effect, boost)`: Rattled vs. Intimidate, Opportunist via `onFoeAfterBoost`, Adrenaline Orb, Mirror Herb.
8. Set `statsRaisedThisTurn` / `statsLoweredThisTurn` (`2101-2104`).
9. Return `success` (null if nothing changed).

**Call sites:**
- `runMoveEffects` (`moveData.boosts`, primary status moves);
- secondaries (`secondary.boosts` on the target, `secondary.self.boosts` via `moveHit(self)`);
- `selfDrops`;
- abilities and items (Intimidate is `boost({atk: −1}, target, pokemon, null, isSecondary=true)` per adjacent foe: `data/abilities.ts:2194-2213`);
- `hitStepStealBoosts`.

### 5.2 Multiplier tables

| Use | Formula | Where |
|---|---|---|
| Atk/Def/SpA/SpD/Spe (damage and `getStat`) | b ≥ 0: `floor(s·(2+b)/2)`; b < 0: `floor(2s/(2−b))` (from table [1, 1.5, 2, 2.5, 3, 3.5, 4]) | `sim/pokemon.ts:604-611, 642-649` |
| Accuracy / evasion (normal moves) | net stage `n = clamp(acc_stage − eva_stage, −6, 6)`; n > 0: `trunc(acc·(3+n)/3)`; n < 0: `trunc(acc·3/(3−n))` | `sim/battle-actions.ts:775-790` |
| Accuracy / evasion (multiaccuracy re-rolls, hits ≥ 2) | float `acc *= [1, 4/3, 5/3, 2, 7/3, 8/3, 3][acc_stage]` (or `/=` for negative), then evasion applied separately; **no trunc** | `ch/scripts.ts:481-503` |
| Crit stage | `critRatio = clamp(R ModifyCritRatio(base 1 + stage), 0, 4)` → 1/24, 1/8, 1/2, 1 | `sim/battle-actions.ts:1678-1696` |

`ModifyBoost` (`R ModifyBoost(pokemon, null, null, boosts)`) runs before every boost lookup:
- for the attacker (`statUser`) when computing damage;
- for both sides in accuracy.

Unaware (`onAnyModifyBoost`) is the main handler.

---

## 6. Accuracy — `hitStepAccuracy`, `sim/battle-actions.ts:752-816`

Per target, in target order (`activeTarget = target`):

```
acc = move.accuracy                               # number or true
if move.ohko:                                     # Sheer Cold, Fissure, Horn Drill, Guillotine: LEGAL in Reg M-C (no OHKO Clause)
    if !target.isSemiInvulnerable():
        acc = 30; if ohko=='Ice' and !user.hasType('Ice'): acc = 20
        if !dynamax and user.level >= target.level and (ohko===true or !target.hasType(ohko)): acc += user.level - target.level   # +0 at L50
        else: -immune [ohko]; fail (Sheer Cold vs Ice-types)
    # no stage or ModifyAccuracy modifiers for OHKO
else:
    acc = R ModifyAccuracy(target, user, move, acc)            # Bright Powder 3686 (target), Wide Lens 4505 (user, onSource...),
                                                               # Compound Eyes 5325, Hustle 3277 (physical), Sand Veil/Snow Cloak, Tangled Feet; final modify
    if acc !== true:
        b = 0
        if !move.ignoreAccuracy: b = clamp(R ModifyBoost(user).accuracy, -6, 6)
        if !move.ignoreEvasion:  b = clamp(b - R ModifyBoost(target).evasion, -6, 6)   # Chip Away/Sacred Sword etc. ignoreEvasion
        acc = trunc(acc*(3+b)/3) if b>0 else trunc(acc*3/(3-b)) if b<0 else acc
if move.alwaysHit or (Toxic by Poison-type) or (self-targeting Status move and target not semi-invulnerable):
    acc = true
else:
    acc = R Accuracy(target, user, move, acc)     # No Guard (onAnyAccuracy -> true), Lock-On, Minimize (Body Slam etc. -> true), Glaive Rush
if acc !== true and !randomChance(acc, 100):      # RNG: random(100) < acc
    -miss (or smartTarget fallback); Blunder Policy (+2 Spe) if not OHKO; fail
```

- **Always-hit moves.** Moves with `accuracy: true` (most status moves, Swift, Aerial Ace, …) skip the roll *and draw no RNG*. ModifyAccuracy still runs, but returns `true` unchanged because the item handlers return early for non-numbers.
- **Weather accuracy.** Hurricane and Thunder in rain become `true` via `onModifyMove` (50 in sun) (`data/moves.ts:9029+`). Blizzard in snow likewise.
- **No Guard** also returns 0 in `Invulnerability` (hits semi-invulnerable targets) (`data/abilities.ts:2971-2984`).
- **Semi-invulnerability** (Fly, Dig, Dive, Bounce, Phantom Force, Shadow Force, Sky Drop) is step 0 (`R Invulnerability`), not accuracy.
- **Multi-accuracy.** Population Bomb / Triple Axel re-roll per hit ≥ 2 inside the hit loop (§2.5). The **float** stage formula is used and Loaded Dice can skip; the loop breaks on the first miss.
- **Evasion stages** are reachable in Reg M-C (Double Team, Minimize, and no Evasion Clause in `ch/rulesets.ts:35-40`). So `b` can be negative, giving e.g. 100 → trunc(300/5) = 60 at +2 evasion.

---
## 7. JAX design implications

### 7.1 Table-driven vs. per-effect code

**Pure data (dense arrays exported from the Champions dex):**
- **Species:** base stats, types, weight, Mega forme mapping.
- **Moves:**
  - core numbers: BP, type, category, accuracy (`true` → flag), priority, PP (after the Champions cap and formula), crit stage;
  - targeting: `target` enum;
  - flags: contact, protect, powder, sound, bullet, punch, bite, slicing, wind, pulse, defrost, bypasssub, mustpressure, …;
  - multihit (fixed / 2–5 / range), `multiaccuracy`;
  - `self.boosts` with chance; secondaries list (chance, boosts / status / volatile / self-boosts);
  - drain / recoil / heal fractions;
  - status / volatileStatus / sideCondition / pseudoWeather / weather / terrain to apply;
  - stat-override fields (Foul Play, Body Press, Psyshock), `ignoreDefensive`, `ignoreEvasion`, `tracksTarget`, `willCrit`, `breaksProtect`, `selfSwitch`, `forceSwitch`, `ohko`;
  - sleepUsable, struggleRecoil.
- **Type chart:** 19×19 int8 codes {0, 1, 2, 3} plus the status pseudo-keys.
- **Condition metadata:** `duration`, `durationCallback` kind (item-extended: Light Clay / rocks / Terrain Extender; ability-extended: Persistent), and for each event `order`/`priority`/`subOrder` (static ints), `effectType` → default subOrder.
- **Residual order table** (§4.3).
- **Stage and modifier tables:** stat stage, accuracy stage, crit table, and the constants 2732 / 3072 / 5324 / 4915 / 5325 / …
- **"Multiplier contributor" handlers** as rows `(hook, effect, prefix, predicate_id, m4096, priority, order, subOrder)`. The heuristic census finds the large majority of BasePower (53/60), ModifyAtk/SpA (all), and over half of ModifyDamage (21/39) are of this form. Typical predicates are a type check, a category check, a flag check, a HP threshold, `typeMod > 0`, crit, a weather/terrain check, or grounded.

**Small, closed code families (`lax.switch` on a small id):**
- `basePowerCallback` (29 legal moves; most are formulas of HP, speed, weight, boosts or counters);
- `damageCallback` (7);
- `ModifyType` (≈10: Weather Ball, Terrain Pulse, Raging Bull, Aura Wheel, the -ate abilities incl. Champions Dragonize, Normalize);
- `ModifyMove` (≈29; many just set a flag such as `ignoreAbility`, `tracksTarget` or `accuracy = true`);
- TryHit gates and absorbers (≈49: 28 pure gates, 19 with side effects such as the absorb boosts and heals);
- BeforeMove gates (≈17);
- `RedirectTarget` (5);
- on-hit reactions (`DamagingHit` 29, `AfterHit` 8, `AfterMoveSecondary` 5);
- `Update` (25; berries and status cures: a "threshold → consume → effect" template covers most);
- `Residual` (≈42 + duration-only).

**Genuinely bespoke code** (each needs its own branch; keep these out of the hot `vmap` path where possible):
- status-move `onHit` (72, e.g. Trick, Skill Swap, Ally Switch, Transform, Encore, Instruct, Court Change);
- Commander;
- Dancer;
- Magic Bounce (re-`useMove`);
- Parental Bond / multihit loop;
- Substitute;
- Ally Switch;
- item theft and swap;
- forme changes (Mega happens via the turn flow).

### 7.2 Recommended hook taxonomy (fixed hook points with call sites)

Each hook gets:
- **(a)** a static candidate-holder layout (§1.13: up to 4 Pokémon × {status, V volatile slots, ability, item, slot-cond} + 2 sides × S side-cond slots + field {weather, terrain, P pseudo-weather slots});
- **(b)** a per-(effect, hook, prefix) row with `(order, priority, subOrder, kind, payload)`;
- **(c)** one of three fold kinds: **MUL** (4096 chain then one `modify`), **GATE** (first falsy wins, result ∈ {pass, fail_loud(false), fail_silent(null), not_fail(''), hit0(0)}), **REPL** (first or last non-undefined replacement).

Hooks marked † belong to the turn-flow agent but are listed for completeness.

| # | Hook | Fold | PS call site(s) | Typical members (Reg M-C) |
|---|---|---|---|---|
| H1 | BeforeMove | GATE + side effects + RNG | `sim/battle-actions.ts:312` | slp / frz / par (Champions 1/8) / confusion / flinch / attract / mustrecharge / choicelock / disable / Truant / taunt |
| H2 | LockMove / OverrideAction | REPL | `sim/pokemon.ts:971`; `sim/battle-actions.ts:285` | lockedmove, twoturnmove, mustrecharge |
| H3 | ModifyType, ModifyMove | REPL (move mutation) | `sim/battle-actions.ts:494-507` | Weather Ball, Terrain Pulse, -ate, Normalize, Mold Breaker, Sheer Force, Stalwart, Hurricane / Thunder / Blizzard accuracy, King's Rock, choice lock, Expanding Force (target change) |
| H4 | RedirectTarget | REPL, first-wins, `compareRedirectOrder` | `sim/pokemon.ts:851` | Follow Me, Rage Powder, Lightning Rod, Storm Drain |
| H5 | DeductPP | sum | `sim/battle-actions.ts:540` | Pressure |
| H6 | MoveTry (`S Try`, `S/R TryMove`, `S/R PrepareHit`) | GATE + side effects | `550-553`, `652-654`, `885-887` | Fake Out first-turn, Follow Me doubles-only, Aurora Veil needs snow, guards `willAct`, Protect StallMove (RNG), Primordial Sea, Parental Bond setup, Protean / Libero |
| H7 | Invulnerability | GATE (left-to-right) | `692` | Fly / Dig / Dive / Bounce / Phantom Force, No Guard (→ 0), commanding |
| H8 | TryHit (array) | GATE + side effects (left-to-right) | `706` (+ move `S TryHit` at `ch/scripts.ts:332`) | Protect family incl. contact punish, Wide Guard, Quick Guard, Psychic Terrain, absorb abilities, Magic Bounce, Good as Gold, Bulletproof, Soundproof, Telepathy; HitProtect / Unseen Fist ×0.25 |
| H9 | TypeImmunity (`runImmunity` + NegateImmunity + grounded) | table + few gates | `723`, `1656` | type chart, Levitate / Air Balloon / Magnet Rise / Gravity / Iron Ball, Scrappy, Ring Target |
| H10 | TryImmunity | GATE | `728-751` | powder vs. Grass / Overcoat / Goggles, Prankster vs. Dark, move-specific (Leech Seed vs. Grass, …) |
| H11 | ModifyAccuracy → stages → Accuracy | MUL then GATE(true) + RNG | `774-800` | Bright Powder, Wide Lens, Compound Eyes, Hustle, Sand Veil; No Guard, Minimize, Lock-On; OHKO branch |
| H12 | TryPrimaryHit | GATE / REPL (HIT_SUBSTITUTE) | `sim/battle-actions.ts:1199` | Substitute |
| H13 | ModifyCritRatio, CriticalHit | additive REPL, GATE | `1678`, `1700` | Scope Lens, Super Luck, Focus Energy, Dragon Cheer, Leek; Battle / Shell Armor |
| H14 | BasePower (+ `basePowerCallback`) | callback, then MUL | `1671-1673`, `1704` | type items / plates, Muscle Band / Wise Glasses, Technician, Sheer Force, Tough Claws, Iron Fist, Sharpness, terrains, Helping Hand, Charge, Gem, Dry Skin (source) |
| H15 | ModifyAtk / ModifySpA | MUL (some REPL: `return this.modify(...)`) | `1762` | Huge Power, Guts, Blaze / Torrent / Overgrow, Solar Power, Gorilla Tactics, Choice Band / Specs (Past), Thick Fat / Heatproof / Water Bubble (source), Ruin abilities |
| H16 | ModifyDef / ModifySpD | MUL / REPL | `1763` | Assault Vest (Past), Eviolite, Fur Coat, Marvel Scale, sand Rock SpD ×1.5 (REPL `modify`), snow Ice Def, Ruin abilities |
| H17 | WeatherModifyDamage | MUL (priorityEvent) | `ch/scripts.ts:216` | rain / sun / primal, Mega Sol |
| H18 | ModifySTAB | REPL | `ch/scripts.ts:259` | Adaptability |
| H19 | Effectiveness | sum + REPL | `sim/pokemon.ts:2236-2237` | Freeze-Dry, Flying Press, Delta Stream |
| H20 | ModifyDamage | MUL | `ch/scripts.ts:296` | Reflect / Light Screen / Aurora Veil (2732), Life Orb, Expert Belt, Multiscale, Filter / Solid Rock, Friend Guard, resist berries, Fluffy, Punk Rock |
| H21 | Damage (pre-HP) | GATE / REPL | `sim/battle.ts:2138` | Focus Sash, Sturdy, Disguise, Rock Head (recoil), Magic Guard (non-move) |
| H22 | MoveEffects (`runMoveEffects` + `S/R Hit`) | data-driven + `onHit` switch | `sim/battle-actions.ts:1241-1372` | boosts / status / volatile / side / field / weather / terrain application; 72 status-move `onHit`s |
| H23 | Boost pipeline: ChangeBoost / TryBoost / AfterEachBoost / AfterBoost | REPL / GATE / reactions | `sim/battle.ts:2049-2100` | Contrary, Simple, Clear Body, Mirror Armor, Guard Dog, Clear Amulet, Defiant, Competitive, Rattled, Opportunist, Mirror Herb, White Herb |
| H24 | Status / volatile admission: Immunity, SetStatus, TryAddVolatile, AfterSetStatus | GATE + reactions | `sim/pokemon.ts:1734-1767, 2008-2019` | Safeguard, Misty / Electric Terrain, Insomnia / Vital Spirit, Limber, Own Tempo, Inner Focus (flinch), Leaf Guard, Flower Veil, Sweet Veil; Synchronize, Lum |
| H25 | Secondaries: ModifySecondaries + per-secondary roll | filter + RNG | `sim/battle-actions.ts:1392-1408` | Shield Dust, Covert Cloak (Past) |
| H26 | OnHit reactions: DamagingHit, AfterHit, AfterSubDamage, AfterMoveSecondary(+Self), EmergencyExit | side effects | `ch/scripts.ts:409-419, 573-585`; `sim/battle-actions.ts:600-607` | Rocky Helmet, Rough Skin, Static, Flame Body, Effect Spore, Weakness Policy, Stamina, Justified, Cursed Body, Knock Off, Rapid Spin, Eject Button, Red Card, Life Orb, Shell Bell, Emergency Exit (Champions) |
| H27 | DragOut / forceSwitch | GATE | `sim/battle-actions.ts:1414` | Suction Cups, Guard Dog, Ingrain, commanded |
| H28 | Update | reactions | after each hit (`ch/scripts.ts:536, 573`), each action (`sim/battle.ts:2926`) | Sitrus / Lum / pinch berries, status-cure abilities |
| H29 | Residual (with duration ticks) | ordered scan | `sim/battle.ts:2856` → `fieldEvent` | §4.3 |
| †H30 | SwitchIn / Start / End | reactions | `sim/battle-actions.ts:237` | Intimidate, weather / terrain setters, Ruin abilities |
| †H31 | ModifySpe / ModifyPriority / FractionalPriority | MUL / REPL | queue | Prankster, Gale Wings, Grassy Glide, Quick Claw, Tailwind, Choice Scarf, paralysis, Swift Swim / Chlorophyll / Sand Rush |
| †H32 | DisableMove / TrapPokemon / MaybeTrapPokemon | masks | `sim/battle.ts:1708-1746` | choicelock, Encore, Taunt, Disable, Fake Out (Champions), Shadow Tag |

**Implementation notes:**
- **Per hook, evaluate all candidate rows in a static canonical order** (sorted by `order, -priority, subOrder`) with the dynamic speed key applied by a fixed small sorting network over the ≤ 4 Pokémon holders. Then fold.
  - With `vmap`, a `lax.switch` over handler ids executes every branch. So split each hook into a MUL table (gather multipliers with predicate masks, a cheap tensor op) and a bespoke switch whose id space is limited to that hook's handlers (≤ ~30 typical).
- **Reproduce chain order** exactly for MUL hooks when ≥ 3 factors can co-occur. BasePower (item + Helping Hand + terrain + Technician) and ModifyDamage (screen + Life Orb + Expert Belt / Multiscale) are the realistic cases.
- **Per-target batching inside a move.** Accuracy for all targets, then crit+roll per target, then the damage application for all targets, then effects, then secondaries per target. RNG event ids should be keyed by (move-instance, target slot, kind, hit#), never by sequence.
- **Champions semantics to bake in:**
  - Sheer Force doesn't block AfterMoveSecondary / EmergencyExit;
  - `onAfterHit` fires even if the attacker fainted;
  - Unseen Fist ×0.25 through Protect;
  - para 1/8; sleep {2, 3, 3}; freeze ≤ 3 turns with 1/4 thaw;
  - PP formula and cap;
  - Trick Room `-speed`;
  - Fake Out disabled after the first action;
  - Encore rewrites the queued action.
- **Non-semantic RNG to drop:**
  - target samples inside `getActionSpeed` for spread moves;
  - `selfDrops`' unconditional `random(100)` when `self.chance` is undefined;
  - handler-tie shuffles. Choose a canonical tiebreak; the only observable effect is ordering among exactly-tied handlers, which for MUL hooks can change rounding by at most 1 HP and is otherwise invisible.
  - For differential testing, map PS draws to engine events by their calling function (as `harness.js` does) and **skip** these kinds.

### 7.3 Estimating handler counts per hook for a dex (method)

`research/ps-move/hook-census.js` implements the method:
1. Take the dex subset (moves / abilities / items), for example the usage-truncated v1 dex from the dex agent.
2. Add every condition those effects can create (`volatileStatus`, `sideCondition`, `pseudoWeather`, `weather`, `terrain`, `status`, secondaries' volatiles / statuses, `move.condition`) plus all of `data/conditions.ts`. These are always reachable: statuses, weather, confusion, flinch, stall, …
3. For every effect, enumerate its `on<Prefix?><Event>` keys (excluding `Order`/`Priority`/`SubOrder`) and the move callbacks (`basePowerCallback`, `damageCallback`, `beforeMoveCallback`, `beforeTurnCallback`, `priorityChargeCallback`). Bucket by event, prefix and effect kind.
4. Classify each handler heuristically from its source text: **constant** (non-function), **multiplier** (`chainModify`/`modify` with no state mutation), **gate** (returns false / null / NOT_FAIL / 0 / true, with no mutation), **pure-other**, or **side-effecting**.
   - Multipliers and constants go to tables.
   - Gates go to small predicate switches.
   - Side-effecting handlers are bespoke.
   - (In the census output, synthetic callback rows show as "constant"; read them as bespoke code.)
5. Remember **duration-only** conditions (no handler, just `duration`). They still need a residual slot.

**Upper bounds on the full Reg M-C legal pool** (510 moves, 216 abilities, 166 items, 119 conditions; output in `hook-census.json`). Format: effects defining a handler (multiplier / gate / side-effecting).

| Group | Counts |
|---|---|
| Hit (move `onHit`) | 72 (55 side-effecting) |
| BasePower | 60 (53 MUL) + `basePowerCallback` 29 |
| TryHit | 49 (28 gate) |
| ModifyDamage | 39 (21 MUL) |
| Residual | 42 |
| Try | 30 |
| ModifyMove | 29 |
| DamagingHit | 29 |
| Update | 25 |
| TryMove | 19 |
| BeforeMove | 17 |
| ModifyAtk / ModifySpA | 16 / 15 (all MUL) |
| SetStatus / TryAddVolatile / Immunity | 14 / 14 / 13 (all gate) |
| TryBoost | 13 |
| ModifySpe | 11 |
| ModifyType | 10 |
| ModifyAccuracy | 9 |
| Invulnerability | 8 |
| AfterHit | 8 |
| `damageCallback` | 7 |
| WeatherModifyDamage, ModifyCritRatio | 6 each |
| RedirectTarget, Effectiveness, AfterMoveSecondary | 5 each |
| ModifyDef | 4 |
| ModifySTAB | 1 |
| Start (switch-in / start) | 130 |
| TakeItem | 83, almost all Mega Stones' constant "can't be removed" (a table) |

**Scaling to the v1 dex.** The scope doc's 99%-usage dex at ≥1760 is about 158 moves, 65 abilities and 60 items, i.e. about 30–40% of the pool. Expect roughly 1/3 of these counts. For example, about 20 BasePower contributors (mostly MUL rows), about 15 TryHit gates, about 15 ModifyDamage rows, about 25 status-move `onHit`s, and about 15 residual handlers.

The census should be re-run on the exact dex list. Run `node hook-census.js moves.json abilities.json items.json` with JSON arrays of names.

### 7.4 Pitfalls found while tracing (each is a likely divergence source)

1. **Spread ×0.75 is decided before filtering.** It applies even if Protect, immunity or a miss leaves one target (`sim/battle-actions.ts:613`, Ex1).
2. **The ally comes first in `allAdjacent` target order** (`sim/pokemon.ts:831-834`). This affects RNG batching and `-damage` order.
3. **`onAlly*` includes the holder itself, and `onAny*` includes everyone.** Many handlers rely on explicit `target !== effectState.target` checks.
4. **ModifyDamage / BasePower / ModifyAtk use the attacker as the event target.** Defender-side handlers are `onSource*`. ModifyDef/SpD are the reverse (§3.3).
5. **The final modify runs only for non-negative integer relays.** Handlers that `return this.modify(x, 1.5)` (sandstorm SpD, Hustle Atk) replace the relay immediately, *before* later chained modifiers are applied at the end.
6. **Stat rounding.** The crit multiplier is `trunc(d·1.5)`, not `modify`. Paralysis is `floor(spe·50/100)` after `finalModify`. Stat-stage multiplication uses floor, not the 4096 rounding.
7. **Protect.** The first use draws no RNG. Consecutive uses draw `random(3^k)`. A failure deletes `stall`, so the next use is free again (verified).
8. **`NOT_FAIL` vs. `false`** decides whether `moveThisTurnResult` is null or false (Stomping Tantrum, etc.) and whether `-fail` is printed.
9. **Weather's last turn:** the end message happens instead of upkeep, so there is no sandstorm chip that turn. Expiry happens at the handler's residual position.
10. **Champions `getHealth` reports floor-percent HP** (`sim/pokemon.ts:2087-2095`). This is relevant only for the observation encoding, not for mechanics.

---

## Appendix A: script index (`research/ps-move/`)

| File | Purpose | How to run |
|---|---|---|
| `harness.js` | Battle builder; PRNG proxy (log/force by calling PS function); event tracer | library. `PS_DIST` env overrides the default `/tmp/psrepo-move/ps/dist/sim`. |
| `damage-examples.js` | 4 verified damage examples × 16 rolls vs. the reference implementation | `node damage-examples.js` → "ALL EXAMPLES VERIFIED" |
| `intermediates.js` | Step-by-step intermediates for §3.7 | `node intermediates.js` |
| `verify-tables.js` | Stat-stage integer form, `modify` closed form, chain order-dependence | `node verify-tables.js` |
| `trace-turn.js` → `trace-turn1.txt` | Full nested event trace of a doubles turn | `node trace-turn.js` |
| `trace-more.js` → `trace-start.txt`, `trace-turn-b{1,2,3}.txt` | Battle-start switch-ins; Protect / crit / KO / status / Tailwind / Icy Wind turns | `node trace-more.js` |
| `rng-order.js`, `smoke.js` | RNG draw order with callers | `node rng-order.js` |
| `durations.js` | Tailwind / Trick Room / Reflect / weather expiry turns; Protect-chain draws | `node durations.js` |
| `hook-census.js` → `hook-census.json` | Handler census per event for a dex subset | `node hook-census.js [moves.json abilities.json items.json]` |
