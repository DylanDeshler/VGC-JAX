# Pokémon Showdown doubles: turn-level control flow, as an implementation spec

**Target:** `[Gen 9 Champions] VGC 2026 Reg M-C`: format id `gen9championsvgc2026regmc`, mod `champions`, gameType `doubles`.
**Oracle:** Pokémon Showdown commit `d601443` ("Add help for /setbot").
**Date:** 2026-10-21.
**Scope:** turn-level control flow only. That means start, team preview, requests, choices, the queue, `runAction`, faints, residuals, end of turn, positions, protocol and serialization, plus a JAX mapping. Move execution, damage and event-system internals are covered by a separate document.

## Conventions

- **Citations.** `file:line` paths are relative to the PS repo root, for example `sim/battle.ts:2718`.
- **[C]** marks behaviour overridden or added by the Champions mod (`data/mods/champions/*`), or by a `currentMod.startsWith('champions')` branch in `sim/`.
- **[not in Reg M-C]** marks code paths that exist in PS but can't be reached in this format at this commit, because nothing that triggers them is legal (checked with `research/ps-loop/legality.js`).
- **Verified** means I confirmed the claim by running PS. Scripts are in `research/ps-loop/`; see the Appendix.
- **Checkout note.** The original checkout at `/tmp/psrepo/ps` was deleted partway through this work. The later runs used an identical source tree (commit `d601443`, copied from `/tmp/psrepo-delta/ps`) built separately at `/tmp/ps-turnloop`. Neither checkout was modified. All scripts take `PS_DIST=/path/to/ps/dist`.

---

## 0. Key facts

1. **Team preview.** Each side picks an ordered list of exactly 4 of its 6 mons (`Picked Team Size = Auto` gives 4 for doubles: `sim/dex-formats.ts:316-322`).
   - The `team` actions rebuild `side.pokemon` to contain only those 4, in the chosen order (`sim/battle.ts:2803-2810`). Indices 0 and 1 are the leads in slots a and b.
   - `pokemonLeft` becomes 4 (`sim/battle.ts:2725`).
   - The 2 unpicked mons no longer exist in the battle state, apart from `side.team`.
2. **Simultaneous switch-ins are batched only sometimes.**
   - Leads at battle start and end-of-turn or mid-turn replacements (`instaswitch`, order 3) are batched into one `fieldEvent('SwitchIn')`. Its handlers are sorted by priority tier, then speed, then subOrder.
   - Ordinary start-of-turn switches (order 103) are **not** batched. Each switch's `runSwitch` (order 101) is inserted ahead of the remaining switches, so its Intimidate or weather fires before the next mon even enters (verified).
3. **Dynamic speed (Gen 8+).**
   - After every action whose successor in the queue is a `move` (or `runDynamax`), PS calls `updateSpeed()` on all actives.
   - It then recomputes `priority` and `speed` for **every** queued action with a pokemon and re-sorts the whole queue, re-shuffling ties (`sim/battle.ts:2975-2984`).
   - So Mega Evolution speed, Tailwind, Icy Wind, Trick Room, paralysis and a terrain-based Grassy Glide all take effect in the same turn (verified for Mega and Icy Wind).
   - `fractionalPriority` (Quick Claw, Quick Draw) is rolled **once**, at commit time (`sim/battle-queue.ts:249`).
4. **[C] Speed key.** `pokemon.getActionSpeed()` is overridden: `speed = getStat('spe')`, negated under Trick Room. There is no `10000 - speed` wraparound and no 13-bit truncation (`data/mods/champions/scripts.ts:46-54` vs `sim/pokemon.ts:662-670`). The 10000 cap in `getStat` still applies (`sim/pokemon.ts:658`).
5. **Order classes.**
   - team 1, start 2, instaswitch 3 (also prioritized actions), beforeTurn 4, beforeTurnMove 5 (Counter, Mirror Coat), revivalblessing 6
   - runSwitch 101, switch 103, megaEvo 104, priorityChargeMove 107 (Focus Punch, Beak Blast, Chilly Reception)
   - move 200, Quashed move 201, residual 300

   Source: `sim/battle-queue.ts:174-195`. Mega Evolution happens after all switches and before any move. Its new speed is used for that turn's moves.
6. **Mid-turn decision points.** After an action completes, if any active has `switchFlag` set, PS emits a `switch` request to the side(s) with flagged slots and `wait` to the others, then pauses (`sim/battle.ts:2936-2969`).
   - Flags come from U-turn, Volt Switch, Flip Turn, Parting Shot, Baton Pass, Shed Tail, Chilly Reception, Revival Blessing, Eject Button and Emergency Exit. Red Card and Roar/Whirlwind/Dragon Tail/Circle Throw are random drags with **no** decision.
   - When choices are committed, the new `instaswitch` actions are sorted and placed **in front of** the untouched remainder of the queue (`sim/battle.ts:3063-3077`).
   - Fainted mons are never replaced mid-turn. They are replaced only when the queue is empty, which in practice means after residuals (`sim/battle.ts:2909-2910`, `sim/battle.ts:2540-2550`).
7. **No Endless Battle Clause in Reg M-C.** The rule table lacks `endlessbattleclause` (verified). The only limit is the hard tie when the turn counter would reach 1001 (`sim/battle.ts:1856-1860`).
8. **Win check after every action.** `resolveFaints({checkWinIfNoFaints:true})` runs in every `runAction` tail and after every residual handler.
   - If every side is out of mons at once, the side of the **last-processed faint** wins (Gen 5+): `sim/battle.ts:2659-2662`.
9. **Legality comes from `side.ts` code, not request JSON.** The JAX mask should match PS's *acceptance* (`side.choose`), not the request JSON.
   - For the "last active" slot, the request hides trapping and hidden disables: `maybeTrapped` / `maybeDisabled`.
   - [C] Fake Out and First Impression become **disabled** (unselectable) once the user has executed any move action since entering (`data/mods/champions/moves.ts:349-356, 381-389`).
10. **Diff source.** `State.serializeBattle` captures complete engine state, including durations, counters, PP, queue and PRNG, and round-trips correctly mid-turn (verified: deserializing at a U-turn pause and continuing gives an identical log). Use it for state diffs at decision points, and use the omniscient protocol log for event ordering inside a turn.

---

## 1. Format facts that shape control flow

### 1.1 Resolved rule table (verified: `research/ps-loop/ruletable.js`)

```
flatrules, obtainable(...), teampreview, speciesclause, nicknameclause, itemclause, adjustlevel,
pickedteamsize, minteamsize, cancelmod, -tag:mythical, -tag:restrictedlegendary, vgctimer
(timerstarting, timergrace, timeraddperturn, timermaxperturn, timermaxfirstturn, timeoutautochoose,
dctimerbank), openteamsheets
pickedTeamSize 4, minTeamSize 6, maxTeamSize 6, adjustLevel 50, endlessbattleclause: false, cancelmod: true
```

- **Format definition.** `config/formats.ts:295-301` gives `ruleset: ['Flat Rules', 'VGC Timer', 'Open Team Sheets']` with `gameType: 'doubles'`.
- **[C] Flat Rules.** Redefined in `data/mods/champions/rulesets.ts:35-40`: Obtainable, Team Preview, Species Clause, Nickname Clause, Item Clause = 1, Adjust Level = 50, Picked Team Size = Auto, Min Team Size = 6, Cancel Mod. It has **no** Endless Battle Clause.
- **[C] Team Preview rule.** Overridden in `data/mods/champions/rulesets.ts:41-61`. It prints `clearpoke`/`poke` lines and calls `makeRequest('teampreview')`.
- **Cancel Mod** sets `battle.supportCancel = true` (`data/rulesets.ts:1352-1359`). This only affects `noCancel` in requests and undo; it has no mechanical effect.
- **VGC Timer** (`data/rulesets.ts:778-786`) and **Open Team Sheets** (`data/rulesets.ts:1960-1981`) are server-side or presentation only. Open Team Sheets just emits `uhtml` lines.
- **Doubles geometry.** `activePerHalf = 2` (`sim/battle.ts:222-224`); `side.active = [null, null]` (`sim/side.ts:248-251`).

### 1.2 Champions overrides that touch turn flow

| Override | Location | Effect on control flow |
|---|---|---|
| `pokemon.getActionSpeed` | `data/mods/champions/scripts.ts:46-54` | Speed key is `getStat('spe')`, or its negation under Trick Room. No wraparound. |
| `pokemon.formeChange` | `scripts.ts:56-121` | Mega Evolution emits `detailschange` then `-mega POKEMON|BASESPECIES|STONE`. Megas **do not revert on faint** (comment at `scripts.ts:55`). |
| `pokemon.clearVolatile` | `scripts.ts:123-176` | Same as base, but also resets `timesAttacked`. Called on switch-out and faint. |
| `actions.canTerastallize → null` | `scripts.ts:179-181` | No Tera. Requests never carry `canTerastallize`, and a `terastallize` choice is rejected (`sim/side.ts:806-810`). |
| `actions.canMegaEvo` | `scripts.ts:182-194` | `item.megaStone?.[species.name]`. Computed once at construction (`sim/pokemon.ts:511`). |
| Curse target | `sim/battle-queue.ts:263-266` | [C] For a non-Ghost user, the queued Curse action's target becomes `self`. |
| `encore` condition | `data/mods/champions/moves.ts:309-340` | On start, if the target still has a queued move with a different id (and no Mental Herb), `queue.changeAction` re-queues it as the encored move with a **random target** and recomputed priority. Base `onOverrideAction` is removed. |
| `fakeout`, `firstimpression` `onDisableMove` | `moves.ts:349-356, 381-389` | Disabled at selection once `activeMoveActions > 0`. |
| `belch`, `stuffcheeks` | `moves.ts:52-55, 989-992` | Base `onDisableMove` removed, so they are always selectable. |
| `ejectbutton` | `data/mods/champions/items.ts:262-277` | Sets `switchFlag = true`. At most one flag per action, because it is skipped if any active already has `switchFlag === true`. Also skipped for Commander pairs. |
| `emergencyexit`, `wimpout` | `data/mods/champions/abilities.ts:26-34, 101-109` | Set `switchFlag = true` when HP crosses 1/2. |
| `naturalcure`, `regenerator` | `abilities.ts:54-75` | Switch-out effects. Natural Cure's `onCheckShow` is removed. |
| `runaway` | `abilities.ts:76-86` | Un-traps at priority −10, like Shed Shell. |

### 1.3 Legality of flow-relevant mechanics in Reg M-C (verified: `legality.js`)

**Legal and relevant:**
- **Self-switch moves:** U-turn (52 species), Volt Switch, Flip Turn, Parting Shot, Baton Pass, Shed Tail, Chilly Reception, Revival Blessing (Pawmot).
- **Drag moves:** Roar, Whirlwind, Dragon Tail, Circle Throw.
- **Queue manipulation:** Ally Switch, Instruct (Oranguru only), After You, Quash, Round.
- **Charge and pre-turn moves:** Focus Punch, Beak Blast (Toucannon), Counter, Mirror Coat.
- **Other moves:** Encore, Trick Room, Tailwind, Icy Wind, Electroweb, Fake Out, Follow Me, Rage Powder, Perish Song, Future Sight, Wish, Healing Wish, Imprison, Mean Look, Block, Jaw Lock, Spirit Shackle, Octolock, Sucker Punch, Upper Hand, Grassy Glide, hazards (Stealth Rock, Spikes, Toxic Spikes, Sticky Web).
- **Items:** Eject Button, Red Card, Quick Claw, Choice Scarf, Shed Shell, Mental Herb, White Herb. There are 81 Mega Stones.
- **Abilities:** Emergency Exit (Golisopod), Intimidate (25 species), Drought, Drizzle, Sand Stream, Snow Warning, Electric/Psychic/Grassy Surge, Prankster, Gale Wings, Quick Draw (Slowbro-Galar), Stall, Shadow Tag (Gengar-Mega), Run Away, Unnerve, Pressure, Stalwart, Lightning Rod, Healer.

**[not in Reg M-C] (Past, or no legal holder):**
- **Abilities:** Commander (Tatsugiri and Dondozo are Past), Dancer, Protosynthesis, Quark Drive, the Ruin abilities, Misty Surge, Orichalcum Pulse, Hadron Engine, Arena Trap, Magnet Pull, Wimp Out, Triage, Mycelium Might.
- **Items:** Booster Energy, Eject Pack, Custap Berry, Assault Vest, Covert Cloak, Clear Amulet, Mirror Herb, Room Service, Lagging Tail, Full Incense, Choice Band, Choice Specs.
- **Moves:** Pursuit, the Pledge moves, Shell Trap, Teleport, Sky Drop, Me First, Metronome.

These are still described below where the brief asks for them, but they're marked as not reachable.

---

## 2. Battle start and team preview

### 2.1 Construction (`new Battle`, `sim/battle.ts:192-317`)

- Logs `|t:|`, then `|gametype|doubles` (`sim/battle.ts:294`).
- Sets `activePerHalf = 2` and `speedOrder = [0,1,2,3]` (`sim/battle.ts:264-267`).
- Creates the queue and actions objects, then calls `setPlayer` for each side.
- Each `Side` builds `side.pokemon` from the 6 submitted sets, in submitted order:
  - `position = index` (`sim/side.ts:302-309`);
  - `pokemonLeft = 6` (`sim/side.ts:259`);
  - `active = [null, null]`.
- Each Pokemon gets `speed = 0` (`sim/pokemon.ts:509`) and computes `canMegaEvo` (`sim/pokemon.ts:511`).
- Rules with event handlers are attached as pseudo-weathers (`sim/battle.ts:297-309`). This has no effect for Reg M-C turn flow.

### 2.2 `start()` (`sim/battle.ts:1922-1986`)

```
start():
  link side.foe pointers                                    # :1944-1951
  log |gen|9, |tier|..., [|rated|]                          # :1953-1959
  format.onBegin(); for each rule: rule.onBegin()           # :1961-1966   (rule messages; cancelmod -> supportCancel)
  runPickTeam()                                             # :1982
  queue.addChoice({choice:'start'})                         # :1983  -> order 2, speed 1
  midTurn = true                                            # :1984
  if (!requestState) turnLoop()                             # :1985  (not taken: requestState == 'teampreview')
```

### 2.3 `runPickTeam` and the `teampreview` request (`sim/battle.ts:1994-2021`)

1. Calls `format.onTeamPreview` (none for Reg M-C), then `rule.onTeamPreview` for each rule in rule-table order.
   - [C] The Team Preview rule (`data/mods/champions/rulesets.ts:43-60`) logs `|clearpoke|`, then one `|poke|pN|DETAILS|` per mon (all 12), then `makeRequest('teampreview')`.
   - Open Team Sheets then logs a split `uhtml|otsrequest|...` (`data/rulesets.ts:1970-1975`).
2. `makeRequest('teampreview')` (`sim/battle.ts:1391-1422`):
   - sets `requestState = 'teampreview'`;
   - calls `side.clearChoice()` for both sides;
   - logs `|teampreview|4` (`:1405-1411`);
   - builds requests through `getRequests` (`:1448-1454`): `{teamPreview: true, maxChosenTeamSize: 4, side: side.getRequestData()}`.
3. At this point `side.pokemon` still has 6 mons. The request's `active` field is a positional artefact (`position < 2`) (`sim/pokemon.ts:1179`).

### 2.4 Parsing the `team` choice (`Side.chooseTeam`, `sim/side.ts:1031-1095`)

Input is `team <data>`. `Side.choose` keeps the whole string as one choice (`sim/side.ts:1202`).

```
chooseTeam(data):
  require requestState == 'teampreview'
  if data is "[...]": bracketed, strip brackets
  positions = data ? split(data, ',' if bracketed or contains ',' or >=10 mons else '') .map(x => int(x)-1)
            : [0..5]                                   # autoChoose
  pickedTeamSize = min(6, 4) = 4                       # side.ts:1027-1029
  if !bracketed: positions = positions[0:4]            # truncate long lists
  if len(positions) < 4: append the lowest unused indices 0,1,2,... until len == 4
  error unless len == 4; each pos integer in [0,6); no duplicates
  (ruleTable.onChooseTeam: none in Reg M-C)
  for index, pos in positions: choice.actions.push({choice:'team', index, pokemon: side.pokemon[pos], priority: -index})
```

**Verified examples** (`scenarios.js preview`):

| Input | Resulting order (0-based team slots) |
|---|---|
| `team 12` | 0,1,2,3 |
| `team 6513` | 5,4,0,2 |
| `team [6,5]` | 5,4,0,1 |
| `team 7123` | error |
| `team 1,1,2,3` | error |

`isChoiceDone()` is true once `actions.length >= 4` (`sim/side.ts:543-545`).

**Action-space note.** The ordered 4-tuple matters:
- tuple positions 0 and 1 become slots a and b;
- positions 2 and 3 become bench indices 2 and 3.

Bench order has no mechanical effect (it only renumbers later `switch N` choices), so the policy-relevant choice is (ordered lead pair) × (bench set), which is 180 options. PS accepts all 360 orderings.

### 2.5 Committing the preview (`commitChoices`, `sim/battle.ts:3056-3088`)

```
commitChoices():
  updateSpeed()                      # no actives yet -> no-op
  oldQueue = queue.list              # [start]
  queue.clear()
  for side: side.commitChoices()     # -> queue.addChoice(team actions)  (side.ts:1186)
  clearRequest()
  queue.sort()                       # speedSort by (order=1, priority=-index desc, speed desc)
  queue.list.push(...oldQueue)       # -> [team x8 ..., start]
  requestState = ''
  turnLoop()
```

- `resolveAction` gives each `team` action `order 1` and calls `getActionSpeed`, which sets `speed = pokemon.getActionSpeed()` (`sim/battle-queue.ts:275`, `sim/battle.ts:2705-2715`).
- Within a side, order is fixed by `priority = -index`. Across sides, ties on equal speed consume PRNG shuffles but don't change the outcome.

`runAction('team')` (`sim/battle.ts:2803-2810`):

```
if action.index == 0: side.pokemon = []
side.pokemon.push(action.pokemon); action.pokemon.position = action.index
return            # no post-action pipeline
```

After the preview, `side.pokemon` has length 4 (verified).

### 2.6 `runAction('start')`: the leads (`sim/battle.ts:2723-2756`)

```
for side: if side.pokemonLeft: side.pokemonLeft = side.pokemon.length (=4); log |teamsize|pN|4
log |start|
for each pokemon: singleEvent('BattleStart', species condition)        # forme stuff; nothing for Reg M-C
format.onBattleStart; rules' onBattleStart                            # OTS: uhtmlchange
for side in [p1, p2]: for i in [0, 1]: actions.switchIn(side.pokemon[i], i)
midTurn = true
```

**`switchIn(pokemon, pos, sourceEffect=null, isDrag=false)`** (`sim/battle-actions.ts:79-215`). For a lead, there is no old active, so the switch-out half is skipped.

- Sets `isActive = true`, `side.active[pos] = pokemon`, `activeTurns = 0`, `activeMoveActions = 0`, `moveSlot.used = false` (`:155-161`).
- Creates fresh `abilityState` and `itemState`. `initEffectState` assigns increasing `effectOrder` (`sim/battle.ts:3380-3390`).
- Fires `runEvent('BeforeSwitchIn')`, then logs `|switch|POKEMON|DETAILS|HP` (`:187-192`) and increments `previouslySwitchedIn`.
- Not a drag, so it calls `queue.insertChoice({choice:'runSwitch', pokemon})` (`:208-212`).
  - `insertChoice` first calls `pokemon.updateSpeed()` (`sim/battle-queue.ts:384-386`). This is the first time a lead's `speed` field is set.
  - It then inserts the action at order 101 by `comparePriority`, choosing a random index among equal entries (`sim/battle-queue.ts:389-408`).

**The post-action tail after `start`** (§6.3):
- no phazing and no faints;
- `nextAction` is a `runSwitch`, so `checkFainted` is not called;
- the `Update` event is explicitly skipped for `start` (`sim/battle.ts:2925`);
- no switch flags, and no re-sort because the next action is not a move.

### 2.7 `runSwitch` and simultaneous-entry handling (`sim/battle-actions.ts:228-249`)

```
runSwitch(pokemon):
  switchersIn = [pokemon]
  while queue.peek().choice == 'runSwitch': switchersIn.push(queue.shift().pokemon)   # BATCH
  allActive = getAllActive(includeFainted=true)
  speedSort(allActive)                       # by pokemon.speed (cached), ties -> PRNG shuffle
  speedOrder = allActive.map(p => p.getFieldPositionValue())      # side.n + 2*position
  fieldEvent('SwitchIn', switchersIn)
  for p in switchersIn: if p.hp: p.isStarted = true
```

**`fieldEvent('SwitchIn', targets)`** (`sim/battle.ts:489-575`) collects handlers as follows:
- field `onFieldSwitchIn` and side `onSideSwitchIn` (none in practice);
- for **every** active (not only switchers), its `onAnySwitchIn` handlers (`:502-504`). Examples: Commander, White Herb, Opportunist.
- for each active **in `targets`**:
  - its own `onSwitchIn` handlers: status, volatiles, ability, item, species, slot conditions (`sim/battle.ts:1111-1170`);
  - side-condition `onSwitchIn` handlers held by that pokemon (hazards);
  - field `onSwitchIn`;
  - format handlers (`:506-509`).

**Abilities and items run `onStart` during `SwitchIn`.** `getCallback` maps a missing `onSwitchIn` to `onStart` for Ability and Item effects (`sim/battle.ts:1030-1036`). This is how Intimidate, Drizzle, Drought, Sand Stream, Snow Warning, the Surges, Pressure, Unnerve, Frisk and Choice Scarf fire.

**Sort key** (`resolvePriority`, `sim/battle.ts:957-1023`, sorted by `speedSort` with `comparePriority`, `sim/battle.ts:409-416`):

1. **`order`**: `on<Cb>Order`, or last if absent.
2. **`priority`** (descending): `onSwitchInPriority` / `onAnySwitchInPriority`.
3. **`speed`** (descending) = `pokemon.speed - speedOrder.indexOf(fieldPos)/4` (`:1015-1020`).
   - The fractional offset is assigned **once per batch**, from the speed-tie-resolved `speedOrder`.
   - So no two pokemon tie, and **all handlers of a faster mon in a priority tier run before any handler of a slower mon in that tier**.
4. **`subOrder`** (ascending). If not explicit, it defaults by effect type (`:964-999`):

   | Effect type | subOrder |
   |---|---|
   | Status (not in the table) | 0 |
   | volatile Condition | 2 |
   | slot condition | 3 |
   | side condition | 4 |
   | field / weather / format | 5 |
   | Ability | 7 (Poison Touch, Perish Body = 6) |
   | Item | 8 |
   | Stall | 9 |

5. **`effectOrder`** (ascending): set for `*SwitchIn` callbacks from `state.effectOrder`, which is creation order (`:1001-1007`).

**Execution.** Each handler runs via `singleEvent`:
- handlers whose holder has fainted are skipped (`:517-519`);
- stale handlers are skipped if the effect was removed earlier in the loop (`:530-563`);
- `resolveFaints()` runs after every handler and the loop aborts if the battle ended (`:572-573`).

**SwitchIn priority tiers in Champions** (`research/ps-loop/switchin_handlers.js`, Past effects excluded):

| Tier | Handlers |
|---|---|
| +2 | Neutralizing Gas, Tera Shift |
| +1 | Unnerve, Klutz, As One |
| 0 | all other ability `onStart`s (Intimidate, weather and terrain setters, Pressure, Frisk, Trace, ...), Air Balloon, Choice Scarf, Metronome item, hazards (Spikes, Stealth Rock, Toxic Spikes, Sticky Web), Healing Wish slot condition (`healreplacement`), `tox` status reset |
| −1 | terrain seeds, Mimicry |
| −2 | Commander (`onAnySwitchIn`), White Herb (`onAnySwitchIn`), Hospitality, Forecast, Flower Gift, Ice Face, Costar, Protosynthesis / Quark Drive |
| −3 | Opportunist (`onAnySwitchIn`) |

Consequences:
- **Weather and terrain wars at battle start.** Within tier 0, the slowest setter acts last and wins.
- **Seeds** (tier −1) react after every tier-0 terrain setter, regardless of speed.
- **Same mon, same tier.** Hazards (subOrder 4) fire before that mon's ability (7), which fires before its item (8).

**Ties.** `speedSort` (`sim/battle.ts:434-465`) is a selection sort. Each maximal group of equal elements is moved into place and then `prng.shuffle`d (`:460-462`), consuming k−1 draws for a k-way tie.

### 2.8 Which simultaneous entries are batched (verified: `scenarios.js doubleswitch`, `fewbench`)

| Situation | Action that switches | Batched? | Why |
|---|---|---|---|
| Leads at battle start | `start` → `switchIn` ×4 | **yes**, one SwitchIn with 4 targets | All 4 `runSwitch` (101) actions are queued before any runs. |
| End-of-turn faint replacement | `instaswitch` (3) ×k | **yes** | `instaswitch` (3) sorts before `runSwitch` (101), so all switches happen first and the runSwitches batch. |
| Mid-turn U-turn / Eject Button / Emergency Exit decision | `instaswitch` | yes if several flags are committed together; usually 1 | Same reason. |
| Start-of-turn chosen switches | `switch` (103) | **no** | Each `runSwitch` (101) is inserted *before* the remaining 103s. Switch A, then A's abilities, then switch B, then B's abilities. |
| Drag (Roar, Dragon Tail, Red Card) | `dragIn` → `switchIn(isDrag)` | n/a | `runSwitch` is called inline (`sim/battle-actions.ts:208-209`). |

**Verified trace (start-of-turn switches).**
- p1 Talonflame switched to Incineroar and p2 Kingambit switched to Arcanine.
- Talonflame, the faster *outgoing* mon, switched first. Incineroar's Intimidate hit Kingambit, which triggered Defiant, *before* Kingambit switched out.
- Then Arcanine entered and Intimidated.

**Order among non-batched switches** is the switch action's `speed`. That is the **outgoing** (`action.pokemon`) mon's speed at commit time, negated under Trick Room [C] (`sim/battle.ts:2713`, `sim/side.ts:1010-1014`).

**Order of replacement `instaswitch`es** is also by `action.pokemon` = the **fainted** mon being replaced (`sim/side.ts:1012`).
- A fainted mon has `isActive = false` and cleared boosts.
- `runEvent('ModifySpe')` only collects field and battle handlers for an inactive target (`sim/battle.ts:1067`).
- So the key is effectively its stored Speed stat (negated under Trick Room). This only affects the order of `|switch|` lines, because the effects are batched.

### 2.9 Start sequence: canonical pseudo-code

```
INIT:  requestState='teampreview'; queue=[start]; midTurn=true
TEAM_PREVIEW decision (both sides)
COMMIT: queue=[team...(sorted), start]
RUN team actions               -> side.pokemon := picked 4 in order
RUN start                      -> pokemonLeft=4; |teamsize| x2; |start|; switchIn(p1[0],0), switchIn(p1[1],1), switchIn(p2[0],0), switchIn(p2[1],1)
                                 (4x |switch|, 4 runSwitch queued)
RUN runSwitch (batch of 4)     -> speedOrder; fieldEvent('SwitchIn',[4]) ; isStarted
TAIL: queue empty -> checkFainted(); Update; EmergencyExit(first switcher); maybe 'switch' request
endTurn()                      -> turn=1 ... |turn|1 ; makeRequest('move')
```

---

## 3. Requests

### 3.1 `makeRequest(type)` and `getRequests(type)` (`sim/battle.ts:1391-1479`)

```
makeRequest(type):
  if type: requestState = type; for side: side.clearChoice()     # clearChoice computes forcedSwitchesLeft/PassesLeft (§4.6)
  for side: side.activeRequest = null
  if type == 'teampreview': log |teampreview|4
  requests = getRequests(type); side[i].activeRequest = requests[i]; sentRequests = false
  assert not every side.isChoiceDone()

getRequests(type):
  'switch':      for side with pokemonLeft:
                   table = side.active.map(p => !!p?.switchFlag)
                   if any(table): req = {forceSwitch: table, side: side.getRequestData()}
  'teampreview': req = {teamPreview: true, maxChosenTeamSize: 4, side}
  'move':        for side with pokemonLeft: req = {active: side.active.map(p => p?.getMoveRequestData()), side}
  for each side: if req: (if !supportCancel or <2 requests) req.noCancel = true
                 else req = {wait: true, side}
```

**When requests are made.** Only three call sites exist in base PS and Champions:
- `endTurn` → `'move'` (`sim/battle.ts:1817`);
- `runPickTeam` / the Team Preview rule → `'teampreview'`;
- the `runAction` tail → `'switch'` (`sim/battle.ts:2966`).

So **every `move` request is preceded by `endTurn()`**, which recomputes disables and trapping (§8.4).

**Transport.** Requests are emitted as `|request|JSON` side updates by `sendUpdates()` (`sim/battle.ts:3325-3332`). `BattleStream` calls it after each write (`sim/battle-stream.ts:46`). `side.activeRequest` holds the same object, so reading it directly is equivalent.

### 3.2 Request shapes (TypeScript types in `sim/side.ts:64-164`)

```jsonc
// 'move' (example; verified output, abridged)
{"active":[{"moves":[{"move":"Heat Wave","id":"heatwave","pp":11,"maxpp":12,"target":"allAdjacentFoes","disabled":false}, ...],
            "trapped":true?, "maybeTrapped":true?, "maybeDisabled":true?, "maybeLocked":true?, "canMegaEvo":true?},
           { ...slot b... }],
 "side":{"name":"A","id":"p1","pokemon":[{"ident":"p1: Typhlosion","details":"Typhlosion-Hisui, L47, F","condition":"132/159",
          "active":true,"stats":{...},"moves":[...],"baseAbility":"blaze","item":"choicescarf","pokeball":"pokeball",
          "ability":"blaze","commanding":false,"reviving":false}, ... 4 entries after preview ...]},
 "noCancel":true?}
// 'switch'
{"forceSwitch":[true,false],"side":{...}}
// 'teampreview'
{"teamPreview":true,"maxChosenTeamSize":4,"side":{... 6 entries ...}}
// 'wait'
{"wait":true,"side":{...}}
```

**Per-mon move data** (`getMoveRequestData`, `sim/pokemon.ts:1104-1172`):

- **`lockedMove = getLockedMove()`** (`:1105`) uses the `LockMove` priority event (`sim/pokemon.ts:970-973`). If it is set:
  - `this.trapped = true` (`:1108`);
  - `moves = [{move, id}]` with **no** pp, target or disabled fields (`:994-999`); for recharge it is `[{move:'Recharge', id:'recharge'}]` (`:988-992`).
  - Sources: `lockedmove` (Outrage, Petal Dance, Raging Fury), `twoturnmove` (Solar Beam, Meteor Beam, Dig, Fly...), `mustrecharge` ('recharge'), Uproar. See `data/conditions.ts:253-323, 364-378`.
- **Otherwise** `moves = getMoves(null, restrictData=isLastActive)`:
  - a slot's `disabled` = `moveSlot.disabled`, or true if `pp <= 0` (`:1037-1044`);
  - `disabled === 'hidden'` is shown as `!restrictData` (`:1046-1048`);
  - the request `target` is adjusted for Curse (non-Ghost becomes `self`), Pollen Puff under Heal Block (`adjacentFoe`) and Tera Starstorm (`:1018-1036`);
  - if **no** move is enabled, the list is empty (`:1062`) and the request becomes `[{move:'Struggle', id:'struggle', target:'randomNormal', disabled:false}]` (`:1124-1127`).
- **Mega flag.** `canMegaEvo` is included only when there is no lockedMove and no Struggle, and `pokemon.canMegaEvo` is truthy (`:1158-1160`). `runMegaEvo` clears `canMegaEvo` for **every** mon on the side (`sim/battle-actions.ts:1958-1966`).
- [C] There is never `canTerastallize`, `canDynamax` (gen ≠ 8) or `canZMove` (Z-crystals are Past).

**`getSwitchRequestData`** (`sim/pokemon.ts:1174-1211`):
- `active = position < 2`;
- `condition` uses the exact HP string `"cur/max[ status]"` or `"0 fnt"`;
- `commanding` and `reviving` are present for gen 9;
- [C] no `teraType` (`:1206`).

### 3.3 What makes a slot need a choice

| Request | Slot requires an action | Auto-passed (the player may still type `pass`) |
|---|---|---|
| `move` | active, not fainted, no `commanding` volatile | fainted active (no replacement available), `commanding` (`sim/side.ts:1312-1321`) |
| `switch` | `switchFlag` truthy | slots without `switchFlag` (`sim/side.ts:1322-1327`) |
| `teampreview` | side must give 4 `team` actions | n/a |
| `wait` | nothing | n/a |

`isChoiceDone()` (`sim/side.ts:539-550`) returns true when:
- `requestState == ''` (a wait request), or
- `forcedSwitchesLeft == 0` and `actions.length >= 2`, after `getChoiceIndex()` has inserted auto-passes.

The battle commits when **all** sides are done (`sim/battle.ts:3118-3127`, `:3036`). A side with a `wait` request is always done.

### 3.4 Information hiding in the request (and why the JAX mask shouldn't copy it)

`isLastActive()` (`sim/pokemon.ts:1213-1220`) is true for the highest-index living active. In doubles that is slot b if alive, otherwise slot a.

**Non-last-active (normally slot a)** (`sim/pokemon.ts:1133-1140`):
- `maybeDisabled`, `maybeLocked` and `maybeTrapped` are cleared;
- `trapped: true` is shown if `this.trapped` is truthy (including `'hidden'`) and (hard-locked or a bench mon exists).

**Last-active** (`:1141-1156`):
- hidden disables (Imprison: `disableMove(id, true)` at `data/moves.ts:9506-9511`) appear **enabled**, with `maybeDisabled: true`;
- hidden traps (Shadow Tag `tryTrap(true)`, `data/abilities.ts:4157-4169` and `sim/pokemon.ts:1628-1633`) show `maybeTrapped: true` instead of `trapped: true`.

**Verified** (`scenarios.js shadowtag`): with Mega Gengar's Shadow Tag active, p2 slot a showed `trapped:true` and slot b showed `maybeTrapped:true`. Both `switch` attempts were rejected with "The active Pokémon is trapped".

**Rule for the harness.** PS acceptance uses the *true* state:
- `chooseMove` re-reads `pokemon.getMoves()` with no restriction, so a `'hidden'` disable is rejected (`sim/side.ts:627, 729-762`);
- `chooseSwitch` tests `pokemon.trapped`, which is truthy for `'hidden'` (`sim/side.ts:985`).

So the JAX mask should model the true state. When comparing to PS, treat `maybeTrapped` / `maybeDisabled` request entries as "unknown". Better still, enumerate legality through `side.choose` (§4.10).

---

## 4. Choice parsing and legality (`sim/side.ts`)

### 4.1 Grammar (`Side.choose`, `sim/side.ts:1189-1305`)

```
choose(input):
  error if requestState == ''                 ("not your turn" / "game is over")
  error if choice.cantUndo                    (only set by maybeTrapped/maybeLocked paths)
  clearChoice()
  parts = input starts with 'team ' ? [input] : input.split(',')
  error if len(parts) > 2
  for part: (type, data) = splitFirst(trim(part), ' ')
    'move'   : strip trailing tokens repeatedly:  /\s[-+]?[1-3]$/ -> targetLoc (not for 'conversion2');
               ' mega' | ' megax' | ' megay' | ' zmove' | ' ultra' | ' dynamax' | ' gigantamax' | ' max' | ' terastal' | ' terastallize' -> event
               (each at most once; otherwise "Conflicting arguments")       -> chooseMove(data, targetLoc, event)
    'switch' : chooseSwitch(data)
    'shift'  : chooseShift()          # triples only -> always error in doubles (side.ts:1103-1104)
    'team'   : chooseTeam(data)
    'pass'|'skip' : choosePass()
    'auto'|'default': autoChoose()
    else: error
  return !choice.error
```

**Notes.**
- `Battle.choose` additionally rejects an *incomplete* choice (`sim/battle.ts:3032-3035`), for example only one slot given when two are needed.
- The convenience method `Battle.makeChoices` commits without that check.
- Choices are positional: the k-th comma part goes to the next slot needing a choice, after auto-passes.

### 4.2 `getChoiceIndex` and auto-pass (`sim/side.ts:1307-1332`)

```
getChoiceIndex(isPass=false):
  index = choice.actions.length
  if !isPass:
    'move'  : while index<2 and (active[index].fainted or active[index].volatiles.commanding): choosePass(); index++
    'switch': while index<2 and !active[index].switchFlag: choosePass(); index++
  return index
```

### 4.3 `chooseMove(moveText, targetLoc=0, event='')` (`sim/side.ts:552-849`)

```
require requestState=='move'
index = getChoiceIndex(); error if index>=2 ("more choices than unfainted Pokémon")
P = active[index]; request = P.getMoveRequestData()
autoChoose = !moveText; if autoChoose: moveText = 1
if moveText is a number k: require 1<=k<=len(request.moves); moveSlot=k-1; moveid=request.moves[k-1].id; targetType=request.moves[k-1].target
else: moveid=toID(moveText) (hiddenpower* -> hiddenpower); find in request.moves -> moveSlot, targetType = entry.target || 'normal'
      (max/z-move name lookups; irrelevant in Champions); error if not found
moves = P.getMoves()                         # TRUE disabled state (hidden disables count as disabled)
if autoChoose: pick first request.moves entry not disabled (and not truly disabled)
zMove/maxMove: undefined in Champions
# --- targeting ---
if autoChoose: targetLoc = 0
elif targetTypeChoices(targetType):          # {normal, any, adjacentAlly, adjacentAllyOrSelf, adjacentFoe}  (battle-actions.ts:3, 1611-1613)
   error if targetLoc==0 ("needs a target")  # because active.length>=2
   error if !validTargetLoc(targetLoc, P, targetType)
else:
   error if targetLoc != 0 ("You can't choose a target for ...")
# --- forced moves ---
locked = P.getLockedMove() || P.getSemiLockedMove()
if locked: push {move, P, targetLoc: P.volatiles[locked]?.targetLoc ?? P.lastMoveTargetLoc ?? 0, moveid: locked}; return true
                                             # NOTE: returns before mega checks: ' mega' is silently ignored
if moves.length==0: push {move, P, moveid:'struggle'}; return true      # also ignores ' mega'; Struggle has no target
# --- disabled check ---
if no slot in moves with id==moveid and !disabled: error "...is disabled" (and request update)
# --- mega ---
if event=='mega' and !P.canMegaEvo: error
if event in (mega, megax, megay) and choice.mega: error ("only mega-evolve once per battle")
if event=='terastallize': error (canTerastallize null) ; 'dynamax' in gen 9: error ; 'zmove' w/o crystal: error
push {choice:'move', pokemon:P, targetLoc, moveid, moveSlot, mega: event=='mega', ...}
if mega: choice.mega = true
```

**Locked-move edge cases** (relevant for canonicalizing a JAX action):
- The locked request entry has no `target`. So `move 1` with **no** target is accepted, and `move 1 2` is rejected ("can't choose a target").
- Typing the move *name* gives `targetType = 'normal'` (`:594`), so a target is then required.
- Canonical form: a locked slot has exactly one action, `move 1`.

### 4.4 Targets at choice time (`validTargetLoc`, `sim/battle.ts:2415-2447`)

Signed target locations, relative to the chooser:
- **+1, +2**: foe slots a and b;
- **−1, −2**: own slots a and b (the user's own location is `−(position+1)`).

In doubles every foe is "adjacent", and the ally is adjacent; self is not adjacent.

| targetType | legal targetLoc for user in slot a (self = −1) | for user in slot b (self = −2) |
|---|---|---|
| `normal` | +1, +2, −2 | +1, +2, −1 |
| `any` | +1, +2, −2 | +1, +2, −1 |
| `adjacentFoe` | +1, +2 | +1, +2 |
| `adjacentAlly` | −2 | −1 |
| `adjacentAllyOrSelf` | −1, −2 | −1, −2 |
| everything else (`self`, `allySide`, `foeSide`, `all`, `allAdjacent`, `allAdjacentFoes`, `allies`, `allyTeam`, `randomNormal`, `scripted`) | 0 only (no target) | 0 only |

`validTargetLoc` doesn't check occupancy. Choosing a fainted or empty foe slot, or a fainted ally, is legal at choice time; resolution happens at execution (§9.4).

### 4.5 `chooseSwitch(slotText)` (`sim/side.ts:915-1017`)

```
require requestState in ('move','switch')
index = getChoiceIndex(); error if index>=2
P = active[index]
if !slotText:
   (move request) error "need to select a Pokémon"
   (switch request) if revivalblessing slot: slot = first fainted; elif forcedSwitchesLeft==0: return choosePass()
                    else slot = first index>=2 not in switchIns and not fainted
slot = int(slotText)-1  (or name/species match)
error if slot >= side.pokemon.length (=4)
error if slot < 2 (an active) and not revivalblessing        "can't switch to an active Pokémon"
error if slot in choice.switchIns                             "can only switch in once"   (two slots -> same mon)
if revivalblessing: require target fainted; forcedSwitchesLeft--; P.switchFlag=false; push {choice:'revivalblessing', P, target}; return
error if target.fainted                                       "can't switch to a fainted Pokémon"
'move' request:   error if P.trapped (truthy, incl. 'hidden')  ; maybeTrapped -> cantUndo
'switch' request: require forcedSwitchesLeft>0; forcedSwitchesLeft--
switchIns.add(slot); push {choice: request=='switch' ? 'instaswitch' : 'switch', pokemon:P, target}
```

**Trapped** (computed in `endTurn`, §8.4, plus a hard lock in `getMoveRequestData`):
- `trapped` volatile (Mean Look, Block, Jaw Lock, Spirit Shackle, Octolock, Ingrain): `data/conditions.ts:208-217`;
- `partiallytrapped`, while the source is active: `:222-252`;
- Shadow Tag (hidden): `data/abilities.ts:4157-4169`;
- hard-locked moves.

**Exceptions:**
- Ghost types are immune via `runStatusImmunity('trapped')` (`sim/pokemon.ts:1629`).
- Shed Shell (`data/items.ts:5663-5670`) and [C] Run Away (`data/mods/champions/abilities.ts:76-86`) un-trap at priority −10.
- `commanded` and `commanding` re-trap at −11 (`data/conditions.ts:824-827, 838-841`).

### 4.6 `choosePass` and forced passes (`sim/side.ts:1334-1361`, `clearChoice` at `:1118-1144`)

- **`move` request.** `pass` is legal only for a fainted or `commanding` active (`:1348-1352`).
- **`switch` request.** `clearChoice` sets:
  - `forcedSwitchesLeft = min(#flagged actives, canSwitch(side))`;
  - `forcedPassesLeft = #flagged - forcedSwitchesLeft`.

  Here `canSwitch(side)` is the number of non-fainted bench mons at index ≥ 2 that aren't already queued to switch in (`sim/battle.ts:1580-1603`). A flagged slot may `pass` only while `forcedPassesLeft > 0` (`:1341-1346`).

**Verified** (`scenarios.js fewbench`): both p2 actives fainted with one bench mon left.
- PS produced `forceSwitch [true,true]`, `forcedSwitchesLeft 1`, `forcedPassesLeft 1`.
- Accepted: `pass, switch 4` and `switch 4, pass`.
- Rejected: `pass, pass` and two switches.
- The default auto choice was `switch 4, pass`.

### 4.7 Mega Evolution rules

A mega is legal on a move action iff all of the following hold:
1. the request has `canMegaEvo`. That needs a Mega Stone matching the species ([C] `data/mods/champions/scripts.ts:182-194`), no Mega on the side yet (`runMegaEvo` clears it, `sim/battle-actions.ts:1958-1966`), and no locked move or Struggle;
2. the other slot isn't also choosing mega in the same choice (`choice.mega`, `sim/side.ts:780-782`);
3. the slot is choosing a normal move, not a switch. With `move X mega` on a locked or Struggling mon, the mega is silently dropped (§4.3).

The `megaEvo` action is added by `resolveAction` (`sim/battle-queue.ts:212-217`) and executed at order 104.

### 4.8 `autoChoose` (timeouts or `default`) (`sim/side.ts:1364-1383`)

| Request | Auto choice |
|---|---|
| preview | `chooseTeam()`, giving team slots 1-4 in submitted order |
| switch | the first available bench mon for each flagged slot, or a pass when forced |
| move | for each slot, the first selectable move with **targetLoc 0**; `resolveAction` then picks a random target (`sim/battle-queue.ts:268-272`) |

### 4.9 Exact legal-action enumeration (the JAX mask)

**`move` decision, per side.** Let `A` be the ordered pair of slots. For each slot s with active mon P:

- **P fainted or `commanding`:** only `pass`.
- **P hard-locked** (`getLockedMove()` non-null): only `move 1`. No target, no mega, no switch, because `trapped` is forced.
- **Otherwise, move actions:**
  - Let `M` = slots j where `moveSlot.disabled` is falsy (*true* state, so `'hidden'` counts as disabled) and `pp > 0`.
  - If `M` is empty: one action, Struggle (`move 1`, no target, no mega).
  - Else, for each j in M, use the request targetType (Curse and Pollen Puff adjusted) and take the targetLoc set from the §4.4 table. Each (j, loc) is one action.
  - If `canMegaEvo`, also add (j, loc, mega).
- **Switch actions** (if `!P.trapped`): one per bench index k ∈ {2, 3} with `!side.pokemon[k].fainted`. Switches are never conditioned on `canSwitch`.

**Where `disabled` comes from at request time** (reset and recomputed in `endTurn`, §8.4):
- choice lock (`data/conditions.ts:324-363`);
- Encore (`data/moves.ts:4771-4780`, inherited by [C]);
- Taunt, Disable, Torment, Throat Chop, Heal Block, Gravity, Imprison;
- `cantusetwice` (`sim/battle.ts:1712-1714`);
- [C] Fake Out / First Impression after the first move action (`data/mods/champions/moves.ts:349-356, 381-389`);
- PP 0.

**Joint constraints for a side:**
1. Two switches may not target the same bench index (`switchIns`).
2. At most one action may carry mega.

No other cross-slot constraint exists. For example, both slots may target the same foe, or each other.

**Counts.**
- Per slot: at most 4 moves × 3 targets × 2 (mega) + 2 switches = 26 actions, or exactly 1 (`pass`, or `move 1` when locked or Struggling).
- The compact 27-action layout in the scope doc (4 × 3 × 2 + 2 + pass) covers this: a self or spread move uses one canonical "target".
- `adjacentAllyOrSelf` needs 2 targets (self and ally). It's within the 3-target axis only if self is mapped onto one of them. Acupressure is the only common move with this target type.

**`switch` decision, per side.**
- Let F = flagged slots, B = available bench indices, `nS = min(|F|, |B|)`.
- Each flagged slot independently chooses `switch k` (k ∈ B) or `pass`.
- Constraints: distinct k across slots; exactly `nS` switches in total (the passes are forced).
- Unflagged slots: `pass` (implicit).
- Revival Blessing slot: choose a **fainted** party index (any index, including an active-position fainted ally).

**`teampreview`:** ordered 4-tuples of distinct indices from 6.

### 4.10 Harness: enumerate PS legality by acceptance, not by parsing JSON

For each candidate per-side choice string, take a fresh deserialized clone of the battle (`State.deserializeBattle(snapshot)`) and call `clone.sides[i].choose(str)` (the **Side** method, which does not commit). Then require `!clone.sides[i].choice.error` and `isChoiceDone()`.

Why a fresh clone each time:
- `Side.choose` returns early if `choice.cantUndo` is set (`sim/side.ts:1196`);
- `emitChoiceError` with an update mutates `pokemon.maybeDisabled/maybeLocked` and `activeRequest` (`sim/side.ts:851-904`, `527-537`).

Per side per decision this is at most 27² (~730) strings, which is cheap. The resulting set is exactly what PS accepts. Compare it with the JAX mask, canonicalizing as above:
- locked moves → `move 1`;
- targets for non-choosable moves → none;
- `mega` on a locked or Struggling mon → treated as no-mega, because PS accepts and ignores it.

---

## 5. The action queue (`sim/battle-queue.ts`)

### 5.1 Action records (`sim/battle-queue.ts:24-114`)

**Fields:**
- `choice` (the kind), `order`, `priority`, `speed`, `pokemon`, `fractionalPriority`;
- moves: `moveid`, `move` (an ActiveMove), `targetLoc` (signed, relative to the user), `originalTarget` (Pokemon at `targetLoc` when resolved, for tracking moves), `mega`, `sourceEffect`;
- switches: `target` (the incoming Pokemon), `sourceEffect` (U-turn etc.);
- team: `index`.

**Actions reference Pokemon objects, not slots.** If a mon leaves the field its action becomes inert (§6.2). Targets are relative slot numbers resolved at execution.

### 5.2 `commitChoices` (`sim/battle.ts:3056-3088`)

```
commitChoices():
  updateSpeed()                          # all non-fainted actives: pokemon.speed = getActionSpeed()
  oldQueue = queue.list ; queue.clear()  # remainder of a paused turn (empty at turn start)
  inputLog += choices
  for side: side.commitChoices()         # queue.addChoice(side.choice.actions) -> resolveAction each
  clearRequest()
  queue.sort()                           # NEW actions only
  queue.list.push(...oldQueue)           # paused remainder appended UNSORTED after the new actions
  requestState=''; turnLoop()
```

### 5.3 `resolveAction(action, midTurn=false)` (`sim/battle-queue.ts:166-277`)

```
if choice=='pass': return []                                      # :168
fill side, move (getActiveMove(moveid))
if !order: order = ORDERS[choice] (default 200 for 'move'/'event')  # :173-204
   ORDERS = {team:1, start:2, instaswitch:3, beforeTurn:4, beforeTurnMove:5, revivalblessing:6,
             runSwitch:101, switch:103, megaEvo:104, megaEvoX:104, megaEvoY:104, runDynamax:105,
             terastallize:106, priorityChargeMove:107, shift:200, residual:300}
if !midTurn:
  if move action:
     if move.beforeTurnCallback: prepend resolveAction({beforeTurnMove,...})         # Counter, Mirror Coat (data/moves.ts:3000, 11998)
     if action.mega: prepend resolveAction({megaEvo, pokemon})                       # :212-217
     (megax/megay/terastallize/runDynamax: unused in Champions)
     if move.priorityChargeCallback: prepend resolveAction({priorityChargeMove,...})  # Focus Punch, Beak Blast, Chilly Reception (data/moves.ts:6015, 1128, 2405)
     action.fractionalPriority = runEvent('FractionalPriority', pokemon, null, move, 0)   # :249  ROLLED HERE, ONCE
  if switch/instaswitch: if typeof pokemon.switchFlag == 'string': sourceEffect = move(switchFlag); pokemon.switchFlag = false  # :250-255
if action.move:
  action.move = getActiveMove(move)
  [C] curse && !user.hasType('Ghost') -> move.target='self'                          # :263-266
  if !targetLoc: target = getRandomTarget(pokemon, move); targetLoc = getLocOf(target)   # :268-272 (autoChoose, Encore[C], Instruct w/o loc)
  originalTarget = pokemon.getAtLoc(targetLoc)
getActionSpeed(action)                                             # :275 (deferPriority only in gen 7)
return actions   # e.g. [priorityChargeMove, megaEvo, move] or [beforeTurnMove, megaEvo, move]
```

**Fractional priority sources** (all evaluated at resolve time):
- Quick Claw: `randomChance(1,5)`, logs `-activate` *at commit time* (`data/items.ts:5013-5021`);
- Quick Draw: 3/10 for non-Status moves (`data/abilities.ts:3736-3743`);
- Stall: −0.1 (`data/abilities.ts:4498`).

`action.priority = movePriority + fractionalPriority` (`sim/battle.ts:2700`).

### 5.4 `getActionSpeed(action)` (`sim/battle.ts:2672-2716`)

```
if choice=='move':
   priority = dex.moves.get(move.id).priority                  # BASE priority every time (no compounding)
   target   = getTarget(pokemon, move, targetLoc)              # may call getRandomTarget -> PRNG sample
   priority = singleEvent('ModifyPriority', move, ..., priority)   # Grassy Glide (data/moves.ts:7667)
   priority = runEvent('ModifyPriority', pokemon, target, move, priority)  # Prankster +1 status, Gale Wings +1 Flying at full HP, Triage
   action.priority = priority + action.fractionalPriority
   action.move.priority = priority                              # gen>5: used by Quick Guard / Psychic Terrain checks
action.speed = action.pokemon ? action.pokemon.getActionSpeed() : 1
```

[C] `pokemon.getActionSpeed() = TR ? -getStat('spe') : getStat('spe')` (`data/mods/champions/scripts.ts:46-54`).
- `getStat('spe')` applies boosts plus `ModifySpe` handlers: Tailwind, paralysis, Choice Scarf, weather abilities, and so on (`sim/pokemon.ts:617-660`).
- It is capped at 10000 unless the format sets `battle.trunc` (Reg M-C doesn't) (`:658`).
- Non-move actions keep `priority = 0` (team uses `-index`) and use the mover's speed: the outgoing mon for switches, the mega-evolver for megaEvo.

### 5.5 Sorting

**`comparePriority(a, b)`** (`sim/battle.ts:409-416`). The key is, in order:
1. `order` ascending (missing → 2^32);
2. `priority` descending;
3. `speed` descending;
4. `subOrder` ascending;
5. `effectOrder` ascending.

**`speedSort(list)`** (`sim/battle.ts:434-465`):
- a stable selection of the maximum group at each step;
- groups that tie on the whole key are `prng.shuffle`d (k−1 draws).

`queue.sort()` calls `speedSort(queue.list)` (`sim/battle-queue.ts:424-428`).

### 5.6 Queue mutation primitives and who uses them in Reg M-C

| Primitive | Semantics | Users (legal in Reg M-C unless noted) |
|---|---|---|
| `addChoice(c)` `:307-313` | `resolveAction` then **push to end** (no sort) | `start`, `residual` (`sim/battle.ts:3003`), side commits, Revival Blessing re-entry (`sim/battle.ts:2824`) |
| `insertChoice(c, midTurn)` `:375-409` | Calls `c.pokemon.updateSpeed()`, resolves, then inserts at the position `comparePriority` dictates **without re-sorting**. Among equal-key entries it picks a **random** index (`:406`). | `beforeTurn` (`sim/battle.ts:3002`), `runSwitch` (`sim/battle-actions.ts:211`), `changeAction` |
| `prioritizeAction(a, effect)` `:282-292` | Remove a, set `a.order = 3`, `sourceEffect = effect`, **unshift** | After You (`data/moves.ts:208`), Instruct on a **new** action (`data/moves.ts:9665-9670`), Round (`:15514`). Not in Reg M-C: Pledges, Shell Trap. |
| `changeAction(p, c)` `:301-305` | `cancelAction(p)`, then `insertChoice(c)` (full resolve, so fractional priority is re-rolled and a random target picked if none given) | [C] Encore (`data/mods/champions/moves.ts:326-336`) |
| `cancelAction(p)` `:334-343` | Remove **all** of p's actions except `runSwitch` | `switchIn` of the outgoing mon (`sim/battle-actions.ts:124`). Not in Reg M-C: Commander (`data/abilities.ts:629`, `data/conditions.ts:845-847`). |
| `cancelMove(p)` `:345-353` | Remove p's first `move` | Pursuit (not in Reg M-C) |
| `willMove(p)` `:324-332` | First queued `move` of p, or null if p is fainted | Sucker Punch, Upper Hand, After You, Quash, Encore, Analytic, Me First, Truant, ... |
| `willAct()` `:315-322` | Is any move/switch/instaswitch/shift left? | Protect family: `return !!queue.willAct() && runEvent('StallMove')` (`data/moves.ts:13970` etc.). **Protect fails when the user is the last to act** (verified twice). Wide Guard and Quick Guard (`:20815`, `:14497`). |
| `willSwitch(p)` `:355-362` | | Natural Cure info (base only) |
| Quash | Sets `action.order = 201` in place (`data/moves.ts:14460-14467`) | The next re-sort moves it after every order-200 move |

**Dancer** [not in Reg M-C] doesn't use the queue. It recursively calls `runMove` inside the dancing move's `runMove`, slowest Dancer first (`sim/battle-actions.ts:376-402`). External moves don't trigger Dancer again.

### 5.7 Dynamic speed (Gen 8+): exact semantics

At the end of `runAction` (`sim/battle.ts:2975-2984`):

```
nextAction = queue.peek()
if gen >= 8 and nextAction and nextAction.choice in ('move','runDynamax'):
    updateSpeed()                                   # every non-fainted active: pokemon.speed = getActionSpeed()
    for a in queue.list: if a.pokemon: getActionSpeed(a)   # recompute priority (ModifyPriority, base+frac) AND speed
    queue.sort()                                    # full re-sort, ties re-shuffled
```

**When this tail is skipped.** It isn't reached if the action:
- ended the battle;
- paused for a `switch` request (`:2964-2968`);
- returned early: `team`, `pass`, and `move` / `beforeTurnMove` / `priorityChargeMove` / `shift` whose pokemon is inactive or fainted (`:2759-2760, 2784-2785, 2793-2794, 2843-2844`).

Those early returns change no state, so the order from the last re-sort is still current, up to PRNG tie draws.

**Effective rule for a re-implementation.** Immediately before **every** move action executes, PS recomputes every remaining move's priority and speed from current state and takes the maximum by (order, priority, speed, uniform random among ties). This holds because:
- at the moment of a re-sort, all remaining queue entries are moves (order 3, 200 or 201) or `residual` (300);
- anything with order < 200 (instaswitch, runSwitch) would be at the head, and then no re-sort happens.

Entries with order < 200 (beforeTurnMove, switch, megaEvo, priorityChargeMove) run in the **commit-time** order, using commit-time speeds.

**Speed is refreshed** (`updateSpeed` call sites):
- `commitChoices` (`sim/battle.ts:3057`);
- the dynamic re-sort (`:2979`);
- the start of the `residual` action (`:2854`);
- endTurn dynamax (`:1651`, unused);
- every `insertChoice` for that pokemon (`sim/battle-queue.ts:385`).

`pokemon.speed` is the cached value used by event-handler sorting (`resolvePriority`, `sim/battle.ts:1008-1010`) and by `runSwitch`'s `speedOrder`.

**Verified:**
- **Mega:** Gengar at 130 Spe was slower than Garchomp at 142. After `megaEvo`, Mega Gengar at 150 moved **before** Garchomp in the same turn (`scenarios.js megaspeed`).
- **Icy Wind:** Kingambit at 102 dropped to 68, and Milotic at 101, chosen as slower, then moved before Kingambit (`scenarios.js icywind`).

Also implied by the code:
- a mid-turn Tailwind or Trick Room re-orders the remaining moves;
- Gale Wings priority is lost once Talonflame takes damage before moving;
- Grassy Glide gains +1 priority if Grassy Terrain appears mid-turn.

---

## 6. `turnLoop` and `runAction`

### 6.1 `turnLoop()` (`sim/battle.ts:2996-3016`)

```
turnLoop():
  log '|' and |t:|
  requestState = ''
  if !midTurn:                                   # fresh turn (not a resume)
     queue.insertChoice({choice:'beforeTurn'})   # order 4 -> ahead of everything chosen
     queue.addChoice({choice:'residual'})        # order 300, pushed at end
     midTurn = true
  while (action = queue.shift()):
     runAction(action)
     if requestState or ended: return            # PAUSE (switch request) or game over
  endTurn(); midTurn = false; queue.clear()
```

**A typical turn queue after commit:**

```
[beforeTurn(4), beforeTurnMove(5)*, revivalblessing(6)*, switch(103)*, megaEvo(104)≤2, priorityChargeMove(107)*, move(200)*, residual(300)]
```

### 6.2 `runAction(action)`, per type (`sim/battle.ts:2718-2859`)

| choice | Body | Applies in Reg M-C |
|---|---|---|
| `start` | §2.6 | yes |
| `team` | §2.5. Returns before the tail. | yes |
| `beforeTurn` | `eachEvent('BeforeTurn')`: actives in speed order (`sim/battle.ts:470-481`) | yes (Commander's cancel is not in Reg M-C) |
| `beforeTurnMove` | Skip if inactive or fainted. `getTarget`, then `move.beforeTurnCallback(pokemon, target)` (`:2783-2791`). | Counter, Mirror Coat |
| `revivalblessing` | `pokemonLeft++`. If the revived mon is in an active slot (a fainted ally), `addChoice({instaswitch, pokemon: target, target})`. Clear `fainted`, set HP to half, log `-heal`, remove the slot condition (`:2821-2838`). | Pawmot |
| `switch`, `instaswitch` | For `switch`, a status-holder triggers Natural Cure `CheckShow` ([C] removed). Then `actions.switchIn(target, pokemon.position, sourceEffect)` (`:2814-2820`). | yes |
| `runSwitch` | `actions.runSwitch(pokemon)`, batched (§2.7) | yes |
| `megaEvo` | `actions.runMegaEvo(pokemon)`, which calls `formeChange(mega, item, permanent)`, logs `detailschange` and `-mega`, clears side `canMegaEvo`, and runs `AfterMega` (`sim/battle-actions.ts:1952-1970`). There is **no** fainted or active check. | yes (1 per side) |
| `priorityChargeMove` | Skip if inactive or fainted. `move.priorityChargeCallback(pokemon)` (`:2792-2798`). | Focus Punch, Beak Blast, Chilly Reception |
| `move` | Skip (`return false`, **no tail**) if `!pokemon.isActive` or `pokemon.fainted`. Otherwise `actions.runMove(move, pokemon, targetLoc, {sourceEffect, originalTarget})` (`:2758-2765`). runMove increments `activeMoveActions`, runs the OverrideAction and BeforeMove events, deducts PP, calls `useMove`, and finally `resolveFaints(); checkWin()` (`sim/battle-actions.ts:267-418`). | yes |
| `shift` | triples only | no |
| `pass` | `return` (no tail) | auto-passes resolve to no action (`resolveAction` returns `[]`) |
| `residual` | §8.1 | yes |
| `runDynamax`, `terastallize`, `event`, `megaEvoX/Y` | n/a | no |

### 6.3 The post-action pipeline: the `runAction` tail (`sim/battle.ts:2861-2986`, gen-9 branches only)

```
# (a) phazing
for side, for p in side.active: if p.forceSwitchFlag: (if p.hp: actions.dragIn(side, p.position)); p.forceSwitchFlag = false
        # dragIn: getRandomSwitchable (PRNG sample over non-fainted bench) -> switchIn(isDrag) -> runSwitch inline  (sim/battle-actions.ts:216-227)
clearActiveMove()
# (b) faints + win
resolveFaints({checkWinIfNoFaints:true, checkWin:true}); if ended: return
# (c) end-of-queue faint flags
nextAction = queue.peek()
if !nextAction: checkFainted()                      # only when the queue is EMPTY (after residual, or after an end-of-turn replacement batch)
# (d) Update + Emergency Exit
if action.choice != 'start' and nextAction?.choice != 'instaswitch':
    eachEvent('Update')                            # speed-sorted actives; item/ability 'Update' handlers (berries, White Herb, ...)
    for (p, hpBefore) in residualPokemon: runEvent('EmergencyExit', p, ..., hpBefore)   # only non-empty after 'residual'
if action.choice == 'runSwitch': runEvent('EmergencyExit', action.pokemon, ..., hpBeforeAction)  # FIRST pokemon of the batch only
# (e) switch requests
switches[i] = any(p.switchFlag for p in side_i.active)
for side i:
   if switches[i] and !canSwitch(side_i): clear all switchFlags on that side except a revivalblessing slot; switches[i] = (revival pending)
   elif switches[i]: for p with hp and switchFlag and switchFlag != 'revivalblessing': runPursuitActivation(p)   # BeforeSwitchOut + Update
if any(switches): makeRequest('switch'); return true      # PAUSE
# (f) dynamic speed (§5.7)
```

### 6.4 Mid-turn switch decisions

**Sources of `switchFlag`** (each sets it on a mon that is still active with HP > 0):

| Source | Where | Reg M-C |
|---|---|---|
| `selfSwitch` moves: U-turn, Volt Switch, Flip Turn, Parting Shot, Baton Pass (`copyvolatile`), Shed Tail (`shedtail`), Chilly Reception | `source.switchFlag = move.id` when the move "did something", the user has HP, isn't `commanded`, and `canSwitch(side)` (`sim/battle-actions.ts:1366-1369`) | yes |
| Revival Blessing | Same line. `switchFlag = 'revivalblessing'`, allowed even with no bench. | yes (Pawmot) |
| Eject Button | [C] `data/mods/champions/items.ts:262-277`: holder hit by a damaging non-future move from another mon, can switch, not already forced out; at most one per action | yes |
| Emergency Exit | [C] `data/mods/champions/abilities.ts:26-34`: HP crosses ≤ 1/2. Called from the damage pipeline (`data/mods/champions/scripts.ts:583`, `sim/battle-actions.ts:1073`), recoil (`:1450`), the post-residual tail (`sim/battle.ts:2927-2929`) and post-runSwitch hazard damage (`:2933`). | yes (Golisopod) |
| Eject Pack, Wimp Out | | no |
| Faint replacement | `checkFainted()` sets `switchFlag = true` and status `fnt` on fainted actives, if `canSwitch(side)`, but **only when the queue is empty** (`sim/battle.ts:2540-2550, 2909-2910`) | yes (end of turn) |

**The pause.** `makeRequest('switch')`:
- each side with ≥1 flagged active gets `{forceSwitch: [flagA, flagB], side}`, the other gets `{wait: true}`;
- `requestState = 'switch'`, and `turnLoop` returns (`:3010`).

Several flags raised during one action, on one or both sides, produce **one** request. Examples are U-turn into an Eject Button holder, or a spread move triggering Emergency Exit and Eject Button.

**The resume.**
1. The chosen `switch k` becomes an **`instaswitch`** action (`sim/side.ts:1011`). `resolveAction` turns a string `switchFlag` into `sourceEffect` (logged `|switch|...|[from] U-turn`) and clears the flag (`sim/battle-queue.ts:250-255`).
2. `commitChoices` puts the sorted new instaswitches in front of the unsorted old remainder (§5.2).
3. Each `instaswitch` runs `switchIn` for the outgoing live mon (`sim/battle-actions.ts:90-136`):
   - `beingCalledBack`, then `runPursuitActivation` (BeforeSwitchOut) and `runEvent('SwitchOut')` ([C] Regenerator and Natural Cure apply here);
   - ability and item `End`;
   - **`queue.cancelAction(oldActive)`**, so its pending move is dropped;
   - `copyVolatileFrom` for Baton Pass or Shed Tail, then `clearVolatile()`;
   - swap `side.pokemon` indices and positions (`:137-154`);
   - mark the new mon active and insert its `runSwitch`.
4. `runSwitch` batches and fires SwitchIn effects. The tail then re-sorts the remaining moves with fresh speeds before the next move.

**Verified trace** (`scenarios.js uturn`):
- Turn 1: p1 Talonflame U-turn and p2 Incineroar Parting Shot.
- After U-turn: `requestState=switch`, p1 `forceSwitch=[true,false]`, p2 `wait`. Queue = `[move Milotic scald, move Incineroar partingshot, residual]`.
- p1 picks `switch 3`: `|switch|p1a: Garchomp|...|[from] U-turn`, then Scald, then Parting Shot.
- That raises a second `switch` request, for p2 only. After p2's switch: residual, then `|turn|2`.
- Serializing at the first pause and deserializing produced a **byte-identical** continuation log.

**What does not create a decision:**
- **Drags.** Roar, Whirlwind, Dragon Tail and Circle Throw set the target's `forceSwitchFlag` (`sim/battle-actions.ts:1409-1425`). Red Card does the same for the attacker (`data/items.ts:5170-5195`). The tail's phazing loop runs `dragIn` immediately: a random bench mon, `|drag|`, inline `runSwitch`.
- **Faint replacements.** These are deferred to the end of the queue.

### 6.5 Revival Blessing flow (Pawmot; legal)

1. The move sets slot condition `revivalblessing` and `switchFlag = 'revivalblessing'`.
2. The tail keeps that flag even if `canSwitch` is 0 (`:2943-2950`). No Pursuit activation.
3. The request shows `forceSwitch` for the slot, with `reviving: true` in the side data (`sim/pokemon.ts:1204`).
4. `chooseSwitch` requires a **fainted** party member (`sim/side.ts:965-978`). This produces an action `revivalblessing` (order 6).
5. Running it revives the mon at half HP. If the target is the user's fainted **ally still occupying a slot**, an `instaswitch` puts it back in the same slot immediately (`sim/battle.ts:2821-2838`).

---

## 7. Fainting, win and tie

### 7.1 `pokemon.faint()` (`sim/pokemon.ts:1602-1614`)

Only queues the faint:
- sets `hp = 0`, `switchFlag = false` (cancelling any pending pivot), `faintQueued = true`;
- pushes `{target, source, effect}` onto `battle.faintQueue`.

`damage()` calls it when HP reaches ≤ 0 (`:1616-1626`).

### 7.2 `resolveFaints()` (`sim/battle.ts:2561-2657`, gen-9 path)

```
while faintQueue:
  fd = faintQueue.shift(); p = fd.target
  if !p.fainted and runEvent('BeforeFaint', p, fd.source, fd.effect):
     log |faint|p ; side.pokemonLeft-- ; side.totalFainted++ (cap 100)
     runEvent('Faint', p, ...)                    # gen>=5 (Aftermath, Destiny Bond, Grim Neigh...)
     singleEvent('End', ability) ; singleEvent('End', item)
     p.clearVolatile(false)  ([C] version) ; p.fainted = true ; p.illusion = null ; p.isActive = false ; p.isStarted = false
     (formeRegression: never set by Mega in [C] -> Megas stay Mega)
     side.faintedThisTurn = p
     if this faint queued another faint: checkWin = true
if checkWin and checkWin(fd): return true
if faints happened: runEvent('AfterFaint', lastTarget, ..., count)
```

**Call sites:**
- every `runAction` tail (§6.3 b);
- after every residual handler (`sim/battle.ts:572`);
- after `runMove` (`sim/battle-actions.ts:411`);
- between Dancers (not in Reg M-C);
- in `runPursuitActivation` when relevant.

### 7.3 When fainted mons leave the field

A fainted mon **stays referenced in `side.active[pos]`**, with `fainted = true`, `isActive = false`, `hp = 0`, until a `switchIn` into that position replaces it. If there is no replacement, it stays there for the rest of the battle. Effects:
- `getAllActive()` excludes it (`sim/battle.ts:1379-1389`);
- `side.allies()` and `foes()` filter by `hp` (`sim/side.ts:390-403`);
- `isAdjacent` returns false for it (`sim/pokemon.ts:763`);
- its queued actions stay in the queue and are skipped when reached (§6.2);
- `switchIn` resets the old fainted mon's status to `''` when it is swapped to the bench (`sim/battle-actions.ts:146`).

There is **no** mid-turn replacement in Gen 5+, so the empty slot persists until the end of the turn.

### 7.4 `checkWin` and ties (`sim/battle.ts:2659-2670`, `win` at `:1534-1557`)

```
checkWin(fd):
  if all sides pokemonLeft == 0: win(fd && gen>4 ? fd.target.side : null)   # side of the LAST processed faint wins; null -> |tie|
  for side: if side.foePokemonLeft() == 0: win(side)
```

`win()`:
- logs `|` then `|win|NAME` (or `|tie|`);
- sets `ended = true`, `requestState = ''`, and every side's `activeRequest = null`.

`tiebreak()` (`:1481-1522`) only runs on the server `>tiebreak` command (timer), never inside the engine.

### 7.5 End-of-turn replacement and "fewer mons than empty slots"

1. After `residual` the queue is empty, so the tail calls `checkFainted()`. Fainted actives on sides with `canSwitch > 0` get `switchFlag = true`. Sides with no bench keep their dead slots, with no flag and no request.
2. Emergency Exit from residual damage can add flags in the same tail (§6.3 d).
3. `makeRequest('switch')` issues one request covering all flagged slots on both sides. `clearChoice` computes forced switches and passes (§4.6, verified).
4. On commit, the `instaswitch`es are sorted by the outgoing (fainted) mon's speed. All switches happen first, then **one** batched `runSwitch` / `fieldEvent('SwitchIn')` (verified: `|switch|p2b ...`, then `|switch|p1b ...`, then `-ability ... Pressure`).
5. The tail after the batch sees an empty queue and calls `checkFainted()` again. If hazards killed a replacement, a new `switch` request is issued. This repeats until stable or the battle ends.
6. Then `endTurn()`.

### 7.6 Fainting during residuals

`fieldEvent('Residual')` calls `resolveFaints()` after **each** handler (`sim/battle.ts:572`):
- `|faint|` lines appear interleaved with residual damage;
- `checkWin` runs each time, and if the battle ends, residual processing stops immediately (`:525, :573`);
- later handlers whose holder has fainted are skipped, except slot conditions such as Wish and Future Sight (`:517-519`).

---

## 8. End of turn

### 8.1 The `residual` action (`sim/battle.ts:2851-2858`) and `fieldEvent('Residual')`

```
log '|' ; clearActiveMove(failed=true) ; updateSpeed()
residualPokemon = [(p, p.hp) for p in getAllActive()]
fieldEvent('Residual')
if !ended: log |upkeep|
```

**Handler collection** (`sim/battle.ts:489-511`, with `getKey = 'duration'`):

| Holder | Handlers collected | Sort-key field |
|---|---|---|
| Field | pseudoweathers, weather and terrain with an `onFieldResidual` callback **or a `duration`**; end function `removePseudoWeather` / `clearWeather` / `clearTerrain` | `onFieldResidualOrder/SubOrder` |
| Each side | side conditions with `onSideResidual` or a `duration`; end `removeSideCondition` | `onSideResidual*` |
| Each active (non-null slot) | status, volatiles, ability, item, species, slot conditions with `onResidual` or a duration (`sim/battle.ts:1111-1170`); plus side, field and format `onResidual` callbacks held per-pokemon (customHolder; no end function, so no double countdown) | `onResidual*` |

**Sort key:**
- **order:** `on*ResidualOrder`, or last if absent;
- **priority:** `on*ResidualPriority`;
- **speed:** the holder's cached `pokemon.speed`, freshly updated. Side and field handlers have speed 0;
- **subOrder:** explicit, or the effect-type default (§2.7);
- **effectOrder** is not set for Residual. Equal keys are shuffled by PRNG, for example both sides' Tailwind.

**Per handler:**
- skip if the holder has fainted (unless it's a slot condition);
- **duration countdown**: `duration--`, and at 0 call the end function and *skip the callback* (`:520-528`);
- skip stale handlers;
- run the callback;
- `resolveFaints()`.

### 8.2 Residual order in Champions (`research/ps-loop/residual_handlers.js`, Past excluded)

| Order (sub) | Effect |
|---|---|
| 1 | weather `onFieldResidual`: `-weather X|[upkeep]`, then `eachEvent('Weather')` for damage. Weather duration (5) also counts down here. |
| 3 | `futuremove` slot condition (Future Sight) |
| 4 | Wish |
| 5 (2) | Grassy Terrain heal (per active) |
| 5 (3) | Healer, Hydration, Shed Skin |
| 5 (4) | Leftovers |
| 6 / 7 / 8 | Aqua Ring / Ingrain / Leech Seed |
| 9 | `psn`, `tox` |
| 10 | `brn` |
| 12 / 13 | Curse / Salt Cure, `partiallytrapped` |
| 14 | Octolock, Syrup Bomb |
| 15 / 16 / 17 / 18 | Taunt / Encore / Disable / Magnet Rise (duration end) |
| 22 / 23 / 24 / 25 | Throat Chop / Yawn / Perish Song / Roost |
| 26 (1, 2, 3, 5, 10) | Reflect, Light Screen, Safeguard, Tailwind, Aurora Veil (side-condition countdowns) |
| 27 (1, 2, 5, 6, 7) | Trick Room, Gravity, Wonder Room, Magic Room, terrains (field countdowns) |
| 28 (1) | Uproar |
| 28 (2) | Speed Boost, Moody, Harvest, Pickup, Cud Chew, Slow Start, Bad Dreams |
| 29 | Opportunist, White Herb, Zen Mode, Schooling, Shields Down, Power Construct, Hunger Switch |
| none (last) | 1-turn volatiles and conditions: Protect family, Helping Hand, Follow Me, Rage Powder, Wide Guard, Quick Guard, flinch, `stall`, `lockedmove`, `twoturnmove`, `mustrecharge`, Ally Switch (2) |

Within an order (and subOrder), mons act fastest first. Trick Room flips this, because the speed key is negated in [C].

### 8.3 After residual

The `runAction` tail (§6.3) does the following:
- `resolveFaints` (already done per handler);
- empty queue, so `checkFainted()`;
- `eachEvent('Update')`;
- `EmergencyExit` for each `(p, hpBefore)`;
- a possible `switch` request (§7.5).

When the queue finally drains, `turnLoop` calls `endTurn()`.

### 8.4 `endTurn()` (`sim/battle.ts:1640-1818`)

```
turn++ ; lastSuccessfulMoveThisTurn = null
(dynamax ending; gen-1 stuff: n/a)
for side, for p in side.active (including fainted-in-place):
   p.moveThisTurn=''; p.newlySwitched=false; p.moveLastTurnResult=p.moveThisTurnResult; p.moveThisTurnResult=undefined
   if turn != 1: usedItemThisTurn=statsRaisedThisTurn=statsLoweredThisTurn=false; hurtThisTurn=null
   maybeDisabled=maybeLocked=false; every moveSlot.disabled=false, disabledSource=''
   runEvent('DisableMove', p)                       # choicelock, encore, taunt, torment, disable, imprison(foe), heal block, gravity, throat chop...
   for moveSlot: singleEvent('DisableMove', move)   # [C] Fake Out / First Impression; Stuff Cheeks/Belch removed in [C]
                 if move.flags.cantusetwice and p.lastMove.id == slot.id: disableMove
   attackedBy bookkeeping; gen>=7 typechange reveal
   p.trapped = p.maybeTrapped = false
   runEvent('TrapPokemon', p); if !knownType or trap-immune: runEvent('MaybeTrapPokemon', p)
   for foe: for each of foe species' OTHER legal abilities: singleEvent('FoeMaybeTrapPokemon', ability, {}, p, foe)   # info hiding
   if p.fainted: continue
   p.activeTurns++                                   # non-fainted only
side.faintedLastTurn = side.faintedThisTurn ; side.faintedThisTurn = null
if maybeTriggerEndlessBattleClause(...): return     # §8.5
log |turn|N ; makeRequest('move')
```

**Turn counters:**
- `battle.turn` is 0 during team preview and the lead switch-ins, and 1 at the first move request.
- `pokemon.activeTurns` is reset to 0 on `switchIn` (`sim/battle-actions.ts:157`) and incremented at each `endTurn` while the mon is active and alive. A lead therefore has `activeTurns = 0` during turn 1.
- `pokemon.activeMoveActions` is reset on `switchIn` and incremented at every `runMove` (`sim/battle-actions.ts:274`). That includes moves that fail in BeforeMove (flinch, sleep, paralysis).
- [C] Fake Out availability depends on `activeMoveActions`, not `activeTurns`.

### 8.5 Turn limit and Endless Battle Clause (`sim/battle.ts:1820-1920`)

- `turn <= 100`: return.
- `turn > 1000`: log `It is turn 1000...`, then `tie()`. The battle ends before `|turn|1001`, so there are **at most 1000 move decisions**.
- Turns 500 and up: `bigerror` warnings only.
- The EBC logic runs only if the rule table has `endlessbattleclause` (`:1871`). **Reg M-C doesn't**, so this is a no-op.

---

## 9. Doubles positional logic

### 9.1 Arrays and invariants

**Invariants:**
- `side.pokemon` has length 4 after preview.
- `side.active` has length 2.
- For every mon, `pokemon.position` equals its index in `side.pokemon`.
- `side.active[i] === side.pokemon[i]` for i ∈ {0, 1}.

These hold because `switchIn` swaps both the array entries and the positions of the incoming and outgoing mons (`sim/battle-actions.ts:137-156`), and so does `swapPosition` (`sim/battle.ts:1605-1624`).

**Slot and position identifiers:**
- Slot ids: `p1a`, `p1b`, `p2a`, `p2b` (`getSlot`, `sim/pokemon.ts:541-545`; `getAtSlot`, `sim/battle.ts:1628-1634`).
- `getFieldPositionValue() = side.n + 2*position`, giving p1a = 0, p2a = 1, p1b = 2, p2b = 3 (`sim/pokemon.ts:547-550`). This is used for `speedOrder`.
- Slot conditions (Wish, Future Sight, Healing Wish, Revival Blessing) are stored per **position** (`side.slotConditions[pos]`, `sim/side.ts:464-511`). They follow the slot, not the mon.

### 9.2 Signed target numbering

- `getLocOf(target)` = `−(target.position+1)` if same side, else `+(target.position+1)` (`sim/pokemon.ts:805-810`).
- `getAtLoc(loc)` = own `active[|loc|−1]` if `loc < 0`, else foe `active[loc−1]` (`sim/pokemon.ts:791-799`).

**From p1a's point of view:**

| loc | Refers to |
|---|---|
| +1 | p2a |
| +2 | p2b |
| −1 | p1a (self) |
| −2 | p1b (ally) |

**From p1b:** +1 = p2a, +2 = p2b, −1 = p1a (ally), −2 = self.

**In the protocol choice string**, `move N +1` and `move N 1` are equivalent. The regex accepts a sign and the digits 1-3 (`sim/side.ts:1229-1232`). `getChoice()` re-serializes positive targets as `+1` (`:331-333`).

### 9.3 Adjacency (`sim/pokemon.ts:737-767`)

- In doubles, `isAdjacent(a, b)` = both not fainted and `a ≠ b` (`activePerHalf <= 2`).
- `adjacentFoes()` = `side.foes()` (living foes).
- `adjacentAllies()` = living allies other than self.
- `validTargetLoc` is purely positional (§4.4).

### 9.4 Target resolution at execution

**`getTarget(pokemon, move, targetLoc, originalTarget)`** (`sim/battle.ts:2453-2504`):
1. If `tracksTarget`, or the user has Stalwart / Propeller Tail, and `originalTarget` is still active: return it.
2. Smart-target moves (Dragon Darts): the mon at loc if alive, else random.
3. Targeting own position with `adjacentAlly`, `any` or `normal` returns null (fail), except future moves.
4. If `targetLoc` is valid for the target type (and the move is not `randomNormal`):
   - fainted ally target: return the fainted ally (no retarget). `adjacentAllyOrSelf` returns self instead;
   - living target: return it;
   - fainted foe: fall through.
5. Otherwise return `getRandomTarget`.

**`getRandomTarget`** (`:2506-2538`):
- `self`, `all`, `allySide`, `allyTeam`, `adjacentAllyOrSelf` → the user;
- `adjacentAlly` → a random living adjacent ally, or null;
- otherwise, in doubles, `side.randomFoe()` (PRNG `sample` over living foes, `sim/side.ts:369-373`) or else `foe.active[0]`.

**`getMoveTargets`** (`sim/pokemon.ts:812-877`) computes the final target list:
- single target: retarget if the foe has fainted, then the `RedirectTarget` event (Follow Me, Rage Powder, Lightning Rod) unless `tracksTarget`;
- `allAdjacentFoes` → living foes; `allAdjacent` → ally plus foes; `all`, `foeSide`, `allySide` handled as their names say;
- the `|move|` line is retargeted for the log.

**PRNG note.** `getActionSpeed` calls `getTarget` on every re-sort (§5.4). A queued `randomNormal` move (Outrage, Struggle) or a move aimed at a fainted foe therefore consumes a `sample` draw on **each** re-sort, even though the result is discarded. Forced-outcome harnesses must classify these as "no-effect" draws.

### 9.5 Ally Switch (`data/moves.ts:302-360`, `swapPosition` at `sim/battle.ts:1605-1624`)

- **Priority +2.** `onPrepareHit` adds the `allyswitch` volatile, which fails with chance 1 − 1/counter on consecutive uses (counter ×3, max 729). See `:311-313`, `:332-350`.
- **onHit.** Fails if the partner slot is empty or fainted (`:322-329`). Otherwise `swapPosition(user, otherPos, '[from] move: Ally Switch')`:
  - swaps `side.pokemon[0]` and `[1]`, `side.active[0]` and `[1]`, and both `position` fields;
  - logs `|swap|POKEMON|POSITION|[from] move: Ally Switch`;
  - runs `runEvent('Swap')` for both mons.
- **Effect on the queue.** Queued actions keep their **Pokemon** references, so both mons still act.
- **Effect on targets.** Every queued `targetLoc` is a slot number. Attacks aimed at the old slot now hit the mon that moved into it (the classic redirect). Tracking moves and Stalwart users follow `originalTarget` instead.
- **Effect on slot conditions.** They stay with the slot.

### 9.6 Commander [not in Reg M-C]

`data/abilities.ts:604-644`, `data/conditions.ts:813-848`.

- Activates in `onUpdate` or `onStart` when Tatsugiri and Dondozo are both active, not during a pending `runSwitch` and with no `switchFlag`.
- On activation: `queue.cancelAction(tatsugiri)`, add `commanding` to Tatsugiri, and `commanded` (+2 to all stats) to Dondozo.

**Turn-level effects of `commanding`:**
- auto-pass in `move` requests (`sim/side.ts:1316, 1349`);
- actions cancelled at each `BeforeTurn`;
- every move aimed at it misses, via `hitStepInvulnerabilityEvent` (`sim/battle-actions.ts:687-688`). It can still be *chosen* as a random target (`randomFoe` filters only by HP), which then results in a miss;
- trapped, and can't be dragged.

**Turn-level effects of `commanded`:** trapped, can't be dragged, `selfSwitch` is ignored (`sim/battle-actions.ts:1366`).

---

## 10. Output protocol and serialization

### 10.1 Log channels

- `battle.log` is an array of lines.
- Private information is emitted through `addSplit`: a line `|split|pN`, then the **secret** line (exact HP, for player N and the omniscient channel), then the **shared** line (HP out of 100) (`sim/battle.ts:3141-3173`).
- `extractChannelMessages(log, [-1])` gives the omniscient stream (`sim/battle.ts:35-59`). **Use channel −1** (or take the first line of each split pair) to get exact HP.
- `|t:|` timestamps are nondeterministic; strip them (`State.normalizeLog`, `sim/state.ts:160-170`).

### 10.2 Lines a state-diff harness should parse

The format reference is `sim/SIM-PROTOCOL.md:42-621`.

| Line | State meaning |
|---|---|
| `\|teampreview\|4`, `\|poke\|pN\|DETAILS\|` | preview phase |
| `\|teamsize\|pN\|4`, `\|start` | battle start; the 4 brought |
| `\|switch\|pNx: NAME\|DETAILS\|HP STATUS[\|[from] MOVE]` | voluntary switch, U-turn and similar, and replacements. Sets slot occupant, HP and status. |
| `\|drag\|...` | phaze or Red Card (random switch-in) |
| `\|replace\|`, `\|detailschange\|POKE\|DETAILS`, `\|-formechange\|` | species changes; Mega emits `detailschange` |
| `\|-mega\|POKE\|BASESPECIES\|STONE` | Mega Evolution ([C] adds the base species) |
| `\|swap\|POKE\|POS\|[from] move: Ally Switch` | position swap |
| `\|move\|SRC\|MOVE\|TGT[\|[spread] p2a,p2b][\|[miss]][\|[still]][\|[notarget]][\|[from]...]` | a move execution; gives action order. `[spread]` lists the hit slots. |
| `\|cant\|POKE\|REASON[\|MOVE]` | action skipped: flinch, par, slp, recharge, nopp, ... |
| `\|-damage\|POKE\|HP STATUS[\|[from] EFFECT][\|[of] SRC]` | HP after the damage, not the amount |
| `\|-heal\|POKE\|HP STATUS[\|[from]...]`, `\|-sethp\|` | HP after |
| `\|faint\|POKE` | faint resolved (`pokemonLeft--`) |
| `\|-status\|POKE\|STATUS`, `\|-curestatus\|POKE\|STATUS`, `\|-cureteam\|` | status |
| `\|-boost\|POKE\|STAT\|N`, `\|-unboost\|...`, `\|-setboost\|`, `\|-clearboost\|`, `\|-clearallboost\|`, `\|-clearnegativeboost\|`, `\|-invertboost\|`, `\|-swapboost\|`, `\|-copyboost\|` | boosts |
| `\|-weather\|WEATHER[\|[from] ability: X\|[of] POKE]`, `\|-weather\|WEATHER\|[upkeep]`, `\|-weather\|none` | weather start, continuation, end |
| `\|-fieldstart\|move: Trick Room\|[of]...`, `\|-fieldend\|...`, `\|-fieldactivate\|` | terrain and pseudo-weather |
| `\|-sidestart\|pN: NAME\|move: Tailwind`, `\|-sideend\|...`, `\|-swapsideconditions` | side conditions |
| `\|-start\|POKE\|EFFECT`, `\|-end\|POKE\|EFFECT`, `\|-singleturn\|POKE\|MOVE`, `\|-singlemove\|` | volatiles (Protect, Encore, Taunt, confusion, ...) |
| `\|-item\|`, `\|-enditem\|POKE\|ITEM[\|[from]...]` | item gained or consumed (Eject Button, Sash, berries) |
| `\|-ability\|POKE\|ABILITY[\|boost]`, `\|-endability\|` | ability activation (Intimidate, ...) |
| `\|-activate\|POKE\|EFFECT` | Quick Claw roll success, Protect block, Eject Button, Emergency Exit, Instruct, After You, Quash |
| `\|-miss\|`, `\|-fail\|`, `\|-immune\|`, `\|-crit\|`, `\|-supereffective\|`, `\|-resisted\|`, `\|-hitcount\|`, `\|-block\|`, `\|-notarget\|` | per-target outcomes |
| `\|-transform\|`, `\|-prepare\|`, `\|-mustrecharge\|`, `\|-waiting\|`, `\|-combine\|` | misc move-state lines |
| `\|` (blank), `\|upkeep`, `\|turn\|N` | turn structure. `\|upkeep\|` marks the end of residuals; end-of-turn replacement `\|switch\|` lines come between `\|upkeep\|` and `\|turn\|`. |
| `\|win\|NAME`, `\|tie` | terminal |
| side updates `\|request\|JSON`, `\|error\|[Invalid choice] ...` / `[Unavailable choice]` | decisions (§3) |

### 10.3 `State.serializeBattle` (`sim/state.ts:61-76`) versus the protocol

**Format.** A JSON tree that reflects the object graph:
- **Skip lists.** Constants and caches are omitted (`sim/state.ts:39-50`).
- **References.** Circular and immutable references become `"[Type:id]"` strings (`toRef`/`fromRef`, `:377-409`).
  - **Pokemon refs are position-based:** `[Pokemon:p1c]` means `side.pokemon[2]` *in that snapshot* (`POSITIONS[pokemon.position]`). Positions change on every switch, so resolve refs through the same snapshot's `side.pokemon`.
- **Team string.** `side.team` is encoded as the permutation string that maps the current `side.pokemon` order back to the submitted team (`:180-202`). It works with only 4 of 6 brought. Verified: `team="1243"` after preview, and deserialization succeeded.
- **ActiveMoves** are diffed against the dex move (`:273-286`).
- **Extras.** `prng` = current seed state; `queue` = the full pending action list (with refs) (`:68-73`).

**Verified example** (`dump_state.js`, at a U-turn pause; about 30 KB of JSON):

| Level | Fields |
|---|---|
| BATTLE | `turn`, `midTurn`, `requestState`, `ended`, `winner?`, `faintQueue`, `lastMove`, `lastSuccessfulMoveThisTurn`, `effectOrder`, `speedOrder`, `prng`, `queue`, `inputLog`, `log`, `supportCancel`, ... |
| FIELD | `weather`, `weatherState{duration, source, effectOrder}`, `terrain`, `terrainState`, `pseudoWeather{trickroom:{duration,...}}` |
| SIDE | `pokemonLeft`, `active[refs]`, `sideConditions{tailwind:{duration,...}}`, `slotConditions[2]`, `faintedThisTurn`, `faintedLastTurn`, `totalFainted`, `team`, `choice{actions, forcedSwitchesLeft, forcedPassesLeft, switchIns, mega, cantUndo}`, `activeRequest` (or a null tombstone) |
| POKEMON | `species`, `baseSpecies`, `position`, `hp`, `maxhp`, `baseMaxhp`, `status`, `statusState{...}`, `boosts`, `moveSlots[{id, pp, maxpp, disabled, disabledSource, used}]`, `item`, `itemState`, `lastItem`, `ability`, `baseAbility`, `abilityState`, `volatiles{id:{duration, counters, source}}`, `types`, `storedStats`, `baseStoredStats`, `isActive`, `fainted`, `faintQueued`, `activeTurns`, `activeMoveActions`, `trapped`, `maybeTrapped`, `switchFlag`, `forceSwitchFlag`, `newlySwitched`, `lastMove`, `lastMoveTargetLoc`, `moveThisTurn`, `moveThisTurnResult`, `speed` (cached), `canMegaEvo`, `attackedBy`, `timesAttacked`, `hurtThisTurn`, `statsRaised/LoweredThisTurn`, `usedItemThisTurn`, `set` |

**Recommendation.**
- **Primary diff source:** `serializeBattle` at every decision point. It is complete:
  - durations and counters (weather, terrain, Trick Room, screens, Tailwind, Encore, Taunt, sleep and toxic counters, the Protect `stall` counter);
  - PP and disabled flags, `activeMoveActions` (needed for the [C] Fake Out mask), trap flags;
  - the pending queue and `switchFlag`s during a pause.

  None of this is recoverable from the protocol. It also gives free "start-from-any-state" fixtures, since `deserializeBattle` plus `restart` works mid-turn (verified).
- **Projection.** Map each snapshot to a canonical, position-independent projection:
  - for each side, index mons by `set.species` (not position); record slot occupancy and the fields listed above;
  - **drop** `effectOrder`, `speed`, `log`, `inputLog`, `prng`, `hints` and `lastMove` ActiveMove internals;
  - normalize refs.
- **Secondary source:** the omniscient protocol log, for within-turn event order (which action produced which change, `[spread]` hit lists, per-target damage) and for alignment with real replays.
- **Granularity.** Diff the snapshot after each `step`. To localize a divergence inside a step, run PS with a `runAction` hook that serializes after every action.

---

## 11. JAX state-machine mapping

### 11.1 Phases and decision points (mirroring `requestState`)

```
TEAM_PREVIEW            both sides act: ordered 4-of-6 (mask: 360 orderings; 180 if bench order canonicalized)
MOVE                    both sides act: per-slot action, joint mask (§4.9); slot auto-pass for fainted/commanding
SWITCH                  sides with any forceSwitch flag act; other side all-pass ('wait').
                        Per flagged slot: bench index or pass (pass only if forced). Revival Blessing variant: fainted index.
DONE                    terminal (win/tie); step is a no-op (lax.cond)
```

**State to carry between steps** (scratch for turns):
- `phase`, `turn`, `mid_turn` (bool);
- the queue (§11.3) and `switch_flag[2,2]` (0 none, 1 faint, 2 pivot, 3 revival), `force_switch_flag[2,2]`, `forced_passes[2]`;
- the per-turn `fractional_priority` and `quick_claw_activated` of each queued move;
- `fainted_this_turn[2]`, `fainted_last_turn[2]`;
- per-mon `active_turns`, `active_move_actions`, `trapped`, `move_disabled[4]` (recomputed in `endTurn`), `can_mega` per side.

### 11.2 `step` structure

```
step(state, a0, a1, key):
  if phase == DONE: return state
  # 1. apply choices
  if phase == TEAM_PREVIEW: reorder party per side; pokemonLeft=4; place leads; queue <- 4 runSwitch (batched)
  if phase == MOVE:   queue <- build_turn_queue(a0,a1)  # beforeTurn, [beforeTurnMove|priorityCharge], switch, megaEvo, move, residual
                      roll fractional priority (Quick Claw/Quick Draw) per move entry  (event ID: FRAC_PRIO)
                      compute commit-time speed/priority for every entry; tiebreak keys for order<200 entries (SPEED_TIE)
  if phase == SWITCH: prepend instaswitch entries (sorted by outgoing mon's commit-time speed), keep remainder
  # 2. resolve until next decision point
  fori_loop(N_ACT):                                     # bounded actor-action loop (§11.4)
     if paused or done or queue has only residual left: no-op
     e = select_next(queue)                             # see 11.3
     run_action(e)                                      # masked branches: beforeTurn | beforeTurnMove | revival | switch | instaswitch(+batched runSwitch)
                                                        #                  | runSwitch | megaEvo | priorityCharge | move
     post_action(e)                                     # phazing(dragIn, RANDOM_SWITCHIN) -> faints/checkWin -> Update -> EE -> switch-request check
  if not paused and not done and queue has only residual left:
     residual()                                         # one fixed block: per-order handler table (§8.2), faints resolved per handler, checkWin
     post_residual()                                    # checkFainted + Update + EE -> maybe SWITCH (end-of-turn)
  if not paused and not done and residual done:
     replacement_loop: (the SWITCH decision itself returns to the agent; on resume the batch of instaswitches + one SwitchIn
                        runs in run_action, then checkFainted again -> maybe SWITCH again)
     if nothing pending: end_turn()                     # turn++, resets, DisableMove/trap recompute, Fake-Out[C] mask, turn>1000 -> tie; phase=MOVE
  return state
```

**Note on branches.** Under `vmap`, `lax.switch` runs every branch. Keep `residual()` and `end_turn()` **outside** the per-action loop so they run once per step, not N_ACT times. Gate them with `lax.cond` on a scalar only when not vmapped; otherwise use a mask.

### 11.3 Queue representation and selection (exactly equivalent to PS)

**Fixed array `Q[QMAX]`.** Each entry: `kind`, `order`, `prio` (integer ×10 so +0.1 fractions are exact), `speed`, `side`, `actor_mon` (party index), `move_slot`, `move_id`, `target_loc`, `original_target_mon`, `mega`, `source_effect` (instruct/encore), `valid`.

**Selection rule:** `argmax` over valid entries of `(−order, prio, speed, u)`, where `u` is a fresh uniform tiebreak per selection. Before the selection, **if the best entry is a move**, recompute for every valid move entry:
- `speed` = signed effective speed: [C] `getStat('spe')`, negated under Trick Room;
- `prio` = base priority + ModifyPriority (Prankster, Gale Wings with current HP, Triage, Grassy Glide with current terrain) + the stored fractional priority.

This is exactly PS's re-sort (§5.7):
- entries with order < 200 keep their commit-time (or insertion-time) keys;
- moves are re-keyed before each execution;
- uniform random tie-breaking per selection has the same distribution as PS's "shuffle the tied group, take the first".

**Order classes as constants:**

| Class | Order |
|---|---|
| prioritized | 3 (After You, Instruct, Round) |
| instaswitch | 3 |
| beforeTurn | 4 |
| beforeTurnMove | 5 |
| revival | 6 |
| runSwitch | 101 |
| switch | 103 |
| megaEvo | 104 |
| priorityCharge | 107 |
| move | 200 |
| quashed | 201 |
| residual | 300 |

Note that a prioritized move (order 3) and an instaswitch (3) can coexist after a pause. PS then orders them by priority and speed; the instaswitch has priority 0.

**Mutation semantics to reproduce:**

| PS operation | JAX equivalent |
|---|---|
| `prioritizeAction` | set `order = 3` |
| Quash | set `order = 201` |
| Instruct | write a **new** move entry (order 3) into a free slot; fractional priority is re-rolled because it calls `resolveAction` |
| `cancelAction(p)` (outgoing mon on switch-out or drag) | invalidate p's entries except runSwitch |
| [C] Encore `changeAction` | rewrite p's move entry: new move id, random target, recompute priority |
| a fainted or inactive actor's move | skipped at selection (invalid) |

**Batched SwitchIn.** When an instaswitch or start batch completes, run one SwitchIn resolution over all entering mons:
- a speed ranking with ties broken once (SPEED_TIE event);
- handlers ordered by (priority tier, ranked speed, subOrder ability-before-item, creation order);
- for a single start-of-turn `switch`, run the SwitchIn for that mon **immediately** after it.

### 11.4 Bounds (Reg M-C at `d601443`), with justification

**B1. Primary actions per turn ≤ 4.** One per active slot (`sim/side.ts:1204`). Auto-passes produce no action (`sim/battle-queue.ts:168`).

**B2. Satellite actions per turn ≤ 4 + 2.**
- Per move action, at most one of beforeTurnMove (Counter, Mirror Coat) or priorityChargeMove (Focus Punch, Beak Blast, Chilly Reception). No move has both. See `sim/battle-queue.ts:207-248`.
- megaEvo ≤ 1 per side (`sim/side.ts:780`), so ≤ 2.

**B3. Queue-inserted moves ≤ 2.**
- Only Instruct creates a new move action (`data/moves.ts:9665-9670`). Instruct can't instruct Instruct (`failinstruct`, `:9651, 9657`).
- Only Oranguru learns it, and Species Clause gives ≤ 1 per side. So ≤ 2 per turn.
- Encore's `changeAction` replaces an entry, so the net change is 0. After You, Quash and Round only reorder.
- Dancer would add inline `runMove`s, not queue entries, and is not in Reg M-C. For a general dex, budget up to 3 inline extra moves per dance move.

**B4. Fixed per-turn entries:** beforeTurn 1, residual 1.

**B5. runSwitch entries:**
- ≤ 1 alive at a time for start-of-turn switches, because each is consumed before the next switch;
- ≤ 4 for a batch (leads, or 4 simultaneous replacements);
- they are consumed inside one `runAction` (`sim/battle-actions.ts:230-233`).

**QMAX.** Live entries at once ≤ 4 (B1) + 4 (B2a) + 2 (B2b) + 2 (B3) + 2 (B4) + 4 (B5) = **18**.
- With a per-kind layout (each move entry carries `has_charge` / `has_beforeturn` / `has_mega` flags), this shrinks to **10 entries**: 4 primary, 2 Instruct, 4 runSwitch/instaswitch.
- beforeTurn and residual become fixed phases.

**N_ACT: action executions between two decision points.**
- **Turn start:** beforeTurn 1 + ≤ 4 satellites + ≤ 2 megas + ≤ 4 primary + ≤ 4 runSwitch (only if those primaries are switches) + ≤ 2 Instruct.
  - A slot either switches (2 executions: switch and runSwitch) or moves (≤ 3: satellite, mega, move).
  - So the maximum is 1 + 4×3 − 2 (mega cap) + 2 = **13**.
- **Resume after a pause:** ≤ 4 instaswitch + 1 runSwitch + the remainder of the turn (≤ 12).
- **Recommendation: N_ACT = 16.** Add an overflow counter that is asserted to be 0 in the harness.

**Mid-turn interrupts per turn.**
- Each executed move action raises **at most one** `switch` decision, because all flags raised during an action are collected into one request (`sim/battle.ts:2936-2969`). That gives ≤ 6 (B1 + B3).
- Each mid-turn switch-in's `runSwitch` can add one more: Emergency Exit from hazard damage on Golisopod (`sim/battle.ts:2932-2934`).
- After residual, one combined end-of-turn decision (faint replacements plus residual Emergency Exit).
- Then ≤ 2 more replacement loops per side (each loop needs a replacement to faint on entry, and a 4-mon side has ≤ 2 bench mons). Simultaneous loops on both sides share one decision, so ≤ 2.
- **Practical bound: ≤ 10 SWITCH decisions per turn.**

This doesn't need to be a static loop bound if each decision is its own `step`. It matters only for fixed-length rollouts.

**Battle length.** ≤ 1000 MOVE decisions (`sim/battle.ts:1856-1860`), plus SWITCH decisions.

### 11.5 Turn-level random events (for the forced-outcome hook)

| Event id | PS site | Notes |
|---|---|---|
| `SPEED_TIE_QUEUE` | `queue.sort` at commit (`sim/battle.ts:3076`) and every dynamic re-sort (`:2983`) | Record the realized order, not raw draws. |
| `SPEED_TIE_SWITCHIN` | `runSwitch` `speedOrder` sort (`sim/battle-actions.ts:235`) | Fixes the per-batch ranking. |
| `SPEED_TIE_HANDLERS` | `fieldEvent` and `runEvent` handler sorts (`sim/battle.ts:512, 801`); `eachEvent` actives (`:473`) | Residual ties (e.g. both Tailwinds ending), Update order. |
| `FRAC_PRIO` | Quick Claw (`data/items.ts:5017`), Quick Draw (`data/abilities.ts:3739`) at `resolveAction` | Once per move action per turn, plus once for an Instruct-created action. |
| `RANDOM_TARGET` | `getRandomTarget` → `randomFoe` / `sample` (`sim/battle.ts:2523, 2537`) | At resolve (no target loc: auto-choose, [C] Encore re-queue, Instruct) and at execution retarget. Re-sort calls are no-op draws. |
| `RANDOM_SWITCHIN` | `dragIn` → `getRandomSwitchable` (`sim/battle.ts:1584-1587`) | Roar, Whirlwind, Dragon Tail, Circle Throw, Red Card. |
| `ALLY_SWITCH_STALL` | `randomChance(1, counter)` (`data/moves.ts:343`) | Consecutive Ally Switch. |
| `INSERT_POS` | `insertChoice` random index among equal keys (`sim/battle-queue.ts:406`) | Outcome-irrelevant for runSwitch (it is batched). |
| `TEAM_SORT` | preview `queue.sort` ties | Outcome-irrelevant. |

### 11.6 PS quirks a faithful engine must reproduce (turn level)

1. **Start-of-turn switches resolve one at a time.** Each switch-in's abilities fire before the next switch.
2. **Switch and Mega order uses commit-time speeds of the *outgoing* or evolving mon.** Under Trick Room the order is reversed ([C] negation).
3. **Mega Evolution happens after switches, before priority-charge messages.** The new speed applies to that turn's moves.
4. **Protect, Wide Guard and Quick Guard fail if no move or switch action remains in the queue** (`willAct`).
5. **The fractional-priority roll happens at commit**, even if the mon later can't move. Instruct re-rolls it.
6. **A KO'd mon's queued move is silently skipped.** No `cant` line, no PP use.
7. **Fainted slots stay empty until after residuals.** End-of-turn replacements are batched for SwitchIn. The replacement switch order uses the fainted mon's speed.
8. **Simultaneous wipe-out:** the side owning the last-processed faint wins.
9. **[C] Encore re-queues the target's move:** random target, new priority (for example Protect's +4 takes effect that turn).
10. **[C] Fake Out and First Impression are unselectable after the first move action.** Belch and Stuff Cheeks are always selectable.
11. **`move X mega` on a locked or Struggling mon is accepted but the mega is ignored.** A locked move must be chosen without a target.
12. **Emergency Exit after a batched SwitchIn checks only the first switcher** (`sim/battle.ts:2932-2934`).

---

## Appendix: scripts and verification runs (`research/ps-loop/`)

Run any script with `PS_DIST=<built ps>/dist node <script>`. The original `/tmp/psrepo/ps/dist` is gone; a rebuilt identical copy is at `/tmp/ps-turnloop/dist`.

| Script | What it does |
|---|---|
| `ruletable.js` | Resolved rule table for `gen9championsvgc2026regmc`: picked size 4, no EBC, Cancel Mod, timer. |
| `legality.js` | `isNonstandard` status and number of legal learners or holders for flow-relevant moves, items and abilities. |
| `switchin_handlers.js` | Every non-Past SwitchIn / onStart handler with its priority (§2.7 table). |
| `residual_handlers.js` | Every non-Past residual handler and duration effect with order and subOrder (§8.2 table). |
| `run_battle.js [seed] [--requests] [--state-at=T]` | Random-legal-choice battle using Champions random-doubles teams. Prints requests, log, or a serialized state. |
| `scenarios.js <name>` | Scripted checks, listed below. |
| `dump_state.js` | Shape of `serializeBattle` at a U-turn pause. |

**Scenarios in `scenarios.js`:**
- **`preview`:** team-string parsing and padding; `side.pokemon` reduced to 4; `pokemonLeft = 4`.
- **`uturn`:** the mid-turn `switch` request (pivot side gets `forceSwitch`, other side `wait`), queue contents at the pause, `[from] U-turn`, the second pause for Parting Shot, and an identical continuation after serialize/deserialize.
- **`megaspeed`:** Mega speed takes effect the same turn.
- **`icywind`:** dynamic re-sort after a speed drop.
- **`fewbench`:** `forceSwitch [true,true]` with 1 bench mon, giving `forcedPasses = 1`; legal and illegal strings.
- **`shadowtag`:** `trapped` vs `maybeTrapped` info hiding; switch rejected.
- **`instruct`:** Instruct-created action executes immediately.
- **`doubleswitch`:** start-of-turn switches are not batched.
