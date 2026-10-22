# 03 — State, data model and identifiers

All shapes are for **one battle**; vmap adds the leading batch dimension.
- **Storage dtypes:** int8 / int16 / uint32 / bool. Compute is always int32 (ADR-011).
- **Naming:** `side ∈ {0,1}` (PS `p1`, `p2`); `slot ∈ {0,1}` (PS position a/b); `b ∈ 0..3` = brought index (team-preview order); `s ∈ 0..5` = sheet index (team-sheet order).
- **Implementation:** `flax.struct.dataclass`. Group fields as shown.
- **Leaf count.** It only costs at the Python boundary (≈ 1 µs/leaf/call; free inside one jit/scan; `research/jax-patterns.md` §3C). Start with named leaves and a budget of ≤ 150. Pack scalar fields into dtype-grouped arrays with generated accessors only if profiling of harness replay or search shows the per-call overhead matters (M9).
- **Field semantics follow PS** (ADR-007). The "PS" column names the PS field each one mirrors; the canonical projection (07 §5) is generated from this mapping.

---

## 1. Identifier spaces (generated in `vgcjax/data/ids.py`)

| Space | Range | Notes |
|---|---|---|
| `SpeciesId` | 0 = none, 1..NS | Every battle-distinct forme: set-level species, Megas, in-battle formes (Aegislash-Blade, Mimikyu-Busted, Palafin-Hero, Castform-*, Morpeko-Hangry, …). Cosmetic formes collapse to one id, but the cosmetic name is kept for interop. |
| `MoveId` | 0 = none, 1..NM | All 510 legal moves, plus Struggle and any move callable by legal moves (Sleep Talk's, Copycat's, Instruct's targets are drawn from movesets). |
| `AbilityId`, `ItemId` | 0 = none, 1..N | Legal pool + Mega abilities |
| `ConditionId` | 0 = none, 1..NC | Statuses (brn, par, psn, tox, slp, frz, **fnt**: set only by checkFainted, cleared when switchIn swaps the fainted mon out), volatiles, side conditions, slot conditions, pseudo-weathers, weathers, terrains |
| `EffectId` (unified) | contiguous ranges `[MOVE_BASE..)`, `[ABILITY_BASE..)`, `[ITEM_BASE..)`, `[COND_BASE..)` | Used by hook tables and `source_effect` fields |
| `TypeId` | 0 = none, 1..18 = Normal … Fairy, 19 = `???` (typeless, e.g. Struggle / Burn Up) | Champions has no Stellar; tables are [20, …] |
| `HookId` | 0..~40 | The hook taxonomy (04 §6) |
| `DrawKind` | enum | `contract/drawkey.md` §5 (forced) and §4 (tie kinds); generated with the mapping to `oracle/data/rng_callsites.json` |

Sub-ranges of `ConditionId` are dense per kind, so per-kind storage can index with `cid - KIND_BASE`:
- `VOL_*` (volatiles, NV of them);
- `SIDE_*` (NSC);
- `SLOT_*` (NSL);
- `PW_*` (pseudo-weathers, NPW);
- `WEATHER_*`, `TERRAIN_*`.

---

## 2. `Dex` (traced, device-resident; shapes are static per dex stage)

| Table | Shape / dtype | Content |
|---|---|---|
| `species_base_stats` | [NS,6] i16 | hp, atk, def, spa, spd, spe |
| `species_types` | [NS,2] i8 | second = −1 (none) when monotype (PS `types` array has length 1) |
| `species_weight_hg` | [NS] i16 | |
| `species_abilities` | [NS,4] i16 | slots 0, 1, H, S (for legality and Trace/obs) |
| `species_mega` | [NS, MAXSTONE] → (stone ItemId, mega SpeciesId) | mega mapping (≤2 per species: X/Y, base/Z) |
| `species_flags` | [NS] u32 | isMega, cosmetic, battleOnlyForme, requiredItem, … |
| `move_core` | [NM, K] i16 | bp, type, category, accuracy (0 = always hits), priority, pp_max (Champions formula applied), crit_ratio, target_enum, multihit_min/max, drain_num/den, recoil_num/den, heal_num/den, ohko, self_switch kind, force_switch, status, volatile_status, side_condition, slot_condition, pseudo_weather, weather, terrain, … |
| `move_flags` | [NM] u32 | contact, protect, mirror, powder, sound, bullet, punch, bite, slicing, wind, pulse, defrost, bypasssub, reflectable, snatch, charge, recharge, gravity, distance, nonsky, mustpressure, metronome, failencore, failinstruct, … |
| `move_secondaries` | [NM, 2, K2] i16 | chance, boosts[7], status, volatile, self-boosts[7] (index 0/1 = secondary ord) |
| `move_self` | [NM, K3] i16 | self boosts + chance (PS `self`), selfdestruct kind |
| `move_misc` | [NM, K4] i16 | override_offensive_stat, override_defensive_stat, ignore_defensive, ignore_evasion, ignore_immunity, ignore_ability, tracks_target, will_crit, breaks_protect, sleep_usable, smart_target, multiaccuracy, struggle_recoil, has_crash_damage, … |
| `ability_flags`, `item_core` | [NA] u32, [NI,K] i16 | breakable, cantsuppress, fling bp, is_berry, natural_gift, mega stone, is_choice, … |
| `typechart` | [20,20] i8 | codes {0: neutral, 1: SE, 2: resist, 3: immune} as PS `damageTaken`, + status-immunity pseudo-columns (brn/par/psn/tox/frz/sandstorm/hail/powder/prankster/trapped) [20, K] |
| `hook_rows` | per hook: [R_h, …] (from `vgcjax/data/hook_rows.npz`) | MUL/GATE/REPL/DURATION rows (04 §6.3) |
| `hook_local_idx` | [NH, NEFFECT] i16 | 0 = no bespoke handler; else 1-based branch index into that hook's switch |
| `effect_event_order` | [NEFFECT, NH, 3] i16 | static (order, priority, subOrder) per handler; PS `on<Event>Order/Priority/SubOrder` |
| `cond_meta` | [NC, K] i16 | default duration, duration-callback kind (item-extended: Light Clay / rock / Terrain Extender, ability-extended), residual order/subOrder, effectType default subOrder, `noCopy`/`baton pass` copyable, … |
| `residual_table` | [NR, K] i16 | static residual walk order (04 §9.4) |
| `learn_legal` | not needed at runtime | legality is offline only |
| `boost_table`, `acc_table`, `crit_table` | small i16 | stage multipliers ([2+n]/2 form), accuracy stages ([3+n]/3), crit [24, 8, 2, 1] |

`Dex` is built by `tools/datagen/codegen.py`, and the entity rows included depend on the stage (v0/v1/v2). An entity outside the stage keeps its id, but its `supported` bit is 0. Hitting it at runtime sets `ERR_UNSUPPORTED_EFFECT`.

---

## 3. `Team` (per battle; produced by `sample_teams` or the trace loader)

| Field | Shape / dtype | Notes |
|---|---|---|
| `species` | [2,6] i16 | sheet order |
| `item`, `ability`, `nature` | [2,6] i16 / i16 / i8 | |
| `gender` | [2,6] i8 | always set explicitly (unset → PS draws a PRNG value at construction) |
| `moves` | [2,6,4] i16 | 0 = empty |
| `sp` | [2,6,6] i8 | 0..32 each, sum ≤ 66 (hidden from the opponent) |
| `level` | const 50 | |
| `sheet_id` | [2] i32 | team-bank index (stats only; never observed) |

---

## 4. `State`

### 4.1 `BattleMeta`
| Field | dtype | PS | Notes |
|---|---|---|---|
| `phase` | i8 | `requestState` | PREVIEW=0, MOVE=1, SWITCH=2, DONE=3 |
| `turn` | i16 | `battle.turn` | |
| `step_counter` | i32 | — | RNG coordinate; increments every `step` |
| `mid_turn` | bool | `midTurn` | paused inside a turn (queue remainder valid) |
| `winner` | i8 | `winner` | −1 none, 0/1 side, 2 tie |
| `ended_reason` | i8 | — | KO / TURN_CAP / PS_TIE_1000 / ERROR |
| `effect_seq` | i32 | `effectOrder` counter | monotonic; stamped onto created conditions |
| `last_move_battle` | i16 | `battle.lastMove` (id) | Copycat, … |
| `last_successful_move_this_turn` | i16 | `lastSuccessfulMoveThisTurn` | |
| `error_flags` | u32 | — | ADR-014 |
| `faint_queue` | [8] i8 | `battle.faintQueue` | FIFO of side·4+b (−1 empty); the order decides double-KO winners (04 §11.1) |
| `faint_queue_len` | i8 | | |
| `rng_words` | — | — | **not stored**: derived per step from the key |

### 4.2 `Sides` (leading dim 2)
| Field | Shape / dtype | PS |
|---|---|---|
| `brought_sheet_idx` | [2,4] i8 | which sheet index each brought mon came from |
| `ps_order` | [2,4] i8 | PS `side.pokemon` order (brought indices); updated on every switch swap |
| `active` | [2,2] i8 | brought index occupying the slot, **including a fainted mon**; −1 only before `start` (04 §10.1) |
| `pokemon_left` | [2] i8 | `pokemonLeft` |
| `total_fainted` | [2] i8 | `totalFainted` (Last Respects / Supreme Overlord) |
| `fainted_this_turn`, `fainted_last_turn` | [2] bool | `faintedThisTurn/LastTurn` | Retaliate tests truthiness only |
| `mega_used` | [2] bool | derived `!canMegaEvo` on all mons after a mega |
| `side_cond_bits` | [2,⌈NSC/32⌉] u32 | `sideConditions` keys (bitfield) |
| `side_cond_state` | [2,NSC,4] i16 | fields: duration, layers, seq, extra (e.g. Wide Guard none) |
| `slot_cond_bits` | [2,2] u32 | `slotConditions` (Wish, Healing Wish, Future Sight, Revival Blessing …) (bitfield) |
| `slot_cond_state` | [2,2,NSL,6] i16 | duration, seq, source (side,b), hp (Wish), move (Future Sight), dmg snapshot fields as PS stores |
| `choice_*` | — | none: choices are step inputs, not state |

### 4.3 `Mons` (leading dims [2,4] = brought mons)
| Field | dtype | PS | Notes |
|---|---|---|---|
| `species` | i16 | `species` | current forme (Mega etc.) |
| `base_species` | i16 | `baseSpecies` | Mega becomes baseSpecies in Champions after mega |
| `types` | [2] i8 + `added_type` i8 | `types`, `addedType` | Forest's Curse / Trick-or-Treat / Soak / Burn Up |
| `stats` | [5] i16 | `storedStats` | atk, def, spa, spd, spe |
| `hp`, `maxhp` | i16 | | |
| `ability`, `base_ability` | i16 | | |
| `ability_state` | [4] i16 | `abilityState` | e.g. Unburden flag, Protean used, Disguise busted …; per-ability schema from codegen |
| `ability_seq`, `item_seq` | i16 | `abilityState.effectOrder`, `itemState.effectOrder` | stamped at switchIn (non-empty id), setAbility, setItem; used by the priorityEvent comparator (04 §6.4) |
| `syrup_triggered` | bool | `syrupTriggered` | Supersweet Syrup; persists across switches; never reset |
| `item` | i16 | `item` | 0 = none |
| `item_state` | [2] i16 | `itemState` | |
| `last_item` | i16 | `lastItem` | Recycle / Pickup / Harvest |
| `ate_berry`, `used_item_this_turn` | bool | | Belch, … |
| `move_ids` | [4] i16 | `moveSlots[].id` | changes with Transform / Mimic |
| `pp`, `maxpp` | [4] i8 | | |
| `move_disabled` | [4] i8 | `moveSlots[].disabled` | 0, 1 = true, 2 = 'hidden'; recomputed per request (04 §4) |
| `move_used` | [4] bool | `moveSlots[].used` | |
| `base_move_ids`, `base_pp` | [4] | `baseMoveSlots` | Transform restore |
| `status` | i8 | `status` | includes `fnt` |
| `status_state` | [4] i16 | `statusState` | slp: `startTime`, `time`; tox: `stage`; frz: `time` [C]; seq |
| `boosts` | [7] i8 | atk, def, spa, spd, spe, accuracy, evasion | |
| `vol_bits` | [⌈NV/32⌉] u32 | `volatiles` keys | presence bitfield (static bit constants `VOL_X = (word, bit)`); measured 20× cheaper than `bool[NV]` (`research/jax-patterns.md` §3E); no uint64 (needs x64) |
| `vol_state` | [NV,6] i16 | col 0 = duration, 2 = seq, 3 = source (side·4+b, −1) are fixed. Cols 1, 4 and 5 are assigned per condition by `gen-projection-map` in first-seen key order, frozen in `projection_map.json`; new keys append | 07 §5.3 |
| `fainted`, `faint_queued` | bool | | |
| `is_active`, `position` | bool, i8 | | derived from `Sides.active` but cached |
| `active_turns`, `active_move_actions` | i16 | `activeTurns`, `activeMoveActions` | Fake Out [C] masking |
| `newly_switched`, `being_called_back`, `dragged_in`, `is_started` | bool | | |
| `switch_flag` | i16 | `switchFlag` | 0 = false, 1 = true (Eject Button / Emergency Exit), otherwise the source move's EffectId (U-turn etc.); faint sets 0 |
| `force_switch_flag` | bool | `forceSwitchFlag` | |
| `speed_cached` | i16 | `pokemon.speed` | written by updateSpeed (commit of MOVE **and** SWITCH, before each Block-B re-key, start of residual, insertChoice for that mon) and by setSpecies (= raw stored Spe: init, clearVolatile on switch-out/faint, formeChange incl. Mega, Transform). Never written by dragIn (04 §5.4) |
| `last_move`, `last_move_used`, `last_move_target_loc` | i16, i16, i8 | | |
| `move_this_turn` | i16 | `moveThisTurn` | |
| `move_this_turn_result`, `move_last_turn_result` | i8 | 0 undefined, 1 null, 2 false, 3 true | Stomping Tantrum, Truant |
| `hurt_this_turn` | i16 | `hurtThisTurn` | −1 = null |
| `last_damage` | i16 | `lastDamage` | |
| `attacked_by` | ring [K_ATT=8, 6] i16 + head i8 | `attackedBy[]` (source side·4+b, damage, damage_is_number, this_turn, move, slot) | PS appends one entry per move per target. Consumers read the newest entries. Evicting a `this_turn` entry sets an overflow flag; evicting older ones is silent. Counter / Mirror Coat / Revenge / Assurance / Metal Burst / Comeuppance |
| `times_attacked` | i8 | `timesAttacked` | Rage Fist [C] reset on switch |
| `stats_raised_this_turn`, `stats_lowered_this_turn` | bool | | Burning Jealousy, Lash Out |
| `trapped`, `maybe_trapped` | i8 | | `trapped` true / 'hidden' |
| `transformed` | bool | | |
| `illusion` | i8 | `illusion` (brought index disguised as, −1) | |
| `can_mega` | bool | `canMegaEvo` | computed at construction from the stone |
| `weight_hg` | i16 | `weighthg` | |
| `previously_switched_in` | i8 | | |
| `truant_turn` | bool | | |

### 4.4 `Field`
| Field | dtype | PS |
|---|---|---|
| `weather` | i8 | `weather` id |
| `weather_state` | [3] i16 | duration, seq, source (side·4+b) |
| `terrain`, `terrain_state` | i8, [3] i16 | |
| `pw_bits` | u32 | `pseudoWeather` keys (Trick Room, Gravity, Magic Room, Wonder Room, Fairy Lock, …) (bitfield) |
| `pw_state` | [NPW,3] i16 | duration, seq, extra |

### 4.5 `Queue` (fixed size QMAX = 14; ADR-009; 04 §12)
| Field | Shape / dtype | Meaning |
|---|---|---|
| `valid` | [Q] bool | |
| `prepend` | [Q] bool | set on entries created by a SWITCH-phase commit; the first selection key; cleared by the re-key before a move (04 §5.5, §2.1) |
| `kind` | [Q] i8 | BEFORE_TURN, BEFORE_TURN_MOVE, REVIVAL, SWITCH, INSTASWITCH, RUN_SWITCH, MEGA, PRIORITY_CHARGE, MOVE, RESIDUAL |
| `order` | [Q] i16 | PS order class (3, 4, 5, 6, 101, 103, 104, 107, 200, 201, 300) |
| `priority10` | [Q] i16 | priority × 10 plus fractional (Quick Claw +1 means +0.1 → stored ×10 exact; Stall −1) |
| `frac_prio10` | [Q] i8 | fractional part, rolled once at resolve |
| `speed` | [Q] i16 | signed key (negated under Trick Room [C]) |
| `side`, `actor_b` | [Q] i8 | |
| `move_slot`, `move_id` | [Q] i8, i16 | |
| `target_loc` | [Q] i8 | PS signed target location relative to user (+1, +2 foes, −1, −2 allies/self) |
| `original_target` | [Q] i8 | side·4+b at resolve (tracksTarget) |
| `mega` | [Q] bool | |
| `switch_target_b` | [Q] i8 | incoming brought index |
| `source_effect` | [Q] i16 | Instruct / Encore / U-turn … |
| `seq` | [Q] i16 | insertion order (stable tiebreak bookkeeping only) |

### 4.6 `TurnScratch` (reset at end of turn)
- `quick_claw_activated[2,4]`, `fainted_this_action`, `forced_passes[2]`, `instruct_inserted` (count).
- Overflow counters: `actions_executed_a`, `actions_executed_b`.
- **Tie ordinals** (`contract/drawkey.md` §4), i16 each: `select_ord`, `runswitch_ord`, `fieldevent_ord`, `eachevent_ord`, `mulhook_ord`. They persist across SWITCH steps within a turn and reset in `end_turn`.
- Switch requests are derived from `switch_flag` and `revival` slot conditions. There is no separate pending-request field.

### 4.7 `PublicEvents` (ring buffer, E = 32)
Each record is `[t, kind, actor(side·4+b), target, move, value1, value2, flags]` i16:
- kinds: SWITCH_IN, MOVE_USED, DAMAGE_PCT, HEAL_PCT, CRIT, SUPER_EFFECTIVE, RESISTED, IMMUNE, MISS, PROTECTED, FAIL, STATUS, BOOST, FAINT, ITEM_REVEALED, ITEM_CONSUMED, ABILITY_REVEALED, MEGA, WEATHER, TERRAIN, FIELD/SIDE start/end, TURN.
- It records only what the PS protocol shows players, with HP as **percentage per the Champions rule**.
- The observation reads the last 16 events. The buffer is also the input for `protocol_reducer` round-trip tests.

### 4.8 `RevealBits` (per observer) — `[2 observer, 2 side, 4 brought]` + `[2,2,6]` sheets
- `brought_known` (unknown / brought / not-brought), `revealed` (has appeared), `item_known_state`, `ability_known_state`.
- `mega_seen`, `moves_seen[4]`, `pp_used_observed[4]`.
- Under OTS most of these are known from the start. The bits exist so closed sheets stay possible.

---

## 5. Step I/O pytrees

```python
class Actions(NamedTuple):          # per battle
    a: i32[2]                       # PREVIEW: 0..359; MOVE/SWITCH: joint index 0..840 (= slot0*29 + slot1)

class ForcedOutcomes(NamedTuple):   # only when EngineConfig.forced = True; KF = 160 per step
    # one row per forced draw or tie record; columns = the DrawKey fields of contract/drawkey.md §3
    kind, turn, step_tag, step_actor, step_subject, actor, target, subject, hit, ord, sub, action_ord  # ints
    value:  i16[KF]                 # outcome in the engine value domain (drawkey.md §5; ties: winner or permutation index)
    valid:  bool[KF]

class DrawLog(NamedTuple):          # only when EngineConfig.draw_log = True
    same DrawKey columns as ForcedOutcomes + arg0:i32, arg1:i32 (e.g. numerator/denominator, n options) + value
    + used_forced: bool + is_tie: bool + tied_mask: u8 (members, for tie records)
    count: i32

class StepOut(NamedTuple):
    obs: Obs[2]; final_obs: Obs[2]          # final_obs differs from obs only on auto-reset steps (08 §5)
    masks: Masks[2]                          # Masks = {preview: bool[360], slot: bool[2,29], pair_ok: bool[29,29]} (08 §3)
    reward: f32[2]; terminated: bool; truncated: bool
    has_decision: bool[2]; phase: i8; events: PublicEvents (new this step); error_flags: u32
    shaping: f32[2,4]                        # HP-fraction sum, mons remaining, fainted, mega used
```

---

## 6. Invariants (checked in debug builds via `checkify` and in property tests)

1. `0 ≤ hp ≤ maxhp`; `fainted ⇔ hp == 0` outside the faint-processing window.
2. `active[s,k] = b ⇒ position[b] == k`; `is_active[b] ⇒ ¬fainted[b] ∧ b ∈ active[s]`; no mon occupies two slots. A fainted mon may occupy a slot with `is_active = False`.
3. `pokemon_left[s] == count(¬fainted over brought)` outside faint processing.
4. Boosts are in [−6, 6]. PP is in [0, maxpp].
5. `ps_order[s]` is a permutation of 0..3, and the active mons occupy `ps_order[s][0..1]` in slot order (the PS invariant).
6. If a volatile's presence bit is clear, then `vol_state[v] == 0`; the same holds for side, slot and pw state.
7. `phase == DONE ⇒ step` is a no-op returning identical state.
8. `mega_used[s] ⇒ ¬can_mega` for every mon of `s`.
9. `queue.valid` count ≤ QMAX. Overflow sets an error flag.

---

## 7. Memory estimate
- Mons: about 120 int16-equivalents of scalars plus NV·7 (NV ≈ 60 → 420), so about 1.1 KB per mon and about 9 KB for [2,4].
- Sides, field, queue and events: about 1.5 KB. Total about 10–11 KB per battle.
- At 64k battles that is about 700 MB of state. This is acceptable but above the scope doc's 2 KB guess.
- If memory or bandwidth binds (09), compress `vol_state` to only the fields each volatile uses, through codegen-generated compact offsets.
