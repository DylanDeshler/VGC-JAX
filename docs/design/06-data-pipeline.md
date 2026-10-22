# 06 — Data pipeline (PS → tables → Dex; teams → TeamBank)

Everything content-related is generated (ADR-020). The pipeline is deterministic: the same PS commit, stats month and stage always produce byte-identical outputs. A `DATA_VERSION` hash (PS commit, stats month, stage, datagen git hash) is embedded in `vgcjax/data/ids.py` and checked at engine load.

## 1. Stages of the pipeline

```
oracle/ps (pinned)
   │  node tools/datagen/export_dex.js --mod champions --format gen9championsvgc2026regmc   (format is a parameter; ADR-028)
   ▼
data/export/{species,moves,abilities,items,conditions,typechart,natures,formats}.json     (lossless, PS field names)
   │  node tools/datagen/legal_pool.js        → data/export/legal_pool.json        (TeamValidator-based)
   │  node tools/datagen/census.js            → data/export/handler_census.json    (fns, templates, hooks, orders)
   │  python tools/datagen/dex_stages.py      → data/export/stages/{v0,v1,v2}.json (needs data/raw usage stats + replay sheets)
   │  (manual/agent) tools/datagen/handler_classification.json                     (ROW_* vs BESPOKE, predicates)
   ▼
python tools/datagen/codegen.py --stage vN
   ▼
vgcjax/data/ids.py            enums: SpeciesId, MoveId, AbilityId, ItemId, ConditionId (+ kind sub-ranges),
                              EffectId ranges, TypeId, HookId, DrawKind, PREVIEW_ORDERS, DATA_VERSION
vgcjax/data/tables.npz        all Dex tables (03 §2)
vgcjax/data/hook_rows.npz     MUL/GATE/REPL/DURATION rows per hook
vgcjax/data/handlers_stub.py  BESPOKE registration stubs (only for new handlers; never overwrite implemented ones)
(projection_map.json is NOT a codegen output: generated in M2.8 by `harness.cli gen-projection-map` from a core corpus,
 re-run per stage; see 07 §5.3)
```

## 2. `export_dex.js` requirements
- Use `Dex.mod('champions')`, which merges the base gen 9 data with the mod overrides. Export **merged** values; don't re-merge yourself.
- **Species.** For each species: base stats, types, weight, abilities by slot, `otherFormes`, `battleOnly`, `requiredItem(s)`, `isMega`, `megaStone` relationships, `changesFrom`, cosmetic-forme flags.
  - Emit Mega rows from the items' `megaStone` map: `{baseSpecies: megaSpecies}`. Note that Meowstic-M and Meowstic-F share one stone.
- **Moves.** All fields used by the engine:
  - `basePower`, `type`, `category`, `accuracy` (true → 0 flag), `priority`, `pp`, `noPPBoosts`, `critRatio`, `target`, `flags`;
  - `secondary`/`secondaries`, `self`, `selfBoost`, `boosts`, `drain`, `recoil`, `heal`, `multihit`, `multiaccuracy`;
  - `status`, `volatileStatus`, `sideCondition`, `slotCondition`, `pseudoWeather`, `weather`, `terrain`, `selfSwitch`, `forceSwitch`, `ohko`;
  - `overrideOffensiveStat`/`overrideDefensiveStat`/`overrideOffensivePokemon`, `ignoreDefensive`, `ignoreEvasion`, `ignoreImmunity`, `ignoreAbility`, `tracksTarget`, `willCrit`, `breaksProtect`, `sleepUsable`, `smartTarget`, `stallingMove`, `selfdestruct`, `hasCrashDamage`, `mindBlownRecoil`, `struggleRecoil`, `callsMove`, `damage` (fixed), `thawsTarget`, `hasSheerForce`;
  - **plus** the names of function-valued keys, so the census can cross-check.
- **PP.** Apply the Champions formula **in datagen**: `pp_c = min(pp, 20)`, then `maxpp = noPPBoosts ? pp_c : (pp_c / 5 + 1) * 4`. Assert it against a constructed PS battle's `moveSlots[].maxpp` for every move (CH §1.1–1.2).
- **Conditions.** For each condition reachable from the stage's entities, plus all of `data/conditions.ts` and `ch/conditions.ts`:
  - `duration`, `durationCallback` presence and kind, `effectType`;
  - every `on<Event>Order`/`Priority`/`SubOrder`;
  - `noCopy`, plus Baton Pass copyability (PS's `pokemon.ts` lists and volatile flags).
- **Handler order metadata.** For every entity × event: `(order, priority, subOrder)` as PS `resolvePriority` would compute it, including the **default subOrder by effectType** (MV §1.8). Use PS's own code path:
  1. instantiate a dummy battle;
  2. call `battle.resolvePriority` on synthetic handler objects;
  3. record the computed fields.
  
  This avoids re-deriving PS defaults by hand.
- **Type chart.** `Dex.types` `damageTaken` codes, plus the immunity pseudo-keys (`brn`, `par`, `psn`, `tox`, `frz`, `sandstorm`, `hail`, `powder`, `prankster`, `trapped`).
- **Natures.** plus/minus stats.
- **Format.** The resolved rule table (`ruleset_resolved.json`, as in `research/champions/`) and the `pickedTeamSize`/`evLimit`/`adjustLevel` values. The engine asserts on them at load.

## 3. Datagen self-tests (run in CI on every datagen change)
Phase-in:
- tests 2 and 4 from M0.5;
- test 1 once engine tables exist (M1.1);
- test 3 once classification exists (M3.2), scoped to the current stage.
1. **Random-entity spot checks against a live PS battle.** Construct sets with chosen moves, items and abilities, and compare `storedStats`, `maxpp`, types and weight with the engine tables. 1,000 random legal sets.
2. **Coverage.** Every entity in the stage list has a row. Every condition referenced by a stage entity (census closure) has a row.
3. **Census consistency.** Every handler key in the census is classified in `handler_classification.json`, or the entity is outside the stage.
4. **Stability.** Re-running datagen produces a byte-identical output.

## 4. Team banks

### 4.0 Seed data (already downloaded during research; M0.1 / M0.7)

| Need | On disk now | Move to |
|---|---|---|
| Chaos stats, Reg M-C 2026-09 (8 files: `regmc` / `regmcbo3` × cutoffs 0/1500/1630/1760) | `research/dex/raw/stats_2026-09/` | `data/raw/smogon-chaos/2026-09/` |
| Chaos stats, Reg M-B 2026-08 (drift check only) | `research/dex/raw/stats_2026-08/` | `data/raw/smogon-chaos/2026-08/` |
| Real Bo3 sheets: 3,850 rows → **3,211 unique** packed teams. **No SP; 977 blank genders; rows include usernames** | `research/dex/raw/replays_2026-10/gen9championsvgc2026regmcbo3_teams.jsonl` | `data/raw/ps-replays/2026-10/` |
| Real Bo1 sheets (146 rows) | `…/gen9championsvgc2026regmc_teams.jsonl` | same |
| VGC-Bench Reg M-B Bo3 logs (26 MB; **licence unchecked**) | `research/dex/raw/vgcbench/` | `data/raw/vgcbench/2026-10/` |

- **Raw data hygiene.** `data/raw/` is git-ignored and holds `SOURCES.md` (URL, retrieval date, licence) and `MANIFEST.sha256`. Keep a tarball backup outside the repo.
- **Frozen real-sheet corpus.** `data/teams/real_sheets_mc_v1.jsonl` holds the 3,211 unique Bo3 sheets, with usernames and ratings stripped (rating bucket kept) and the sha256 recorded.
  - G-v0, G-v1 and G-v2 use **this frozen file**.
  - New scrapes go to `real_sheets_mc_v2` and enter gates only at a milestone boundary, by ADR.
- **Deterministic fills:**
  - SP per §4.2, seeded `sha256('spfill', canonical_sheet)`;
  - blank gender → the species' fixed gender if it has one, else `M` if allowed, else `F`.

### 4.1 Sources (`research/dex-scope.md` §7; `research/rng-harness-poc.md` §3)
| Pool | Source | Use |
|---|---|---|
| A: real sheets | PS Reg M-C Bo3 replay `\|showteam\|` lines (scraper from `research/dex/scripts/fetch_replay_teams.py`). VGC-Bench M-A/M-B logs and teams are legal under M-C | evaluation (held-out split) + 50% of training |
| B: usage sampler | Port `research/dex/scripts/team_sources.js`, the stage-aware sampler: Teammates chain, item/ability/move marginals restricted to the stage, Spreads in SP format, Mega base abilities from the replay distribution. `research/poc/lib/teams.js` supplies only `HarnessRNG` and pool C | training diversity |
| C: exploration | PS Champions random-doubles generator, repaired (level 50, re-draw duplicate items) | conformance coverage only |
| D: coverage-required | sampler variant with a required entity (`--require <id>`) | conformance targeting |

### 4.2 SP spreads
- Sheets from replays hide SP. Fill them by sampling from chaos `Spreads` for that species, conditioned on the shown nature. If no spread exists for that nature, fall back to the species' top spreads re-labelled with the nature.
- Training banks store **K = 4 candidate spreads per mon with prior weights**. `reset` samples one per battle (08 §6). The same table serves as the belief prior.

### 4.3 Validation and packing
- Every team passes oracle `validate_team` (TeamValidator) **and** the stage filter. Gender is always set.
- **Packing:** the `TeamBank` npz holds `species, item, ability, nature, gender: [N,6]`, `moves: [N,6,4]`, `sp_cand: [N,6,K,6]`, `sp_logit: [N,6,K]`, plus `meta.json` (source, rating, date, `DATA_VERSION`).
- **Split:** a deterministic hash of the canonical sheet string assigns 90% train and 10% held-out. Held-out teams never appear in training banks.

## 5. Usage stats and replay data hygiene
- Store raw downloads in `data/raw/<source>/<date>/` with `SOURCES.md` (URL and retrieval date). Treat downloads as untrusted data: parse as JSON only, never execute them.
- Re-pull monthly (Smogon publishes at month start) and when the regulation changes.

## 6. Re-pin and regulation-change checklist (ADR-021)
1. Bump `oracle/ps`, `npm ci`, `node build`.
2. `oracle check-callsites`. If it fails, update the inventory annotations and the DrawKind mapping (`contract/drawkey.md` §5, §8).
3. Run datagen, then `git diff data/export/`. Review the changed entities and add their rows to `docs/oracle-changelog.md`.
4. Re-run the census and classification check. New or changed handlers become catalogue `todo` rows.
5. Regenerate stages, if the regulation or stats changed.
6. Run the full conformance suite plus scenario tests. Triage new divergences against PS commits touching `data/mods/champions*` and `sim/`.
7. Land the re-pin only when the gates are restored.
