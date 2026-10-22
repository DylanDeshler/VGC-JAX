# 07 — Conformance: oracle harness, trace format, forcing contract, diffing, shrinking, coverage, gates

Proof of concept: `research/rng-harness-poc.md` (**RNG**), with code in `research/poc/`. Every load-bearing assumption was verified there:
- the instrumented PS log is byte-identical to an uninstrumented one;
- all draws are classified;
- context-keyed forcing reproduced 300/300 state trajectories;
- the serialize/deserialize round trip is exact (~1 ms);
- crafted states work;
- request-derived masks were never too strict.

Port that code; don't rewrite it from scratch.

---

## 1. Contract (ADR-015)

A battle is **clean** iff, at every decision point k and at the end:
1. **State.** `project(engine_state_k) == ps_projection_k`, field by field (§5).
2. **Masks.** The engine's legal joint-action set equals the set PS accepts (§6).
3. **Draws.** For the step k → k+1, draws and tie records match per `contract/drawkey.md` §9. Don't-care and degenerate draws are excluded.
4. **Errors.** `error_flags == 0`.

**Metrics.** For battle b, let **f(b)** be the first decision index at which state, recorded masks, draws or error flags fail, with f(b) = ∞ if the battle is clean. Only decision points k ≤ f(b) are counted.

| Metric | Definition |
|---|---|
| CBR (clean-battle rate) | #{b : f(b) = ∞} / #b |
| TMR (turn-match rate) | Σ_b #{k < f(b)} / Σ_b #{k ≤ f(b)} |
| MPR (mask-parity rate) | over k ≤ f(b) where `legal` was recorded, the fraction of equal sets |
| DPR (draw-parity rate) | over steps k ≤ f(b), the fraction of equal draw multisets |
| Error rate | fraction of battles with any error flag at k ≤ f(b) |
| Coverage | §9 |

All metrics are reported per stage and per team source.

---

## 2. Oracle (`oracle/`, Node)

**Port.** Take `research/poc/lib/*` as is:
- `ps.js`: loader, with `PS_ROOT` pointing at `oracle/ps`;
- `teams.js`, `player.js`, `context.js`, `callsite.js`, `classify.js`, `rng.js`, `project.js`, `runner.js`, `experiment.js`.

**Add:**
- `trace.js`: the writer for the format in §3;
- `coverage.js`: §9;
- `scenarios.js`: crafted states, §7;
- `bin/oracle.js`: the CLI and RPC server (02 §4.1–4.2).

**Runtime choices:**
- Battles use the **direct API**: `new Battle({formatid, seed, prng: InstrumentedPRNG, strictChoices: true})`. It is faster than the BattleStream and gives access to the battle object.
- Recording uses the **marker classifier**, which costs about 6%. The stack classifier (+25%) runs in CI and on 1% of production battles as the drift check. Any `UNMAPPED` draw or disagreement between the two classifiers fails the run.
- Concurrency is a worker-thread pool, one worker per core. Observed: 12 workers ≈ 163 battles/s on a loaded machine with the full oracle config; about 450 battles/s is estimated uncontended (RNG §8).
- The player policy uses its **own RNG, never `battle.prng`**. It handles `[Unavailable choice]` retries (hidden-info `maybeTrapped`/`maybeDisabled`) by keying projection capture on request identity (RNG §11 #6).
- Every record carries `ps_commit`.

### 2.1 Determinism (ADR-030)
- **Per-battle seeds.** For run seed `S` and battle index `i`:
  - PS seed = `"sodium," + hex(sha256("battle", S, i))[:32]`;
  - team-sampler seed = `sha256("team", S, i)`;
  - policy seed = `sha256("policy", S, i)`;
  - SP-fill seed = `sha256("spfill", canonical_sheet)`;
  - `battle_id = hex(sha256(S, i, ps_commit, oracle_git))[:16]`.
- **Worker assignment.** Worker w processes indices `i ≡ w (mod W)`. Each shard holds a fixed index range and is sorted by i, so outputs are byte-identical for any `--workers` value.
- **Records.** Every report records `S`, the full command line, `ps_commit`, `oracle_git` and `contract_version`. Gate runs use a fresh `S` never used for debugging.
- **Tests.** Engine natural-mode tests use `jax.random.PRNGKey(i)`. Hypothesis runs with `derandomize=True` in T0/T1 and a printed random seed in T2.

**Choice policies:**

| Policy | Behaviour |
|---|---|
| `random` | uniform over legal joint choices |
| `random-weighted` | biases toward non-pass, Mega and switch actions to exercise more paths |
| `heuristic` | deterministic given the policy seed, applied per slot in this order: (1) Mega whenever legal; (2) on the mon's first active turn, Fake Out if legal; (3) otherwise Protect with p = 0.25 if legal and not used last turn; (4) otherwise a spread move when both foes are active; (5) otherwise uniform over damaging moves. Overall switch probability 0.05. Task M3.8b |
| `replay` | from a given choice list |
| `engine-policy` | later: actions supplied over RPC from a JAX policy, so the visited-state distribution matches training |

---

## 3. Trace format (NDJSON, zstd-compressed, one battle per line group)

```jsonc
{"type":"header","trace_version":1,"ps_commit":"d601443…","format":"gen9championsvgc2026regmc",
 "battle_id":"<uuid>","seed":"sodium,<hex>","stage":"v1","team_source":["chaos","replay:abc"],
 "teams":["<packed p1>","<packed p2>"],              // PS packed format incl. SP as evs, gender set
 "scenario":null | {"name":"…","snapshot":<PS serialized battle JSON>},  // crafted start
 "policy":"random","oracle_git":"…"}
{"type":"decision","k":0,"turn":0,"phase":"preview",
 "requests":[<p1 request JSON>,<p2 request JSON>],
 "projection":<canonical projection BEFORE choices (§5)>,
 "legal":null | [[<canonical choice strings p1>],[...p2]],   // only in mask-check mode (§6)
 "choices":["team 3125","team 1246"],
 "draws":[{"kind":"DAMAGE_ROLL","key":{"turn":1,"step_tag":"B_MOVE","step_actor":4,"step_subject":"heatwave",
           "actor":4,"target":0,"subject":"heatwave","hit":1,"ord":0,"sub":0,"action_ord":0},
           "method":"random","args":[16],"value":7,"site":"sim/battle.ts:2409","dontcare":false,"degenerate":false}, …],
 "ties":[{"kind":"SPEED_TIE_SELECT","key":{"turn":1,"step_tag":"B_MOVE","sub":3,"ord":0},"tied":[0,5],"value":5}]}
 // keys and tie records exactly per contract/drawkey.md; mons = side*4 + brought index
 …
{"type":"final","winner":"p1","turns":9,"projection":{…},"coverage":{"sites":{…},"handlers":{…},"pairs":{…}}}
```

Notes:
- `draws` holds every draw between this decision's commit and the next decision request. The next record's `projection` is the state those draws produced.
- Raw draw records are kept: dropping don't-care draws is the **harness's** job, which makes contract changes cheap.
- Size: about 35 KB of JSON per battle uncompressed with full projections. Use zstd, and shard 10k battles per file.

---

## 4. Forcing contract (engine ↔ oracle)

**Normative definition:** `contract/drawkey.md` (protected, ADR-029; `contract_version` in every record). It defines:
- the DrawKey fields and packing;
- the `step_tag` mapping;
- the per-kind field rules and value domains;
- **tie records**, which replace all `speedSort` shuffle draws, with their ordinals;
- the don't-care, degenerate and unreachable lists;
- comparison semantics.

This section only adds harness-side procedure.

### 4.1 Oracle side
- `classify.js` and `context.js` compute the DrawKey for every draw. Mons are identified as side·4 + brought index through the preview order.
- Shuffle draws inside `speedSort` are not emitted as draws. The sort's callers (`queue.sort`, runSwitch speedOrder, `fieldEvent`, `eachEvent('Weather')`, MUL-hook `runEvent`, Magician) are instrumented to emit tie records per drawkey §4.
- `RANDOM_TARGET` draws made under the `getActionSpeed` frame are tagged `RANDOM_TARGET_NOEFFECT` (don't-care). This needs the stack or the context hook on `getActionSpeed`.
- `SECONDARY` and `SELF_EFFECT_ROLL` draws whose chance is undefined are tagged don't-care.

### 4.2 Harness side
- Drop don't-care and degenerate draws.
- Translate PS `sample` indices into the engine value domain.
- Build `ForcedOutcomes[B, KF=160]` per step from draws plus tie records.
- After replay, compare per drawkey §9.

### 4.3 Re-validation of don't-care kinds
Every week (T3), run 10k battles that leave each don't-care kind free in PS while forcing everything else. If any kind's state impact is > 0%, promote it to forced. That promotion is a protected contract change (ADR-029).

### 4.4 Failure semantics
- **A PS draw or tie record with no matching engine request, or the reverse:** `DRAW_MISMATCH`. The first unmatched key localises the bug.
- **Same key, different `args`:** `DRAW_ARGS`, the most precise localisation available.
- **A forced key the engine looked up but couldn't find:** `ERR_FORCED_MISSING`.

## 5. Canonical projection

### 5.1 PS side
`oracle/lib/project.js` (from the proof of concept) emits deterministic JSON at every decision point:
- **battle:** turn, ended, winner;
- **field:** weather and terrain with their effect states, `pseudoWeather`;
- **sides:** `pokemonLeft`, `sideConditions`, `slotConditions`, active labels, and party in **PS order**;
- **each pokemon:**
  - id `side:sheet_index`, slot, species, types, hp/maxhp, fainted. (`baseSpecies` changes on Mega in Champions, so it can't be the id; RF #32.)
  - status + `statusState`, boosts, `storedStats`, `volatiles` (effect states);
  - item + `itemState`, `lastItem`, ability + `abilityState`;
  - moves (id, pp, maxpp, disabled), `canMegaEvo`, `lastMove`, `trapped`, `weightkg`.

Effect states keep primitive fields only. Bookkeeping (`effectOrder`, `source`, `sourceSlot`, `sourceEffect`, `id`, `target`, `started`, `isSlotCondition`, `name`) is dropped.

### 5.2 Engine side
`harness/project.py` provides two functions:
- **`engine_to_json(state)`** gives the same canonical JSON, used for reports and fixtures.
- **`ps_to_expected(projection) → (expected_arrays, compare_mask)`** produces dense arrays shaped like the engine state, plus per-field validity masks. This enables **on-device diffing**: `(engine != expected) & mask` is reduced per battle to give the first mismatching decision and field group. Only failing battles are converted to JSON. This keeps the replay from bottlenecking on host conversion.

### 5.3 Effect-state mapping (`vgcjax/data/projection_map.json`)
- For each condition, map PS effect-state keys (`duration`, `time`, `startTime`, `stage`, `counter`, `layers`, `hp`, `move`, `lostFocus`, `typeIds`, …) to engine fields (`vol_state[v, i]`, `side_cond_state`, …).
- **Generation:** run a 50k-battle corpus, collect every (condition, key) pair seen, and fail datagen if any pair lacks a mapping. Add mappings as new pairs appear.
- **Mapping entries:** each entry is `(engine field, transform)`, with transform ∈ {identity, ×2 (half-HP values such as Wish `hp = maxhp/2`), count-of-×1.5 (Helping Hand multiplier), ref → side·4+b, bool}. Extra keys such as Wish `startingTurn` are mapped like any other.
- **Column assignment:** `vol_state` columns 1, 4 and 5 are assigned here per condition in first-seen order, then frozen (03 §4.3).
- **Unmapped keys** in a trace raise `PROJECTION_UNMAPPED`, a harness error rather than an engine divergence. Fix the map.

### 5.4 Known normalisations
- `moves[].disabled`: compare only at decision points, where the requests recompute it. PS `'hidden'` maps to the engine's 2, and both count as disabled.
- `trapped`: PS distinguishes `true` from `'hidden'`. Compare truthiness, plus hiddenness separately.
- Types: compare the `getTypes(true)` semantics, including `addedType`.
- `lastMove`: PS stores the move object. Compare ids.
- `weightkg` vs `weight_hg / 10`.

---

## 6. Legal-set comparison

- **Mask-check mode** (`--legal on`). At each decision the oracle enumerates PS acceptance:
  1. Take a fresh deserialized clone (TL §4.10).
  2. Submit every canonical candidate string per side to `side.choose()` (it doesn't commit).
  3. Record the accepted set.
  
  Cost is up to about 841 strings per side per decision, so run it on a sampled 10% of battles in nightly runs and 100% in PR smoke runs.
- **Canonicalisation (both sides map to the engine's 29-action encoding):**
  - a locked move → `move 1`, with no target and no mega;
  - targets for non-choosable target types → none (engine t = 0);
  - `mega` on a locked or Struggling mon → treated as no-mega (PS accepts and ignores it);
  - PS `switch N` → brought index, through the current PS party order.
- **Hidden-info exception.** The engine forbids an action that PS's request offers under `maybeTrapped` / `maybeDisabled`. This counts as a match **only if** PS rejects the choice with `[Unavailable choice]` when it's submitted.
- **Joint constraints.** At most one mega, distinct switch targets, and exactly `min(|F|, |B|)` switches in a forced switch. The proof of concept caught a real bug here (RNG §10).

---

## 7. Crafted scenarios (`oracle/lib/scenarios.js`, RPC `craft`)

- **Inputs.** A scenario is a declarative JSON object:
  - teams and leads;
  - field state (weather/terrain with duration, pseudo-weathers);
  - side conditions with durations;
  - per-mon overrides (hp, status + state, boosts, volatiles + state, item, ability, PP, `active_turns`, …);
  - the turn number.
- **Build** (`research/rng-harness-poc.md` §9): JSON-edit the serialized battle.
  - Keep `effectOrder` monotonic and refs valid.
  - Delete `activeRequest` so it gets recomputed.
  - Use APIs (`formeChange`, `setAbility`, `faint`) for derived fields.
- **Normalise** by running one `eachEvent('Update')` pass and re-serializing. PS states are not fixpoints; for example, a Sitrus holder below 50% eats its berry immediately. The engine is built from the **normalised** snapshot via `from_ps_snapshot`.
- **Scenario tests** (`tests/scenarios/**`) are pytest files declaring:
  - a scenario;
  - choices for 1–3 turns;
  - forced outcomes as a policy (e.g. `{"DAMAGE_ROLL": 0, "CRIT": false}`) or explicit keys.
  
  The test runner calls the oracle RPC (results cached by content hash in `tests/.oracle_cache/`, keyed by `ps_commit`), runs the engine and compares projections. **No hand-written expected values**: PS is always the expectation.
- **`from_ps_snapshot(json) → State`** is a required engine-side feature (M2). It supports snapshots at any decision point, including mid-turn SWITCH pauses with a queue remainder.

---

## 8. Batched forced replay (`harness/replay.py`)

1. Load B traces (B = 4k–64k).
2. Build the initial states: `init` from teams plus the preview action, or `from_ps_snapshot` for crafted starts.
3. For each decision index k (padded to `D_max`; finished battles are DONE no-ops):
   - map choices to engine actions (`interop/ps_choice.parse`, which needs the engine's `ps_order`);
   - build the `ForcedOutcomes[B, KF]`;
   - run the jitted `step` with `forced` and `draw_log` on;
   - run the on-device diff against `expected_arrays[k+1]`;
   - compare draw multisets and masks;
   - record the first failure per battle.
4. Implement the scan over k as `lax.scan` over pre-stacked per-decision inputs when memory allows, or a host loop over k with a jitted step otherwise.
5. **Output:** a per-battle result with `clean`, first-failure (k, category, field path or draw key), and counts. Write it to `reports/conformance/<run>/results.parquet` and the summary JSON.

---

## 9. Coverage (ADR-017)

### 9.1 Counters (oracle side; `oracle/lib/coverage.js`)
- **RNG sites:** hits per call site (from the classifier).
- **Handlers:** wrap `battle.runEvent`, `singleEvent`, `priorityEvent` and `fieldEvent` per battle instance (no prototype edits), and count `(effect id, event name, prefix)` whenever a callback actually runs. Also count `(effect, 'data:<field>')` for data-driven effects as they apply: secondaries applied, boosts applied, volatiles added.
- **Pairs:** `(effect A, effect B)` co-occurring in the same `runEvent` dispatch with ≥ 2 live handlers. This captures ordering-sensitive interactions.
- **Catalogue entities:** presence on a team, and used/triggered at least once.

### 9.2 Steering
`oracle gen --coverage-steer` keeps a deficit score per target (entity, handler, site, pair):
- `deficit = max(0, goal − hits)`, with goals of 50 for handlers and sites and 10 for pairs.
- Pool-D teams are sampled requiring high-deficit entities, with weight ∝ deficit.
- **Crafted templates** cover hard-to-reach states:
  - Trick Room plus speed ties;
  - low-HP berry thresholds;
  - Perish Song endgames;
  - simultaneous-KO endgames;
  - Instruct / After You / Quash orderings;
  - Ally Switch + Follow Me;
  - Revival Blessing;
  - Illusion;
  - Transform;
  - Emergency Exit at hazard entry, including during a start-of-turn double switch (queued-switch exclusion);
  - Round partner left in the queue after an Eject Button / Emergency Exit pause (RF #15);
  - Revival Blessing re-entry before and after the ally has moved (RF #16);
  - simultaneous last-mon KOs from weather chip under speed ties, and from Explosion/recoil (faint-queue order, RF #11, #24);
  - Instruct on Counter / Mirror Coat / Chilly Reception users (RF #17);
  - equal-speed Lightning Rod holders (priorityEvent order, RF #9b);
  - Commander-like cases, if they ever become legal.

  Templates live in `oracle/lib/scenarios.js`. Each targets a list of handler keys.

### 9.3 Coverage gates
- **Handler coverage.** Every BESPOKE handler and ROW entity in the stage has ≥ 50 **fuzz hits**. A fuzz hit is a distinct gate-corpus battle in which the entity's callback ran at least once (handler entities) or its data effect applied (data entities).
- **RNG sites.** Every RNG site with `reachable ∈ {yes, cond}` in `rng_callsites.json` has ≥ 50 hits.
- **Pairs.** Every pair in `docs/coverage/pairs_<stage>.json` has ≥ 10 hits. The list contains:
  - co-dispatch pairs observed at least once in the M3.8 1M-battle coverage run;
  - census `mentions` confirmed reachable by a probe.
  
  Raw census mentions are **not** the list; many are spurious.
- **Waivers.** A target can be waived only in `docs/coverage/waivers.json` (protected, ADR-029), with a probe under `oracle/tools/probes/` and an ADR. Every gate report lists all waivers.
- **Open divergence.** A bucket with ≥ 1 battle in the latest gate run. Per-entity 0 open divergences and corpus CBR must **both** hold. Buckets whose signature names no entity (SYSTEM) count only against CBR.

---

## 10. Triage and shrinking

### 10.1 Categories (the first failing check decides)
| Category | Meaning | Typical owner |
|---|---|---|
| `DRAW_ARGS` | same key, different distribution arguments | mechanic numeric bug |
| `DRAW_MISMATCH` | unmatched draw key | control flow or gating bug, or key-contract bug |
| `MASK` | legal sets differ | requests / disabled / trapped logic |
| `STATE:<field-group>` | projection mismatch (hp, status, boosts, volatiles.X, side.X, field.X, ps_order, pp, item, ability, species) | mechanic bug |
| `ERROR:<flag>` | an engine error flag | overflow, unsupported effect |
| `HARNESS` | projection unmapped, trace parse failure, oracle inconsistency | harness bug |

**Signature** = category + field path or draw kind + the set of catalogue entities involved (from coverage counters on the failing step). Battles are bucketed by signature, and the report ranks buckets by count.

### 10.2 Shrinking (`harness/shrink.py`)
For a failing battle with first failure at decision k:
1. **Turn isolation.** Use `snapshot` at decision k−1, then `from_snapshot` with choices_k and the forced draws. Build the engine from the snapshot and confirm the failure reproduces with the same signature.
2. **Reduction loop.** Greedy: each candidate is re-checked in both PS (`from_snapshot` + forced draws; keys that no longer exist are dropped) and the engine, and is kept if the signature persists.
   - replace unrelated moves with a neutral DATA_ONLY move such as `splash`;
   - remove items and set abilities to a neutral one: an entry from `tools/datagen/core_inert.json` that the mon can legally have, otherwise keep the original ability;
   - remove field and side conditions;
   - faint or remove bench mons;
   - set HP to max.
3. **Emit a fixture:** `tests/fixtures/conformance/<category>/<signature-hash>.json` containing the snapshot, choices, forced draws, the expected PS projection after the step, the signature and the `ps_commit`. Pytest collects the fixtures automatically, and a fixture that fails blocks merges.

---

### 10.3 Versioning and re-pin handling (ADR-030)
- **Record fields.** Every trace, fixture and cache entry carries `ps_commit`, `oracle_git` and `contract_version`.
- **Oracle cache key** = `sha256(ps_commit, oracle_git, contract_version, canonical_request)`. `tests/.oracle_cache/` is not committed; T1 rebuilds it.
- **Fixture provenance.** Fixtures keep `source_battle_id` and `source_run_seed`.
- **On re-pin, `harness.cli refresh-fixtures`** re-runs each fixture's snapshot, choices and forced draws on the new PS:
  - **PS projection unchanged:** update `ps_commit`;
  - **projection changed:** move the fixture to `tests/fixtures/conformance/_repin/<old_commit>/` and open a `PS_CHANGE` bucket citing the PS commit range;
  - **snapshot fails to deserialize:** regenerate from the source trace, or delete with a line in `docs/oracle-changelog.md`.
- **Fixtures are never deleted by hand.**

## 11. Distribution tests (natural-mode RNG)
For every DrawKind, engine natural draws must follow PS's distribution:
- **Analytic checks:** sample 10⁵ draws through crafted scenarios and run chi-square tests against PS's formula (e.g. `randomChance(1,8)`, `sample([2,3,3])`, the multihit table, `random(16)`), with p ≥ 0.001 after Bonferroni correction.
- **Two-sample checks**, with free RNG on both sides over 10k runs each, for chains: Protect consecutive-use success, sleep turn counts, freeze thaw counts, confusion. Use the same tolerance.
- Run them weekly and on any change to `core/rng.py`.

---

## 12. Gates and CI tiers

### 12.1 Stage gates (also in 10)
| Gate | CBR (usage-sampled teams, random policy) | CBR (coverage-steered + crafted) | MPR | DPR | Coverage | Error rate |
|---|---|---|---|---|---|---|
| G-core (M2) | ≥ 99.5% on 50k `core`-stage battles (05 §1.1) | n/a | 100% | 100% | core RNG kinds (ACCURACY, CRIT, DAMAGE_ROLL, MULTIHIT_COUNT, RANDOM_TARGET, SECONDARY, and tie kinds SELECT / SWITCHIN / FIELD_EVENT) ≥ 50 hits each | 0 |
| G-v0 (M4) | ≥ 99.0% | ≥ 98% | ≥ 99.99% | ≥ 99.9% | §9.3 for v0 | 0 |
| G-v1 (M6) | ≥ 99.5% | ≥ 99% | ≥ 99.99% | ≥ 99.9% | §9.3 for v1 | 0 |
| G-v2 (M7) | ≥ 99.9% | ≥ 99.5% | 100% on 1M decisions | ≥ 99.99% | §9.3 for v2 | 0 |

**Gate run composition** (fresh seed `S`; configured in the protected `harness/gates.yaml`):
- **(a)** Pool-B usage-sampled battles under the `random` policy: 200k for G-v0, 1M for G-v1/G-v2. This gives the first CBR column.
- **(b)** Coverage-targeted battles (70% pool D with `--require`, 30% crafted templates): 100k for G-v0, 300k for G-v1/G-v2. This gives the second CBR column.
- **(c)** The frozen real-sheet corpus (`real_sheets_mc_v1`) restricted to the stage. Each team plays 10 battles against uniformly drawn other in-stage real sheets under `random`, and 10 under `heuristic`. It must meet column 1's threshold.
- **Mask checks:** `--legal on` for 10% of (a) and (b), and for 100% of (c). G-core runs `--legal on` for 100% of its corpus.
- **DPR and error rate** are computed over all battles.
- **Compile budgets** are measured on the M4 Max with a cold persistent cache.

### 12.2 CI tiers
| Tier | When | Content | Budget |
|---|---|---|---|
| T0 | pre-commit | lint (no Python control flow on tracers; every handler has `ps=`), `tests/unit`, and up to 20 touched-family scenarios run under `jax.disable_jit()` | < 2 min |
| T1 | before every merge to main | all unit and golden tests; **one compile per stage**, then all scenarios and fixtures replayed as one vmapped batch (09 §3); the canary test (ADR-029); a 2k-battle fuzz smoke with `--legal on`; compile-time budget check | < 20 min CPU |
| T2 | nightly | 200k battles (50% usage-sampled, 30% coverage-steered, 20% crafted), mask check on 10%, coverage report; writes `reports/catalogue_status.json`, which the agent folds into `docs/catalogue.csv` at session start in a `data(catalogue): …` commit | < 6 h on the dev box |
| T3 | weekly | 1M battles, the real-sheet corpus, distribution tests, don't-care re-validation (§4.3), perf benchmarks (09) | < 24 h |
| T4 | on re-pin | T3 plus the call-site drift check, datagen diff and census diff (06 §6) | as needed |

---

## 13. Real-log replays (supplementary, not gating)
- Public replays lack Stat Points and RNG. Even PS reproduces only about 33% of real turns from reconstructed teams (vgc-engine audit: https://github.com/colefoster/vgc-engine/blob/main/docs/accuracy/2026-09-accuracy-proof.md).
- Use them only as a **smoke signal**: re-simulate with sampled spreads and check that engine and PS agree with each other under forcing. Their main value is **team sheets** (pool A) and **policy realism**.

## 14. Sim-to-real check (M5+)
Run engine policy vs policy matches and the same matches in PS through `interop` (actions mapped to choice strings, chance trace forced into PS). The engine and PS outcomes must be identical under forcing. This doubles as replay export (08 §9).
