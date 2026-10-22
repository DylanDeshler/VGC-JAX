# 01 — Decision record (ADRs)

Each decision has an ID, the decision itself, the rationale, and the evidence it rests on. Every one is **binding** for the implementer. To change one, the implementer adds a new ADR that supersedes it (§"Changing a decision" at the end) instead of quietly deviating.

Evidence paths are relative to the project root (`/Users/dylan.d/Documents/research/pokemon/`).

---

## A. Scope and target

### ADR-001 — Target format: Champions VGC Reg M-C, doubles, open team sheets
- **Decision.**
  - The engine simulates Pokémon Showdown's `[Gen 9 Champions] VGC 2026 Reg M-C` (`gen9championsvgc2026regmc`, mod `champions`, `gameType: doubles`).
  - Rules: bring 6, pick 4, level 50, Item Clause, Species Clause, Stat Points (SP), Mega Evolution, no Tera.
  - Open team sheets (OTS) are assumed. The Bo3 variant `gen9championsvgc2026regmcbo3` is the deployment and evaluation format.
- **Rationale.**
  - This is the user's goal: playing Pokémon Champions.
  - It is where the players and data are: about 1.63M Reg M-C Bo1 and about 223k Bo3 ladder games in 2026-09.
  - OTS appears in 100% of Bo3 games but only about 15% of Bo1 games.
- **Evidence.** `research/champions-mechanics-delta.md` §9; `research/rl-training-design.md` §6.1, §9.1.

### ADR-002 — Correctness oracle: Pokémon Showdown, pinned
- **Decision.**
  - "Faithful" means **matching Pokémon Showdown's (PS) behaviour at a pinned commit**. The initial pin is `d601443456c795f77c6e4f2b5b459167f2b10a3c` (2026-10-21).
  - Where PS and the real game are known to disagree, follow PS by default. A feature flag can select the other behaviour (ADR-019).
  - PS is vendored at a durable path inside the repository (`oracle/ps`, a git submodule). It is never kept in `/tmp`, because `/tmp` was wiped twice during research.
- **Rationale.**
  - There is no programmatic access to the Champions game.
  - PS is the reference implementation used by the whole community and by every prior bot.
  - Policies are evaluated on PS anyway (ADR-001).
  - PS's Champions mod tracks in-game research closely: it matched every listed numeric change (`research/champions-mechanics-delta.md` §11.1).
- **Consequence.** PS bugs are faithfully reproduced. The oracle also moves: about 16 mechanics commits in 3 months. A re-pin procedure is mandatory (ADR-021).

### ADR-003 — Content scope: the full legal pool is the end state, reached in stages v0 → v1 → v2
- **Decision.**
  - The end state supports every legal Reg M-C entity: 293 set-level species, 82 Mega formes, 510 moves, 215 abilities and 166 items. This is "v2".
  - The engine is built in three cumulative content stages:
    - **v0**: 220 CUSTOM entities. Covers about 90% of 1760 Bo3 teams and 82% of real team sheets.
    - **v1**: 376 CUSTOM entities. Covers 99.7% of 1760 Bo3, 95.7% of all-ladder Bo1, and 99.4% of real sheets.
    - **v2**: 483 CUSTOM entities. Covers 100%.
  - Every species is in every stage, because species are pure data.
- **Rationale.**
  - Usage-mass dexes badly overstate coverage. The "99% usage" dex fully expresses only 12.5% of real teams, because Mega Stone holders play with their *base* ability.
  - v1 already costs about 80% of v2's handler code.
  - Playing arbitrary ladder opponents needs near-complete coverage.
- **Evidence.** `research/dex-scope.md` §0, §4.
- **Consequence.**
  - Teams used for training and testing are restricted to the *currently supported* stage by an offline validator.
  - The engine raises an error flag if an unsupported effect ever executes.

### ADR-004 — Non-goals for this project
The following are out of scope:
- closed team sheets;
- other regulations, except through regenerated data tables (ADR-020);
- singles;
- Tera, Dynamax and Z-moves;
- timers and the in-game 20-minute clock;
- the Bo3 series layer (the engine plays single games; a series is a trainer-level concern);
- live play inside the actual Champions game (prohibited by the Pokémon Terms of Use);
- team building.

The observation design keeps reveal bits, so a closed-sheet mode stays cheap to add later.

---

## B. Engine architecture

### ADR-005 — Pure functional single-battle engine, vmapped from outside
- **Decision.** The engine is a set of pure functions over one battle's state pytree:
  - `init`, `step`, `observe`, `legal_masks` and the helpers in 08.
  - Batching is done by the caller with `jax.vmap`. RNG keys are explicit arguments and never live in the state (pgx v2 convention).
- **Rationale.**
  - pgx, minimon, gymnax and jumanji all do this.
  - It enables nested vmap for search (battles × worlds × rollouts).
  - Cloning a state is free.
- **Evidence.** `research/jax-patterns.md` §7; `research/rl-training-design.md` §7.1.

### ADR-006 — Three-layer configuration: static Config, traced Dex, per-battle State
- **Decision.**
  - **`EngineConfig`** is a frozen, hashable dataclass passed as a static argument to `jit`. It holds the ruleset constants, turn cap, trace/forced-outcome modes, and the set of handlers compiled in. Python `if`s on it run at trace time.
  - **`Dex`** is a pytree of dense integer tables. It is traced, so it can be swapped without recompiling as long as shapes match.
  - **`State`** is the per-battle pytree.
- **Rationale.**
  - Compile time is the main development-speed risk. Config-gated handler subsets let developers compile only what their test teams need.
  - This is the minimon pattern.
- **Evidence.** `research/jax-patterns.md` §6, §7.

### ADR-007 — Mirror PS's state representation and control flow wherever practical
- **Decision.**
  - Engine state uses **PS's own counters and semantics**: condition `duration` exactly as PS decrements it, sleep `startTime`/`time`, toxic `stage`, stall `counter`, `activeMoveActions`, the cached `pokemon.speed`, PS party order (`side.pokemon` order), and per-condition creation sequence numbers (PS `effectOrder`).
  - Engine control flow mirrors PS's queue/`runAction` semantics: order classes, re-sort before every move, batched vs non-batched switch-ins, pauses for mid-turn switch decisions.
- **Rationale.**
  - Differential testing compares projections field by field. Mirroring semantics removes a whole class of translation bugs.
  - It makes every engine function traceable to a PS function.
- **Exception.** PS behaviours that are provably state-irrelevant may be canonicalised: tie shuffles in handler sorts, nominal random targets for spread moves, and the unconditional self-drop roll. The list is in `contract/drawkey.md` §6.

### ADR-008 — Step = one simultaneous decision point; four phases
- **Decision.**
  - `step(state, actions[2], key)` applies both players' choices and resolves until the next point where some player has a choice to make, or the game ends.
  - Phases are `PREVIEW`, `MOVE`, `SWITCH` and `DONE`.
    - `SWITCH` covers end-of-turn replacement, mid-turn pivots (U-turn, Parting Shot, Eject Button, Emergency Exit, …) and the Revival Blessing variant.
  - A player with nothing to decide gets a pass-only mask and `has_decision = False`.
- **Rationale.** This mirrors PS's request model (`teampreview` / `move` / `switch` / `wait`) and keeps simultaneity explicit.
- **Evidence.** `research/ps-turn-loop-spec.md` §3, §11.1; `research/rl-training-design.md` §7.3.

### ADR-009 — Split turn execution into a "light action" block and a "move" block
- **Decision.** Within a step, executed actions are split into three blocks:
  - **Block A:** a bounded loop over non-move actions in PS order: beforeTurn, beforeTurnMove, revival, switch, instaswitch, runSwitch, megaEvo and priorityChargeMove. **Bound 10**, plus an overflow flag. Each action is followed by PS's runAction tail.
  - **Block B:** a bounded loop over move actions. **Bound 8** = 4 primary + 2 Instruct + 2 runSwitch entries that PS sorts behind prioritized moves after a pause (04 §2.1). Plus an overflow flag. Before each move selection it re-keys every queued move's speed and priority, then selects by (prepend, order, priority, speed, tiebreak).
  - **Block C:** residual and end-of-turn, run at most once per step.
  - All blocks are masked no-ops when not reached, for example after a pause.
- **Rationale.**
  - Under vmap every branch runs. A single 18-iteration loop that can run any action kind would execute the heavy move pipeline 18 times per step; this layout executes it at most 8 times.
  - Order-<200 entries precede moves, and PS re-sorts only before moves (TL §5.7).
  - The two exceptions (a runSwitch behind a prioritized move after a pause; Revival Blessing re-entry) are handled by Block B's light branch, the `prepend` flag and Block C (04 §2.1, §10.4). Both were verified in PS (RF #15, #16).
- **Bounds evidence.** `research/ps-turn-loop-spec.md` §11.4 and `research/review-fidelity.md` #1, #15, #28.

### ADR-010 — Effect dispatch: tables first, small per-hook switches second, barriers between hooks
- **Decision.**
  1. Every PS event used in-format maps to one of about 32 **hook points** (`research/ps-move-damage-event-spec.md` §7.2). Each hook has one fold kind: MUL (a 4096 chain, then one `modify`), GATE (first falsy wins) or REPL (replacement).
  2. **Multiplier and constant handlers are data rows** `(hook, effect, prefix, predicate_id, params, order, priority, subOrder)`, evaluated by a few generic predicate templates. Gather and mask; no switch.
  3. Remaining **bespoke handlers** go in one `lax.switch` *per hook*. The switch is indexed by a static remap `local_idx[hook, effect]`, and holds only the effects that implement that hook.
  4. Put a `lax.optimization_barrier` on the hook context after every hook dispatch.
  5. The 3-target axis is a vectorised (`vmap`) axis inside the move body, never an unrolled loop around a switch.
- **Rationale.** Measured on CPU (`research/jax-patterns.md` §1):
  - a batched switch costs time linear in total branch work;
  - tables flatten that cost: 29 ns vs 105 ns at N = 512, and compile is 5× faster;
  - chained switches without barriers compile superlinearly: 24.5 s → 1.7 s with barriers, and run 795 → 93 ns/elem;
  - unrolled loops around switches explode compile time (>90 s).

### ADR-011 — Integer-exact arithmetic
- **Decision.**
  - All mechanics math is int32, with PS `trunc`, `modify` (4096 fixed point, round half down), `chainModify` and `finalModify` reproduced bit-exactly.
  - Storage is int8/int16. There are no floats anywhere in mechanics.
- **Rationale.**
  - Damage must match PS for all 16 rolls; this was verified in `research/ps-move-damage-event-spec.md` §3.7.
  - Native int8/int16 arithmetic is slower and overflows (`research/jax-patterns.md` TL;DR 6).

### ADR-012 — RNG: counter-based, semantically keyed draws; forced-outcome and draw-log modes
- **Decision.**
  - Each `step` receives a key. The engine derives two uint32 words from `fold_in(key, step_counter)`.
  - Every random decision is `hash32(words, pack(DrawKey))`; the DrawKey is defined in `contract/drawkey.md`.
  - Draws are never positional in a stream, so masked branches can't shift later draws.
  - Two compile-time modes are added (ADR-006):
    - **forced-outcome mode:** a fixed-size table of outcomes keyed exactly like PS's classified draws overrides the natural draw;
    - **draw-log mode:** the engine records every draw request (kind, coordinates, distribution arguments, value used) into a fixed buffer.
- **Rationale.**
  - Determinism, vmap safety and replay from (teams, key, actions).
  - Common random numbers across actions, which helps search.
  - Differential testing needs context-keyed forcing; PS's sequential stream can't be reproduced under all-branches execution.
- **Normative key.** The authoritative key, kinds and value domains live in `contract/drawkey.md`. Every other document references it rather than restating it.
- **Evidence.** The proof of concept reproduced 300/300 battle state trajectories with the engine-implementable key (`research/rng-harness-poc.md` §6.2). Threefry with many `fold_in`s is slow, and `rbg` breaks per-battle keys under vmap (`research/jax-patterns.md` TL;DR 7).

### ADR-013 — State layout: per-brought-mon arrays [2,4], sheet data [2,6], dense per-kind condition storage
- **Decision.**
  - Battle-dynamic per-mon arrays have shape `[2 sides, 4 brought]`.
  - Team-sheet data, needed for observations, has shape `[2, 6]`.
  - Volatiles live **on the mon**: `[2,4,NV,FIELDS]`, dense by volatile kind. Side, slot and field conditions are also dense by kind.
  - `active[2,2]` maps slots to brought indices.
  - `ps_order[2,4]` mirrors PS's `side.pokemon` order, which is needed for `switch N` mapping and Illusion.
  - The exact schema is in 03.
- **Rationale.**
  - Mechanics only ever touch the 4 brought mons.
  - Dense per-kind storage avoids slot-allocation logic. Per-mon volatiles make Ally Switch a pure index swap.
  - Leaf count only costs at the Python boundary: about 1 µs per leaf per call, and it is free inside `jit`/`scan` (`research/jax-patterns.md` §3C). Budget ≤ 150 leaves. Pack into dtype-grouped arrays only if profiling shows a need (M9).

### ADR-014 — No exceptions inside jit: error bitfield
- **Decision.** A `uint32 error_flags` in State records:
  - illegal action substituted;
  - unsupported effect executed;
  - block overflow (A/B/queue/hook);
  - invariant violation;
  - forced-outcome key not found;
  - draw-log overflow.
  
  Debug builds additionally use `checkify`. Illegal actions are replaced by the first legal action, never by a terminating penalty.
- **Rationale.** pgx's "illegal = loss" silently shapes the policy if masks are buggy (`research/rl-training-design.md` §7.4).

---

## C. Validation

### ADR-015 — Fidelity contract = state-trajectory equivalence under forced outcomes, plus legal-set equality
- **Decision.** A battle is **clean** when all of the following hold:
  1. Replaying PS's choices with PS's classified outcomes forced gives an engine **canonical projection** equal to PS's at **every decision point**, and at the end.
  2. The engine's legal joint-action set equals the set PS *accepts* at every decision point, modulo the documented canonicalisations and hidden-info `maybe*` cases.
  3. Every PS draw not on the don't-care list has a matching engine draw request with equal distribution arguments, and vice versa.
  4. The engine raised no error flags.
- **Rationale.**
  - Win-rate equivalence (PokeJAX) is far too weak; a policy can exploit bugs that don't move aggregate win rates.
  - Per-turn state diffs localise bugs to the turn and field.
  - The draw-argument check localises them to the mechanic.
- **Evidence.** `research/rng-harness-poc.md` §6, §10, §12; `research/rl-training-design.md` §3.2.

### ADR-016 — The conformance harness is built before the long tail, and drives all mechanics work
- **Decision.**
  - Milestone order puts the oracle harness (trace generation, forced replay, projection diff, shrinking, coverage) *before* bulk content work.
  - Content work then proceeds as a loop: generate failing battles, shrink, fix, add a fixture.
- **Rationale.**
  - vgc-engine's audit shows mechanics bugs, not RNG, dominate divergence: 94% of first divergences, and only 16.5% of battles clean.
  - The harness turns content implementation into a closed loop an agent can run unattended.
- **Evidence.** `jax-vgc-engine-scope.md` §0, §6a; `research/rng-harness-poc.md` §12.

### ADR-017 — Coverage-guided generation is mandatory
- **Decision.**
  - Battle generation is steered toward unexercised RNG call sites, handler functions and handler pairs, using counts gathered from PS (`runEvent` hooks plus the call-site inventory).
  - Crafted mid-battle states built through `State.deserializeBattle` cover rare situations.
- **Rationale.** 300 usage-weighted random battles left 20–26 reachable RNG sites unexercised (`research/rng-harness-poc.md` §2.2).

---

## D. Interfaces and data

### ADR-018 — Action encoding
- **Decision.**
  - **Preview:** the engine accepts an index into the static table of all 360 ordered 4-of-6 tuples. Order matters: the first two are leads in slots a/b, and bench order matters for Illusion and PS indexing. The env wrapper can restrict the policy to the 90 canonical (bring-set × lead-pair) options.
  - **MOVE and SWITCH, per active slot, 29 actions:**
    - `a = 6·m + 2·t + mega` (a < 24), with m ∈ 0..3, t ∈ {0 → foe slot a (+1), 1 → foe slot b (+2), 2 → ally}, mega ∈ {0,1}. For `adjacentAllyOrSelf`, t = 0 is self and t = 2 is the ally. Non-choosable target types use t = 0 only;
    - locked moves and Struggle use only `a = 0`;
    - `24–27`: switch to (or, under Revival Blessing, revive) the mon at **brought index** 0..3, which is identity-stable;
    - `28`: pass.
  - 04 §4.3 is the single source of truth for this encoding.
  - **Joint action:** 29 × 29 = 841, with an exact joint mask.
- **Rationale.**
  - It covers every legal PS choice in-format (`research/ps-turn-loop-spec.md` §4.9).
  - Switch targets are pointer-style and identity-stable.
  - An exact joint mask is needed because slot coordination matters: in PokaiTrainer, a joint shortlist covered the realised joint action only 64% of the time, against 97% for per-slot marginals.
- **Evidence.** `research/rl-training-design.md` §7.4.

### ADR-019 — Known PS-vs-game disputes are feature-flagged, defaulting to PS
Fire Mane (×1.5 Atk vs a base-power modifier, PS issue #12377), simultaneous Intimidate resolution and Encore target re-roll each get an `EngineConfig` flag. The default is PS behaviour, and conformance always runs with defaults.

### ADR-020 — All content data is generated from PS; nothing hand-typed
- **Decision.** Species, moves, items, abilities, conditions, the type chart, learnsets (legality only), handler metadata (order/priority/subOrder per event), multiplier rows and the PP/stat formulas' inputs are all exported from the pinned PS by `tools/datagen`. Hand-written code exists only for bespoke handler *logic*.
- **Rationale.** Regulations change every ~12 weeks, and PS changes mechanics weekly.

### ADR-021 — PS re-pin procedure
- **Decision.** Re-pin PS:
  - **after G-v0** (see ADR-028 for the freeze before that), monthly;
  - on every regulation switch, run as its own mini-milestone (ADR-028).
- **Procedure.**
  1. Bump the submodule.
  2. Run the call-site drift check (`scan_callsites --check`).
  3. Regenerate data.
  4. Diff the handler census.
  5. Run the full conformance suite.
  6. Triage new divergences against the PS commit log for `data/mods/champions*` and `sim/`.
  7. Update `docs/oracle-changelog.md`.
- **Merge rule.** A re-pin lands only when conformance gates are back at their previous level.

### ADR-022 — Observations reproduce PS's information exactly
- **Decision.**
  - `observe(state, player)` reads only public state, the player's private state and reveal bits.
  - Opponent HP uses PS's Champions rule `floor(100·hp/maxhp) || 1`, plus the 20%/50% colour bits.
  - Nature is public under Champions OTS. The hidden information is the opponent's SP spread, the unrevealed brought mons, exact HP and hidden counters.
  - The engine **does not** bake damage calculations from the true state into observations. A model-callable `damage_calc` runs on hypothesised spreads instead.
- **Evidence.** `research/rl-training-design.md` §6.1–6.4.

### ADR-023 — Rewards and truncation
- **Decision.**
  - Reward is a zero-sum terminal outcome only. Shaping *features* are exposed separately.
  - `max_turns` is configurable:
    - training default 40, with a tiebreak (remaining mons, then HP fraction, then draw) and `truncated = True`;
    - conformance mode uses 1000, PS's hard tie.

### ADR-024 — Tech stack
- **Engine and harness:** Python 3.12 (`.python-version`), pinned exactly:
  - `jax==0.12.0.1`, `jaxlib==0.12.0.1`, `flax==0.12.10`, `chex==0.1.92`;
  - NumPy, pytest, hypothesis, orjson, zstandard, pyarrow;
  - a `gpu` extra pinning `jax[cuda12]==0.12.0.1`;
  - `uv` for environments, with its version pinned in `README`.
- **Oracle and data export:** Node 26.8.1 (`.nvmrc`; the version used in research), plain CommonJS JavaScript, reusing the proof-of-concept modules in `research/poc/lib/`. It talks to Python over NDJSON files and a line-delimited JSON-RPC subprocess.
- **Changing pins:** only by ADR, with a T3 run.
- **Not used for the engine:** Rust or C. The engine must stay inside the jitted rollout, as in PureJaxRL.

### ADR-025 — Licensing hygiene
- **Allowed sources:**
  - PS (MIT): porting its logic and generating data from it is allowed; keep its licence notice in `THIRD_PARTY_NOTICES`.
  - pkmn/engine, minimon, pgx, gymnax, jumanji: permissively licensed and fine as references.
- **Forbidden source:** Foul Play is GPL-3.0. Do not copy code from it.
- **Data:** keep usage-stats data in `data/raw/` (git-ignored) with its source URL and licence in `SOURCES.md`.
- **Supply chain:** run `npm ci` only against a committed lockfile, and record `npm ls --all` per PS pin.
- **Datasets:** check each dataset's licence (e.g. VGC-Bench HF logs) before committing anything derived from it.
- **Personal data:** strip usernames and ratings from committed team banks; keep only a rating bucket.
- **Never commit** downloaded web pages (Bulbapedia, Serebii, official sites), virtualenvs or third-party clones.

### ADR-028 — Target freeze until G-v0
- **Decision.**
  - Conformance targets `gen9championsvgc2026regmc` at PS `d601443` until G-v0 passes.
  - **No scheduled re-pins happen during M0–M4.** An unscheduled re-pin is allowed only when a PS fix removes a bucket we would otherwise have to emulate, by ADR.
- **When the next regulation is published** (Reg M-C ends 2026-12-02):
  - run datagen for its format id on a branch and record the delta in `docs/oracle-changelog.md`;
  - don't switch targets.
- **After G-v0:**
  - monthly re-pins resume (ADR-021);
  - a regulation switch is its own mini-milestone that re-runs G-v0 on the new format;
  - the M-C engine stays available through its `DATA_VERSION`.
- **Parameterisation.** `export_dex.js`, `dex_stages.py` and team builders take `--format`. Nothing hard-codes `regmc`.
- **Rationale.** Hard gates and a moving oracle conflict. G-v0 is about 3–4 months out, while the oracle and regulation move weekly or quarterly.

### ADR-029 — Protected conformance contract
- **Protected files:**
  - `harness/contract/{dontcare.json, degenerate.json, normalisations.json, canonicalisation.json, projection_fields.json}`;
  - `docs/design/contract/drawkey.md`;
  - `harness/gates.yaml` (thresholds, corpus composition and sizes, coverage goals);
  - `docs/coverage/waivers.json`.
- **Changes** need an ADR with the evidence experiment (e.g. the 07 §4.3 0%-impact run) **and human approval**.
- **Hash check.** `harness.gates` prints the sha256 of each protected file in every gate report. A pass counts only if the hashes match `docs/decisions/contract-hashes.md`, which only the human edits.
- **Canary test (T1).** `tests/harness/test_canary.py` runs 200 core battles with the test-only flag `cfg.flags.canary_crit_2048` (crit ×1.25 instead of ×1.5). It asserts that the harness reports ≥ 1 `DRAW_ARGS` or `STATE` failure, proving the harness can see bugs.
- **Fixtures** are removed only by `harness.cli refresh-fixtures` (07 §10.3).
- **Rationale.** Under gate pressure, widening the don't-care list or the normalisations is the cheapest way to "pass". This makes that impossible to do silently.

### ADR-030 — Determinism and versioning
- **Seeding.** Every random choice in the oracle, team sampling, tests and fuzzing derives from a recorded run seed (07 §2.1). Outputs are byte-identical regardless of worker count.
- **Versioning.** Every trace, fixture and cache entry carries `ps_commit`, `oracle_git` (the hash of `oracle/lib/**`) and `contract_version`. The latter is bumped on any change to the protected contract files.
- **Freshness.** Gates use only traces with the current `contract_version`.
- **Re-pin handling.** Old fixtures are refreshed per 07 §10.3.

---

## E. Process

### ADR-026 — Definition of done for any mechanic
A mechanic or entity is done only when all of the following hold:
1. Its code cites the PS `file:line` it mirrors.
2. It has at least one deterministic scenario test, a crafted PS state with forced outcomes, that passes.
3. The fuzz corpus exercises it at least N times (default 50) with zero open divergences.
4. Its entry in the mechanics catalogue (`docs/catalogue.csv`) is marked `done` with the PS commit.

### ADR-027 — Progress, gates and escalation
- **Progress.** The implementer keeps `PROGRESS.md` (milestone status, gate metrics, open divergence classes) and `docs/decisions/` (new ADRs).
- **Gates.** Milestone gates (10) are hard: the next milestone's *content* work does not start until the previous gate passes. Infrastructure work may run ahead.
- **Gate checkpoint.** When a gate passes, send the gate report to the human. Until it is acknowledged, continue only infrastructure lanes.
- **Escalate to the human only when:**
  - a decision here proves infeasible;
  - PS behaviour looks like a PS bug that matters to training;
  - the work needs compute, accounts or network access the agent lacks.

---

## Changing a decision
Add `docs/decisions/ADR-1xx-<slug>.md` containing:
- the ADR it supersedes;
- what changed;
- the evidence (benchmarks, PS source, conformance numbers);
- the migration impact.

Update the affected design docs in the same commit.
