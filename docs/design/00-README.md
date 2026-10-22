# Design package: a faithful JAX engine for Pokémon Champions VGC (Reg M-C), for fast RL policy training

**Status:** complete design, ready to hand off for implementation. Date 2026-10-21.
**Audience:** the implementing agent (or engineer). Start with `11-agent-operating-manual.md` §1 for the reading order.

---

## 1. What is being built

A pure-JAX, `jit`/`vmap`-able battle engine for **Pokémon Showdown's `[Gen 9 Champions] VGC 2026 Reg M-C`** format, with these properties:
- doubles, bring 6 pick 4, level 50, Stat Points, Mega Evolution, no Tera, open team sheets;
- **state-trajectory-equivalent to Pokémon Showdown** (PS) under forced random outcomes. PS is the pinned oracle;
- **fast enough** to train Jaxcalibur-scale self-play policies: ≥ 10⁵ env-steps/s per H100 target;
- an RL-ready API: simultaneous actions, exact joint masks, information-state observations, auto-reset with team banks, and search and deployment hooks.

**How it is built.**
- A **differential-testing harness** against PS comes first. It runs millions of battles per day on CPU.
- Then content is added in three stages, after a narrow `core` stage proves the state machine: **v0** (core meta, 220 CUSTOM entities including conditions), **v1** (99.4% of real teams), **v2** (full legal pool).
- The harness drives everything: generate battles, find divergences, shrink them to fixtures, fix them, repeat.

## 2. Key facts the design rests on (all verified during research)

| Fact | Value | Source |
|---|---|---|
| Official VGC game | Pokémon Champions (released 2026-04-08). Reg M-C runs 2026-09-09 → 2026-12-02 | `research/champions-mechanics-delta.md` §11.3 |
| PS pin | `d601443` (2026-10-21). The Champions mod is "Gen 9 + thin delta"; 20 numeric/behavioural deltas | CH §0 |
| No Tera; Mega instead | `canTerastallize → null`. 82 Mega formes from 81 stones; one Mega per side | CH §1.7, §8 |
| Stats | `base + SP + 20` (HP `+75`), nature ×1.1/×0.9; SP ≤ 32 per stat, ≤ 66 total, IV 31 | CH §1.3, §9.4 |
| Not legal in Reg M-C | Commander, Paradox/Booster Energy, Ruin abilities, Dancer, Covert Cloak, Clear Amulet, Safety Goggles, Assault Vest, Choice Band/Specs, Spore, Eject Pack, Weakness Policy | `research/dex-scope.md` §3, TL §1.3 |
| Legal pool | 293 set-level species (263 battle-distinct), 82 Megas, 510 moves, 215 abilities, 166 items, 102 conditions | dex-scope §3 |
| Handler cost | v2: 483 CUSTOM entities, 747 distinct handler templates on 135 hooks. v1 is ~80% of that | dex-scope §0 |
| Usage-mass dexes mislead | the 99%-usage dex expresses only 12.5% of real teams, because Mega holders keep their *base* abilities | dex-scope §2 |
| Oracle harness works | instrumented PS is byte-identical; 88 RNG call sites classified; context-keyed forcing reproduced 300/300 trajectories; serialize round trip exact; masks verified | `research/rng-harness-poc.md` §0 |
| Oracle throughput | ~40–54 battles/s/thread; 12 workers ≈ 160–190 battles/s on a loaded machine (≈ 450 estimated uncontended) | RNG §8 |
| Damage formula | reproduced exactly for all 16 rolls in 4 verified examples | `research/ps-move-damage-event-spec.md` §3.7 |
| Game length | ≈ 8 turns, ≈ 12.5 env steps per game | `research/rl-training-design.md` §8.1 |
| JAX dispatch | tables + per-hook switch + `optimization_barrier`. Without barriers, compile time is superlinear (28.7 s vs 2.5 s) | `research/jax-patterns.md` §1, §5 |
| Prior art | Jaxcalibur (singles randbats, unreleased); PokaiTrainer (Champions VGC search agent, Rust engine); PokeJAX (LLM-translated JAX Showdown, weak validation); vgc-engine (Rust doubles, 16.5% clean battles even with forced RNG) | `jax-vgc-engine-scope.md`, RL §3 |

## 3. Package contents

| File | Contents |
|---|---|
| `00-README.md` | this overview |
| `01-decisions.md` | **30 binding ADRs** (scope, architecture, validation, interfaces, determinism, protected contract, target freeze, process) |
| `02-architecture.md` | components, repository layout, module responsibilities, oracle CLI/RPC |
| `03-state-and-data-model.md` | ID spaces, Dex tables, Team, the full State schema with PS field mapping, step I/O, invariants |
| `04-engine-semantics.md` | normative engine behaviour: step machine, blocks A/B/C, queue, hooks, move pipeline, damage, RNG, conditions, switching, faint, Mega, quirks |
| `05-content-plan.md` | dex stages, the mechanics catalogue, handler classification, implementation order, the per-entity recipe |
| `06-data-pipeline.md` | PS → generated tables; team banks; re-pin checklist |
| `07-conformance.md` | trace format, forcing contract, projection, mask parity, crafted scenarios, replay, coverage, triage and shrinking, versioning, gates, CI tiers |
| `contract/drawkey.md` | **normative** DrawKey: fields, step tags, the inventory → DrawKind table, value domains, tie records (protected, ADR-029) |
| `08-rl-interface.md` | env API, actions and masks, observation and leak tests, reward and truncation, auto-reset and team bank, search and deployment hooks |
| `09-performance.md` | targets, mandatory JAX rules with measurements, build profiles, benchmarks, GPU playbook |
| `10-implementation-plan.md` | milestones M0–M9 with task IDs, gates and estimates |
| `11-agent-operating-manual.md` | how the implementer works: loops, rules, escalation, conventions |
| `12-risks-and-open-questions.md` | risk register, resolved questions, open questions with owners |
| `CLAUDE.md` | drop-in agent instructions for the implementation repo root |
| `catalogue.csv` | seed mechanics catalogue: 1,025 rows (every legal entity + engine/system features) with class, hooks and usage. The `stage` column is approximate; regenerate it in M0.5 (05 §1) |
| `tools/make_catalogue.py` | regenerates `catalogue.csv` from the research outputs |

## 4. Research artefacts handed over (in `../research/`, copied to `docs/research/` in the repo)

| Path | What | Status |
|---|---|---|
| `ps-turn-loop-spec.md` (**TL**) | PS turn-level control flow, legality, queue, faint, residual, positions, protocol, JAX mapping, bounds | verified by running PS; scripts in `ps-loop/` |
| `ps-move-damage-event-spec.md` (**MV**) | event system, move pipeline, damage (verified), conditions, boosts, accuracy, hook taxonomy | verified; scripts in `ps-move/` |
| `champions-mechanics-delta.md` (**CH**) | every Champions override, Mega flow, legality, timers, real-game discrepancies, PS commit velocity | verified; scripts and data in `champions/` |
| `dex-scope.md` | usage coverage, team coverage, legal pool, handler census, v0/v1/v2, team sources | data and scripts in `dex/` |
| `rng-harness-poc.md` (**RNG**) | oracle proof of concept: RNG inventory (88 sites), forcing modes, serialization, crafted states, masks, throughput | **working code** in `poc/` (port this) |
| `rl-training-design.md` (**RL**) | Jaxcalibur, VGC-Bench, PokaiTrainer, game-theoretic RL, engine API requirements, throughput targets, deployment and ToS | literature + measured replay statistics |
| `jax-patterns.md` (**JP**) | measured JAX dispatch, loops, state, RNG, rollout, debugging; H100 transfer analysis | benchmarks in `jax-bench/` |
| `../jax-vgc-engine-scope.md` | the original scoping doc (superseded where it conflicts with this package) | — |
| `review-implementability.md`, `review-fidelity.md` | independent reviews of this package; their accepted fixes are already applied | informative |

**Caveats on the research environment:**
- PS checkouts used during research lived in `/tmp` and were wiped twice by reboots. All results were re-verified on identical rebuilt checkouts at commit `d601443`.
- Some throughput numbers were measured on a heavily loaded machine (lower bounds).
- No GPU was available, so all GPU numbers are estimates.

## 5. Glossary

| Term | Meaning |
|---|---|
| PS | Pokémon Showdown (the simulator; the oracle) |
| [C] | Champions-specific behaviour (mod override) |
| Brought index `b` | 0..3: position in team-preview order |
| Sheet index `s` | 0..5: position on the team sheet |
| Slot | active position a/b (0/1) on a side; `actor = side*2 + slot` |
| Hook | one of the ~32 PS event families the engine implements (04 §6) |
| Row / bespoke | a handler expressed as table data vs as code (05 §3) |
| DrawKey | the semantic key identifying a random decision on both PS and engine sides (`contract/drawkey.md`) |
| Clean battle | engine == PS at every decision under forcing, with equal masks and draws (07 §1) |
| CBR / TMR / MPR / DPR | clean-battle / turn-match / mask-parity / draw-parity rates |
| v0 / v1 / v2 | content stages (05 §1) |
| Gate | hard acceptance criterion at a milestone (07 §12, 10) |

## 6. What this package does not include
- Trainer code (PPO, league, model): only the API that supports it (08).
- Implementation code. The proof-of-concept oracle code in `research/poc/` is the exception and should be ported.
- A GPU performance validation.
