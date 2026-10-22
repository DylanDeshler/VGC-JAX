# 02 — System architecture and repository layout

## 1. Components

```
                    ┌──────────────────────────── oracle/ (Node) ────────────────────────────┐
                    │ ps/  (git submodule, pinned PS commit; built once: npm ci && node build)│
 tools/datagen ───► │ lib/ (from research/poc/lib): ps loader, teams, player, context hooks,  │
 (export dex)       │      InstrumentedPRNG (record|replay|keyed|override), classify,         │
                    │      project (canonical projection), coverage counters                   │
                    │ bin/oracle.js: CLI (gen corpus) + JSON-RPC server (replay/snapshot/legal)│
                    └───────────────┬───────────────────────────────▲───────────────────────────┘
                                    │ NDJSON(.gz) traces             │ JSON-RPC (stdin/stdout)
                                    ▼                                │
 ┌──────────── vgcjax/ (Python package; the engine) ───────────┐   ┌─┴──────── harness/ (Python) ──────────┐
 │ data/   generated tables + id enums (from datagen)          │   │ oracle_client.py  (spawn/RPC pool)     │
 │ core/   fixedpoint, rng, stats, typechart, dex pytree       │◄──┤ corpus.py   (load/shard traces)        │
 │ battle/ state, phases, queue, blocks A/B/C, move pipeline,  │   │ replay.py   (batched forced replay)    │
 │         damage, targeting, boosts, status, conditions,      │   │ project.py  (engine → canonical JSON)  │
 │         switching, faint, residual, requests(masks)         │   │ diff.py / triage.py / shrink.py        │
 │ effects/ hook framework + rows + bespoke handler modules    │   │ coverage.py / report.py / gates.py     │
 │ env/    api (init/step/observe/legal), obs, actions,        │   └────────────────────────────────────────┘
 │         autoreset, team bank, episode stats                 │
 │ interop/ ps_choice, protocol_reducer, trace export          │   tests/ (pytest): unit, golden, scenario,
 │ search/  infostate, determinize, hash (later milestones)    │          property, conformance-smoke, perf
 └─────────────────────────────────────────────────────────────┘   bench/: throughput & compile benchmarks
```

Data flow for a conformance run:
1. `oracle gen` runs N battles in PS, using a worker-thread pool with the instrumented PRNG and coverage-steered team and scenario selection. It writes trace shards (07 §3).
2. `harness.replay` loads shards and builds a batched initial `State` from each trace header (teams, SPs, preview choices). Then it runs the jitted `step` with `forced` outcomes and `draw_log` on, for every decision point in the trace.
3. `harness.project` converts each engine state to the canonical projection. `harness.diff` compares it with the trace's PS projection. It also compares masks and draw requests.
4. Mismatches go to `harness.triage`, which classifies them by first-divergence field or draw kind and buckets them by signature. One representative per bucket goes to `harness.shrink`, which calls the oracle over RPC to re-run reduced variants. The shrunk failures become **fixtures** in `tests/fixtures/conformance/`.
5. `harness.report` updates the metrics dashboard (`reports/conformance/<date>.md` + JSON), and `harness.gates` evaluates the milestone gates.

Data flow for training (out of scope to build here; the API must support it):
- `env.autoreset` plus `vgcjax.env.api` live inside one `jit(scan(...))` together with the policy.
- `TeamBank` arrays are device-resident.

## 2. Repository layout (create exactly this)

```
vgc-jax/
├── CLAUDE.md                     # copy of design/CLAUDE.md (agent instructions for this repo)
├── .nvmrc                        # 26.8.1
├── .python-version               # 3.12
├── .gitignore                    # data/raw/, traces/, reports/**/shards, scratch/, tests/.oracle_cache/, .venv
├── Makefile                      # oracle, t0, t1, t2, t3, gate, bench targets
├── ops/launchd/                  # nightly T2 (02:00) and weekly T3 (Sunday) launchd agents
├── scripts/merge.sh              # refuses to merge to main unless `make t1` passes
├── PROGRESS.md                   # milestone status, gate metrics, open divergence classes
├── README.md
├── THIRD_PARTY_NOTICES           # PS MIT licence etc.
├── pyproject.toml                # python deps pinned (jax, numpy, flax, chex, pytest, hypothesis, orjson, zstandard)
├── uv.lock
├── docs/
│   ├── design/                   # copy of this design package (read-only reference; changes go via ADRs)
│   ├── decisions/                # new ADRs (ADR-1xx-*.md)
│   ├── catalogue.csv             # mechanics catalogue: one row per entity (generated, then status-edited) — see 05
│   ├── coverage/                 # pairs_<stage>.json, waivers.json (protected, ADR-029)
│   ├── blocked/                  # one note per time-boxed blocked bucket (11 §3)
│   ├── oracle-changelog.md       # PS re-pin log
│   └── research/                 # SELECTED research artefacts only (manifest in 10 M0.1; never venvs, ext/, web/, raw/)
├── oracle/
│   ├── ps/                       # git submodule: smogon/pokemon-showdown @ pinned commit (never edited)
│   ├── package.json              # node deps for oracle only (source-map etc.); "engines": {"node": ">=22"}
│   ├── lib/                      # ported from research/poc/lib (ps.js, teams.js, player.js, context.js,
│   │                             #   callsite.js, classify.js, rng.js, project.js, runner.js, experiment.js)
│   │   ├── coverage.js           # NEW: runEvent/handler hit counters, site counters
│   │   ├── scenarios.js          # NEW: crafted-state builders (JSON-edit based) + Update-normalisation
│   │   └── trace.js              # NEW: trace record writer (format in 07 §3)
│   ├── tools/                    # scan_callsites.js, build_inventory.js, inventory_annotations.js (from POC)
│   ├── data/rng_callsites.json   # inventory (regenerated per pin)
│   └── bin/oracle.js             # CLI + RPC server
├── tools/
│   └── datagen/
│       ├── export_dex.js         # PS Dex.mod('champions') → data/export/*.json (raw, lossless)
│       ├── census.js             # handler census (from research/dex/scripts/handler_census.js)
│       ├── legal_pool.js         # from research/dex/scripts/legal_pool.js
│       ├── dex_stages.py         # v0/v1/v2 entity lists (from research/dex/scripts/dex_select.py)
│       ├── codegen.py            # export JSON → vgcjax/data/{tables.npz, ids.py, hook_rows.npz, handlers_stub.py}
│       ├── scaffold.py           # copies a handler stub into the right vgcjax/effects/*.py module
│       ├── core_inert.json       # abilities allowed in the `core` stage (protected by ADR; 10 M2.9)
│       └── teams/                # team bank builders (usage sampler, replay scraper, validator bridge)
├── data/
│   ├── raw/                      # git-ignored: smogon-chaos/<month>/, ps-replays/<date>/, vgcbench/<date>/ + SOURCES.md + MANIFEST.sha256
│   ├── export/                   # datagen output (committed; diffable on re-pin)
│   └── teams/                    # validated team banks (npz + json manifest); real_sheets_mc_v1.jsonl (frozen, usernames stripped)
├── vgcjax/
│   ├── __init__.py
│   ├── config.py                 # EngineConfig, Ruleset
│   ├── data/                     # GENERATED: ids.py, tables.npz loader, hook tables — never hand-edit
│   ├── core/                     # fixedpoint.py, rng.py, stats.py, typechart.py, dex.py, bits.py
│   ├── battle/                   # state.py, init.py, snapshot.py (from_ps_snapshot), step.py, queue.py, block_a.py, block_b.py, block_c.py,
│   │                             # move_pipeline.py, targeting.py, damage.py, accuracy.py, boosts.py,
│   │                             # status.py, conditions.py, switching.py, faint.py, residual.py,
│   │                             # mega.py, requests.py, events_log.py, errors.py
│   ├── effects/                  # hooks.py (framework), predicates.py, rows.py, and bespoke handler modules:
│   │                             # moves_hit.py, moves_try.py, moves_bp.py, moves_misc.py, abilities_*.py,
│   │                             # items_*.py, conditions_*.py  (each handler: decorator registering
│   │                             # (hook, effect_id) + PS source citation)
│   ├── env/                      # api.py, obs.py, actions.py, autoreset.py, teambank.py, stats.py, wrappers.py
│   ├── interop/                  # ps_choice.py, protocol_reducer.py, trace_export.py
│   └── search/                   # infostate.py, determinize.py, hashing.py   (M8)
├── harness/                      # oracle_client.py, corpus.py, replay.py, project.py, diff.py, triage.py,
│   │                             # shrink.py, coverage.py, report.py, gates.py, cli.py
│   ├── contract/                 # PROTECTED (ADR-029): dontcare.json, degenerate.json, normalisations.json,
│   │                             #   canonicalisation.json, projection_fields.json
│   └── gates.yaml                # PROTECTED: thresholds, corpus composition/sizes, coverage goals
├── tests/
│   ├── unit/                     # pure functions (fixedpoint, stats, typechart, rng, queue selection…)
│   ├── golden/                   # damage/stat goldens generated from PS
│   ├── scenarios/                # one file per mechanic family; each = crafted PS state + forced outcomes
│   ├── fixtures/conformance/     # shrunk regression traces (auto-added by harness)
│   ├── property/                 # leak/symmetry/determinism/invariants (hypothesis)
│   └── perf/                     # compile-time & throughput budget checks
├── bench/                        # throughput, compile time, memory, hlo_kernels (CPU + GPU variants)
└── reports/                      # conformance + perf reports (generated)
```

## 3. Module responsibilities (engine)

| Module | Responsibility | PS counterpart |
|---|---|---|
| `core/fixedpoint.py` | `trunc`, `modify`, `chain_modify`, `final_modify`, `poke_round`, 16-bit wrap | `sim/battle.ts` `modify/chainModify/finalModify/trunc` |
| `core/rng.py` | counter hash, draw kinds, forced lookup, draw log | `sim/prng.ts` (semantics only) |
| `core/stats.py` | Champions stat formula, stage multipliers, `calculateStat` | `ch/scripts.ts statModify`, `sim/pokemon.ts calculateStat/getStat` |
| `core/typechart.py` | effectiveness and immunity tables + overrides | `sim/dex-data.ts`, `runEffectiveness` |
| `battle/state.py` | State pytree definitions + constructors | `Battle/Side/Pokemon/Field` |
| `battle/step.py` | phase dispatch, apply choices, Blocks A/B/C, pause/resume | `commitChoices`, `turnLoop`, `runAction` |
| `battle/queue.py` | queue entries, resolveAction, re-keying, selection | `sim/battle-queue.ts`, `getActionSpeed`, `comparePriority` |
| `battle/move_pipeline.py` | runMove→useMove→trySpreadMoveHit step chain | `sim/battle-actions.ts` + `ch/scripts.ts` overrides |
| `battle/damage.py` | getDamage/modifyDamage incl. crit, roll, STAB, type, burn, final chain | `sim/battle-actions.ts:1639-1776`, `ch/scripts.ts:196-312` |
| `battle/targeting.py` | getTarget, getRandomTarget, getMoveTargets, redirection, adjacency | `sim/battle.ts`, `sim/pokemon.ts` |
| `battle/switching.py` | switchIn, runSwitch batching, dragIn, cancelAction, party order | `sim/battle-actions.ts:switchIn/runSwitch/dragIn` |
| `battle/conditions.py` | add/remove/restart for status, volatile, side, slot, field, weather, terrain; duration semantics | `addVolatile/addSideCondition/setWeather/...` |
| `battle/residual.py` | ordered residual walk + duration ticks | `fieldEvent('Residual')` |
| `battle/faint.py` | faint queue, resolveFaints, checkWin | `faintMessages/resolveFaints/checkWin` |
| `battle/mega.py` | runMegaEvo (forme change, ability swap+Start, side lock) | `runMegaEvo`, `formeChange` |
| `battle/requests.py` | legal masks per phase, joint constraints | `side.ts chooseMove/chooseSwitch/chooseTeam` |
| `effects/hooks.py` | holder enumeration, handler ordering, MUL/GATE/REPL folds, barriers | `runEvent/singleEvent/priorityEvent/eachEvent/fieldEvent` |
| `env/*` | RL-facing API (08) | — |
| `interop/*` | PS choice strings, PS protocol → InfoState, trace export | — |

## 4. Interfaces between components

### 4.1 Oracle CLI
```
node oracle/bin/oracle.js gen  --n 10000 --workers 12 --seed 1 --teams <bank.json|chaos|replays|randdoubles>
                               --policy random|random-weighted|heuristic --scenario fresh|crafted:<name>
                               --stage core|v0|v1|v2 --require <entity-id>[,...] --legal on|off|<fraction>
                               --coverage-steer on --format gen9championsvgc2026regmc
                               --out <dir>              # shards named shard-%05d.ndjson.zst; deterministic per 07 §2.1
node oracle/bin/oracle.js legal --snapshot <file> --side p1|p2   # acceptance-enumerated legal set
node oracle/bin/oracle.js serve                      # JSON-RPC on stdin/stdout (one request per line)
node oracle/bin/oracle.js check-callsites            # drift alarm (non-zero exit on change)
node oracle/bin/oracle.js coverage --in traces/      # aggregate coverage report
```

### 4.2 Oracle RPC methods (line-delimited JSON; every response carries `ps_commit`)
| method | params | result |
|---|---|---|
| `run` | battle spec (teams, seed, policy, scenario) | full trace record |
| `replay` | spec + choices + forced draws (engine-key dict) + `until` | trace record (or projection at `until`) |
| `snapshot` | spec + choices + `at` decision index | serialized PS state JSON |
| `from_snapshot` | snapshot JSON + choices + forced | trace record continuing from snapshot |
| `legal` | snapshot JSON + side | list of accepted canonical choice strings (acceptance enumeration, 07 §6) |
| `craft` | scenario description (field/side/mon overrides) | normalised snapshot JSON (after one `Update` pass) |
| `validate_team` | packed team | validator errors (empty = legal) |
| `trace_events` | snapshot + choices + forced | full PS event-dispatch trace for one turn (port `research/ps-move/trace-turn.js`); used for debugging |

**Test-support RPCs (batched; used by golden and unit tests):**

| method | params | result |
|---|---|---|
| `fixedpoint` | list of `(op, args)` for `trunc`, `modify`, `chainModify`, `finalModify` | PS results (`battle.*` on a dummy battle) |
| `stats` | list of sets | `storedStats`, `maxhp`, `maxpp` per set |
| `damage` | list of (attacker set + boosts + status, defender set + boosts + status, move, field and side state, crit, roll, spread flag) | PS `getDamage` results through a crafted battle with forced draws (port `research/ps-move/damage-examples.js`) |
| `sort_actions` | synthetic queue entries (order, priority, speed) + forced tie permutation | PS `comparePriority` / `speedSort` order |
| `resolve_priority` | synthetic handler set (effect, holder speed, event) | PS `resolvePriority` keys and the sorted order |

### 4.3 Engine ↔ harness
The harness only uses public engine APIs (`init`, `init_from_order`, `step`, `legal_masks`), plus `vgcjax.battle.snapshot.from_ps_snapshot(json)` for crafted scenarios and shrinking.

The canonical projection lives in `harness/project.py`: `engine_to_json` and `ps_to_expected`.

Parsing rules:
- **Trace NDJSON** is parsed only in `harness/`.
- **PS *state* JSON** is the single exception: it may be parsed in `vgcjax/battle/snapshot.py`.

## 5. Build and run matrix

| Target | Hardware | Used for |
|---|---|---|
| CPU dev | Apple M4 Max (14 cores) or any x86 | all unit, scenario and conformance work; compile-time budgets |
| GPU perf | 1× H100, if available | throughput gates at M4, M6 and M9. If unavailable, record CPU proxies and flag "GPU gate pending" in PROGRESS.md |
| Oracle pool | same CPU host | 12 workers ≈ 160–450 battles/s (`research/rng-harness-poc.md` §8) |

## 6. Execution environment and "CI"

There are no hosted runners. T2/T3 need hours of oracle time on the dev box.

**Repo.**
- Location: `~/Documents/research/pokemon/vgc-jax`, git, branch `main`. A private remote is optional.

**CI is local make targets:**
- `make t0` runs as a git pre-commit hook.
- `make t1` must pass before any merge to `main`. `scripts/merge.sh` enforces this; merge with `--no-ff`.
- `make t2` (nightly, 02:00) and `make t3` (Sunday 03:00) run from launchd agents in `ops/launchd/`. They write `reports/LATEST.md` and `reports/catalogue_status.json`, which every agent session reads first (11 §9).

**`make oracle`:**
1. `git submodule update --init oracle/ps`;
2. `cd oracle/ps && npm ci` (or `npm install --no-save` if PS ships no lockfile at the pin; record `npm ls --all` in `oracle/ps-deps.lock.txt`);
3. `node build`.

**Submodule hygiene.** `.gitmodules` sets `ignore = untracked` for `oracle/ps`, because `node_modules/` and `dist/` are build outputs. Cache them keyed by the submodule SHA.

**Storage:**
- Budget 200 GB of disk.
- Keep the last 7 nightly runs plus every gate run. Prune the rest (`make prune`).
- Passing shards are deleted after a run unless `--keep`. Failing battles' traces are kept in `reports/conformance/<run>/failures/`.
