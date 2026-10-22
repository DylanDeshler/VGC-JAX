# 10 — Implementation plan: milestones, tasks, gates, ordering

## Conventions
- **Task IDs** (`M2.4` etc.) are referenced in commits and in `PROGRESS.md`.
- **Gates are hard** (ADR-027): content work for milestone N+1 does not start until gate N passes. Infrastructure lanes may run ahead. When a gate passes, send the report to the human (gate checkpoint).
- **Target freeze** (ADR-028): no scheduled PS re-pins until G-v0.
- **Lanes:**
  - **O**: oracle and harness (Node + Python harness);
  - **E**: engine core;
  - **C**: content (entities);
  - **R**: RL interface;
  - **P**: performance.
  
  With one agent, lanes are serial. Parallel agents can own distinct lanes (11 §6).
- **Estimates** are in focused engineer-weeks (EW) for an experienced engineer with heavy LLM assistance, ±50%. The critical path is M0 → M1 → M2 → G-core → M3 → M4 → G-v0 → M6 → G-v1 → M7 → G-v2.

---

## M0 — Foundations (O, E) · ~2.5–3 EW

| ID | Task | Done when |
|---|---|---|
| M0.1 | **Create the repo layout (02 §2), copying by manifest.** **Copy:** `design/` → `docs/design/`; `design/CLAUDE.md` → `CLAUDE.md`; `design/catalogue.csv` → `docs/catalogue.csv`; `../jax-vgc-engine-scope.md` → `docs/research/`. **Into `docs/research/` copy only:** `*.md`; `poc/{lib,tools,scripts,results,rng_callsites.json}`; `dex/*.json`; `dex/scripts/` (minus `__pycache__`); `dex/out/`; `ps-loop/`; `ps-move/`; `champions/*.{js,py,json,txt}`, `champions/gh/`, `champions/diff*`; `jax-bench/*.{py,sh}`, `jax-bench/results/`. **Never copy:** `jax-bench/.venv*`, `jax-bench/ext/`, `**/__pycache__`, `champions/web/`, `dex/raw/`. **Move** `research/dex/raw/*` → `data/raw/` (git-ignored) per 06 §4.0. Write `.gitignore` | tree matches 02; `git count-objects -vH` shows < 50 MB; no file > 10 MB tracked |
| M0.2 | PS submodule at `d601443`; `make oracle` (02 §6); `.nvmrc`; `ps-deps.lock.txt` | `node -e "require('./oracle/ps/dist/sim')"` works; `git status` clean |
| M0.3 | Port the POC oracle (`docs/research/poc/lib`, `tools`) into `oracle/`. Change only default paths: the inventory goes in `oracle/data/`, `source-map` comes from `oracle/node_modules`, and `PS_ROOT`/`CHAOS_STATS` defaults point into the repo | `PS_ROOT=oracle/ps CHAOS_STATS=data/raw/smogon-chaos/2026-09/gen9championsvgc2026regmcbo3-1760.json node docs/research/poc/scripts/0N_*.js` reproduces the verdicts in `docs/research/poc/results/0N_*.json` for N = 1..8; `oracle check-callsites` passes |
| M0.4 | `oracle gen` writing traces per 07 §3, with the **DrawKey per `contract/drawkey.md`** and **semantic tie records** (instrument `speedSort` callers: `queue.sort`, `runSwitch` speedOrder, `fieldEvent`); deterministic seeding (07 §2.1); `serve` RPC with `run`, `snapshot`, `from_snapshot`, `validate_team` | 1k-battle shard is byte-identical across `--workers 1` and `--workers 12`; every draw maps to a DrawKind or is on the don't-care/degenerate lists; RPC round-trip tests |
| M0.5 | Datagen: `export_dex.js` (`--format`), `legal_pool.js`, `census.js` (ports of `research/dex/scripts`), `dex_stages.py` producing `core`, v0, v1, v2 (05 §1); regenerate the catalogue `stage` column | datagen self-tests 2 and 4 pass (06 §3); stage self-check numbers match (05 §1) |
| M0.6 | Python env via `uv` with exact pins (ADR-024); pytest + hypothesis; Makefile `t0`/`t1`; pre-commit hook; `scripts/merge.sh`; persistent compile cache configured; `ops/launchd/` agents for T2/T3 (initially running whatever exists) | `make t1` green on empty suites; launchd agent writes `reports/LATEST.md` |
| M0.7 | Team banks: **frozen real-sheet corpus** `real_sheets_mc_v1.jsonl` (06 §4.0: strip usernames, deterministic SP/gender fill); pool B sampler ported from `research/dex/scripts/team_sources.js` (stage-parameterised); validator bridge | ≥ 2k valid core-stage-compatible pool-B teams *where possible* (the core pool is narrow, 05 §1.1); ≥ 2k valid v0 and v1 teams; real-sheet corpus sha256 recorded |
| M0.8 | **Oracle test-support RPCs** (02 §4.2): `fixedpoint`, `stats`, `sort_actions`, `legal` (+ `gen --legal`), `replay` (forced by DrawKey), minimal `craft` (teams, leads, hp, boosts, pp, turn, active_turns, status), `damage` (core-subset sets first), `trace_events`. `resolve_priority` may slip to M3.1 | each RPC has a round-trip test; `bench/oracle_legal_cost.js` records the per-decision cost of acceptance enumeration |

## M1 — Core math and data (E) · ~1.5 EW
| ID | Task | Done when |
|---|---|---|
| M1.1 | `codegen.py` → `ids.py`, `tables.npz`, `hook_rows.npz` (empty for now) for the core stage and v0 | Dex loads; `DATA_VERSION` check; datagen self-test 1 passes |
| M1.2 | `core/fixedpoint.py` | 10⁶ random cases equal the `fixedpoint` RPC |
| M1.3 | `core/stats.py`: Champions stats, stage multipliers, getStat incl. paralysis and the 10000 cap | 1,000 random sets equal the `stats` RPC |
| M1.4 | `core/typechart.py` (20×20 + immunity pseudo-columns) | equal to PS `Dex.types` |
| M1.5 | `battle/damage.py` core: no hooks; explicit spread/crit/roll/burn; ModifyDamage chain = identity until M3.1 | the 4 verified examples × 16 rolls (`research/ps-move/damage-examples.js`) plus 100k `damage` RPC cases restricted to the core subset (no items, core abilities, no side conditions/weather/terrain): exact. Screens, weather, items and abilities join the goldens when their hooks exist (M3/M4) |
| M1.6 | `core/rng.py`: words, hash, distributions, forced lookup, draw log | chi-square tests pass; vmap determinism; hash statistics sanity check (χ² on 10⁷ draws per distribution) |

## M2 — Harness core, state machine, queue, switching, requests (O, E) · ~3–3.5 EW
| ID | Task | Done when |
|---|---|---|
| M2.0 | **Harness core**: `harness/oracle_client.py` (process pool + content-hash cache per 07 §10.3), `harness/project.py` (both directions, core fields), the scenario-test runner (uses `craft` + `run`, batched per 09 §3) | runner executes a trivial scenario end to end against a stub engine |
| M2.1 | `battle/state.py` (03), `init`, `init_from_order`, preview | `eval_shape` works; invariants hold |
| M2.2 | Queue (04 §5): resolve, order classes, fractional priority at commit, re-key, selection with tie draws, mutations | selection equals the `sort_actions` RPC on 10k synthetic queues; unit tests per mutation |
| M2.3 | Blocks A, B, C; pause/resume; `post_action`; faint/win; end of turn; turn cap | property test: invariants hold under random legal actions |
| M2.4 | Switching: switchIn, party order, runSwitch batching rules, cancelAction, instaswitch, EOT replacement loops | scenarios: start leads, unbatched double switch, EOT replacement with 1 bench, U-turn pause (pivot moves are DATA_ONLY `selfSwitch`) |
| M2.5 | Move pipeline for core moves: targeting (spread, ally-first, retarget), accuracy/evasion/OHKO, crit (incl. Battle/Shell Armor veto), damage, multihit, data boosts, recoil/drain/heal, PP, Struggle; Levitate grounding; Stall | scenario per path |
| M2.6 | `requests.py`: masks, joint constraints, SWITCH rules, hidden-info semantics | MPR 100% on 10k core battles with `--legal on` |
| M2.7 | `battle/snapshot.py` `from_ps_snapshot` (decision points incl. mid-turn SWITCH pauses) | `engine_to_json(from_ps_snapshot(s)) == project(s)` on 10k core-stage snapshots (3 random decision points per battle, including pivot pauses) |
| M2.8 | Harness replay pipeline: `corpus.py`, `replay.py` (batched forced replay, sharded CPU), `diff.py` (on-device), draw-multiset and mask comparison, `report.py`, `gates.py` (reads protected `gates.yaml`, prints contract hashes); `harness.cli gen-projection-map` | end to end on 10k core battles; report generated |
| M2.9 | Core-stage team generator (05 §1.1) | ≥ 500 distinct valid core teams (pool is narrow) |
| M2.10 | Canary test (ADR-029) | T1 includes it; it fails if the harness misses the injected bug |
| **G-core** | CBR ≥ 99.5% on 50k core battles (fresh seed); MPR 100% (`--legal on` 100%); DPR 100%; 0 error flags; core RNG sites (05 §1.1 kinds) each ≥ 50 hits | → M3 |

## M3 — Hook framework, coverage tooling, v0 foundations (E, O, C) · ~3.5 EW
Order matters: the infrastructure tasks M3.1, M3.8, M3.8b and M3.9 come **before** the content tasks M3.3–M3.7.

| ID | Task | Done when |
|---|---|---|
| M3.1 | `effects/hooks.py`: holder enumeration, order metadata, dynamic speed, sort, MUL/GATE/REPL folds, suppression masks, barriers, per-hook switch registry, `scaffold.py`; `resolve_priority` RPC if it slipped | ordering equals `resolve_priority` on synthetic handler sets; compile-time unit benchmark |
| M3.8 | Coverage counters and steering (07 §9); full `craft` RPC + crafted templates; `--require`; 1M-battle coverage run → `docs/coverage/pairs_v0.json` | nightly coverage report |
| M3.8b | `random-weighted` and `heuristic` policies (07 §2) | deterministic under seed |
| M3.9 | Shrinker, triage buckets, fixture emission, `refresh-fixtures` (07 §10) | first real fixture auto-generated and collected by pytest |
| M3.2 | `handler_classification.json` for v0 (auto + review); rows codegen; datagen self-test 3 (v0 scope) | static check: every v0 handler classified and registered |
| M3.3 | Conditions framework (04 §9): add/restart/end, durations, duration callbacks, residual walk with faint checks | scenarios: Tailwind / Trick Room / screens / weather expiry turns match PS |
| M3.4 | Statuses (Champions) + core volatiles (05 §4 step 2) | catalogue rows `impl` |
| M3.5 | Protect family + Fake Out ([C] mask) + Wide/Quick Guard + Feint | |
| M3.6 | Mega Evolution end to end (04 §13) | scenarios: Mega speed same turn, Mega weather ordering, side lock |
| M3.7 | Mid-turn switch family: Parting Shot condition, Eject Button [C], Emergency Exit [C], Red Card / drag moves | scenarios incl. two pending switches from one action |

## M4 — v0 content complete (C) · ~4–6 EW
| ID | Task | Done when |
|---|---|---|
| M4.1 | 05 §4 v0 order, steps 4–11 | all v0 catalogue rows `impl` |
| M4.2 | Divergence burn-down (11 §3 loop, with time-boxing) | G-v0 |
| M4.3 | Perf pass: CPU proxy budgets (09 §1.1); **request GPU access now** (09 §1.1) and, if granted, run 09 §5 steps 1–3 | `reports/perf/` entry; GPU request recorded in PROGRESS |
| **G-v0** | 07 §12.1 G-v0 with the gate-run composition (a)+(b)+(c); coverage per 07 §9.3; CPU proxy budgets; 0 error flags | → M6 content; resume monthly re-pins (ADR-028) |

## M5 — Env API and RL integration (R) · ~2–3 EW (starts after G-core; parallel with M3/M4)
| ID | Task | Done when |
|---|---|---|
| M5.1 | `env/actions.py` (encodings, preview tables, canonical/Illusion rule), `interop/ps_choice.py` (to/parse) | round trip over all legal actions in a 100k-decision corpus |
| M5.2 | `env/obs.py` (08 §4), `PublicEvents` emission, HP rule | leak and symmetry property tests pass |
| M5.3 | `observe_privileged`, `hidden_targets` | |
| M5.4 | `env/autoreset.py` (incl. `final_obs`), `teambank.py`, `stats.py`, seat-symmetry test | |
| M5.5 | PureJaxRL-style PPO **smoke trainer**: core-stage teams before G-v0, v0 teams after; 5×10⁷ env steps (≤ 8 h on sharded CPU) | ≥ 95% vs uniform random and ≥ 70% vs `MaxBPPolicy` (picks the legal move maximising BP × STAB × type effectiveness against the current foe target using public info only; ties to lowest index; never switches). 2,000 games, seats alternated; report the 95% CI lower bound. Re-run on v0 teams after G-v0 |
| M5.6 | Replay export via oracle (08 §9); `engine-policy` oracle policy (actions supplied over RPC) | exported engine games re-simulate identically in PS |

## M6 — v1 content (C) · ~5–7 EW
| ID | Task | Done when |
|---|---|---|
| M6.1 | Framework extensions for v1 (05 §6): Revival Blessing variant, Transform, Illusion, Magic Bounce, forme changes, ability copy | scenario tests |
| M6.2 | All v1 catalogue rows | `impl` |
| M6.3 | Burn-down | G-v1 |
| **G-v1** | 07 §12.1 G-v1; GPU ≥ 3×10⁴ steps/s, or CPU proxy budgets plus "GPU gate pending" | → M7 |

## M7 — v2 full legal pool (C) · ~3–4 EW
| ID | Task | Done when |
|---|---|---|
| M7.1 | Remaining entities | all rows `done` |
| **G-v2** | 07 §12.1 G-v2; all reachable RNG sites ≥ 50 hits | project "faithful" milestone |

## M8 — Search and deployment support (R) · ~2–3 EW (after G-v0; parallel)
| ID | Task | Done when |
|---|---|---|
| M8.1 | `infostate`, `sample_hidden` (incl. hidden counters), `determinize` + round-trip property | property tests |
| M8.2 | `hash_infostate`, `apply_preview` (vmappable 90×90), `damage_calc` on hypotheses, `public_outcome_match` (MAY) | unit tests |
| M8.3 | `protocol_reducer` + protocol round-trip test (08 §4); poke-env transport client against a **local** PS server | round-trip equality on 10k engine games |

## M9 — Performance hardening (P) · ~2 EW (after G-v1)
H100 profiling (09 §5), or CPU-only hardening against the proxy budgets if there is no GPU. Release-build options; memory budget; batch-size sweep.

---

## Recurring events
- **Before G-v0:** no scheduled re-pins (ADR-028). When the next regulation is announced (Reg M-C ends 2026-12-02), run datagen for it on a branch and record the delta only.
- **After G-v0:** monthly re-pin (06 §6, about 0.5–2 EW each) plus regulation-switch mini-milestones.

## Totals (single agent, serial lanes)
| Milestones | Total |
|---|---|
| M0–M4 + M5 (trainable, validated v0) | ≈ 18–22 EW |
| + M6 (v1) | ≈ 24–30 EW |
| + M7, M8, M9 (v2 with search and deployment support and performance hardening) | ≈ 31–40 EW |

Parallel agents on lanes O, R and P shorten wall-clock time, but not the content critical path.

## Dependency graph
```
M0 ─► M1 ─► M2 ─► G-core ─► M3(3.1→3.8→3.8b→3.9→3.2→3.3..3.7) ─► M4 ─► G-v0 ─► M6 ─► G-v1 ─► M7 ─► G-v2
                    │                                                     │              │
                    └──► M5 (R lane; M5.5 on core teams until G-v0) ──────┤              └──► M9
                                                                          └──► M8
```
