# PROGRESS
## Milestone / task
M0 — Foundations (O, E) — done except M0.6 launchd agents and M0.7 pool-B sampler (infrastructure; not blocking)
- M0.1 repo layout — done (research/dex/raw *copied*, not moved, to data/raw/ so research/ stays reproducible)
- M0.2 PS submodule @ d601443, `make oracle` — done
- M0.3 POC oracle port — done (POC scripts 01–05, 08 reproduce research verdicts byte-for-byte except timings; check-callsites passes)
- M0.4 `oracle gen` + DrawKey + tie records + serve RPC — core done: byte-identical shards for 1 vs 12 workers; 4k battles
  (chaos + randdoubles, both classifiers) with 0 unmapped draws, 0 classifier disagreements, 0 forced-key collisions;
  RPC round trips (run/snapshot/from_snapshot/validate_team) tested at every decision point. Pending: `--legal`, `--stage`
  (needs M0.5), coverage counters (M3.8).
- M0.5 datagen — done: export_dex.js (lossless export, Champions PP asserted vs PS for every move, resolvePriority
  order metadata incl. order-only residual callbacks), legal_pool.js, census.js, dex_stages.py, catalogue stage column
  regenerated. Self-checks at d601443 reproduce exactly: v0 377/110/142 + 220 CUSTOM, v1 454/172/160 + 376,
  v2 510/215/166 + 483, core 197 moves / 6 abilities / 20 species. 17 datagen tests (self-tests 2 and 4).
- M0.6 Python env / Makefile — partial (pins + Makefile + lint + merge.sh; launchd agents todo)
- M0.7 team banks — partial: frozen real-sheet corpus done (3,211 sheets, all TeamValidator-valid, sha256 abacca4c…); pool-B sampler + validator bridge todo
- M0.8 test-support RPCs — done except `resolve_priority` (may slip to M3.1 per plan): fixedpoint, stats, sort_actions,
  legal (+ `gen --legal on|off|<fraction>`; parity with request-derived sets on 732 side-decisions), replay forced by
  DrawKey + tie records (600/600 battles reproduce under a different PS seed), minimal craft, damage (equal to the 4
  verified research examples), trace_events. `bench/oracle_legal_cost.js`: 12.4 ms per decision.


M1 — Core math and data (E) — done
- M1.1 codegen → vgcjax/data/{ids.py, tables.npz}: stage-independent ID spaces (NS 383, NM 512, NA 217, NI 167, NC 104,
  NEFFECT 1009 < 4096), per-stage `supported_*` masks, DATA_VERSION embedded and checked; deterministic; self-test 1
  (1,000 random legal sets: stats, maxpp, types, weight) passes
- M1.2 fixedpoint: 10^6 cases equal PS (domain value·m4096 < 2^31, documented)
- M1.3 stats: 1,000 random sets incl. stages, paralysis, Trick Room action speed equal PS
- M1.4 typechart: every attacking type × pool typing, plus the 10 immunity pseudo-keys, equal PS
- M1.5 damage core: 120k random core-subset cases + 3 verified research examples × 16 rolls equal PS
- M1.6 rng: contract §2 hash (checked vs a Python reference), χ² on distributions incl. 10^7 draws, vmap determinism,
  forced/draw-log/degenerate semantics, tie records (perm rank/unrank, SELECT winner)
- T0 25 s, T1 46 s (local); pre-commit hook installed

M2 — Harness core + core state machine (O, E) — done; **G-core run complete, pending human decision (ADR-102)**
- G-core run `gcore-2026-10-21-a` (50,000 fresh-seed core battles, random policy, --legal on 100%):
  CBR 100%, TMR 100%, MPR 100%, DPR 100%, error rate 0 — gate report:
  reports/conformance/gcore-2026-10-21-a/gate_report.md. Verdict FAIL only on coverage: RANDOM_TARGET 0 and
  SPEED_TIE_FIELD_EVENT 0 (structurally near-impossible / impossible in core). ADR-102 proposes counting RANDOM_TARGET
  from crafted Struggle scenarios (400/400 clean, 360 hits) and waiving FIELD_EVENT for G-core only.
- M2.7 from_ps_snapshot done (10k snapshots project equal; continued replays clean)
- M2.8 done: contract files (harness/contract/*, gates.yaml, waivers.json), gates.py (contract sha256 check),
  streaming corpus reader, coverage counts, parallel replay (~85 battles/s on 12 workers); on-device batched diff
  deferred to M9 (host replay is fast enough for now)
- done: M2.1 state/init/preview, M2.2 queue (re-key, SPEED_TIE_SELECT), M2.3 blocks A/B/C + tail + faint/win + end_turn,
  M2.4 switching (batched runSwitch, SPEED_TIE_SWITCHIN, dragIn), M2.5 core move pipeline (targeting incl. Dragon Darts,
  accuracy/OHKO/multiaccuracy, crit/roll, multihit, spread, drain/recoil/heal, boosts, secondaries, pivots, drags,
  Memento, Struggle), M2.6 masks, M2.9 core teams (oracle/lib/stageteams.js), M2.10 canary (tests/harness/test_canary.py)
- (superseded) M2.0/M2.8 harness — host-side single-battle replay (harness/replay_one.py) + parallel CLI
  (`python -m harness.cli replay <dir>`); still todo: on-device batched diff, triage/report/gates.py, protected
  harness/contract/* files and harness/gates.yaml (human must create docs/decisions/contract-hashes.md)
- todo: M2.7 from_ps_snapshot
- compile time of `step` (core, CPU): ~10 s after converting heavy unrolled loops to fori

M5 — Env API (R) — in progress (infrastructure lane)
- done: actions/canonical preview, to_ps_choice round trip, observations (leak + symmetry tests), api.step/StepOut,
  TeamBank + pair sampling (data/teams/bank_core.npz, 2000 validated core teams), AutoResetEnv + EpisodeStats
- CPU throughput (core, unoptimised): ~3,000 env-steps/s at batch 256-1024 (bench/throughput.py)
- todo: PublicEvents (obs.events is zeros), observe_privileged/hidden_targets, M5.5 PPO smoke trainer, M5.6 export

Comparison-quality work (2026-10-22)
- extended non-gating projection: found + fixed 2 hidden engine bugs; ADR-103 for promotion + SELECT-tie gap
- heuristic policy (M3.8b); 2000 heuristic core battles clean
- don't-care re-validation (07 §4.3): all kinds 0 state impact in 4000 battles except QUEUE_INSERT_TIE (1/424),
  evidence appended to ADR-101

## Last T2/T3 run
local conformance runs (not formal gate runs), PS d601443, contract_version 1:
| corpus | battles | CBR | TMR | MPR | DPR | errors |
|---|---|---|---|---|---|---|
| core seed 1 (random) | 500 | 100% | 100% | 100% (no legal recorded) | 100% | 0 |
| core seed 101 (random, --legal on) | 2000 | 100% | 100% | 100% | 100% | 0 |
| core seed 102 (random-weighted, --legal on) | 2000 | 100% | 100% | 100% | 100% | 0 |

## Top 5 buckets
none yet (no engine)

## Watch items
- [CONFIRMED, ADR-103] SPEED_TIE_SELECT ties whose members include inactive/fainted actors are not recorded (contract §4 filter); seen in
  ~0.8–2% of battles (`final.diag.unrecordedSelectTies`). If these move state (e.g. instaswitch order replacing two
  fainted mons), it will surface as STATE buckets and needs a contract ADR.
- G-core coverage requires ≥ 50 SPEED_TIE_FIELD_EVENT hits, but the core stage has no residual/SwitchIn handlers, so
  that tie kind cannot occur (0 in 4.5k core battles). Needs a waiver in docs/coverage/waivers.json (protected → human).
- PS draws Population Bomb's per-hit accuracy with a float numerator under accuracy/evasion stages; the engine logs
  ceil(); DRAW_ARGS would differ if it ever occurs (not seen yet). May need a normalisation ADR.
- SPEED_TIE_HANDLER_MUL ordinal (`mulhook_ord`) counts every MUL-hook runEvent call; the engine will have to count
  identically once hooks exist (M3.1). Brittle; watch.
- Trace size ≈ 95 KB/battle uncompressed, ~7 KB zstd (requests dominate); design estimate was 35 KB.

- ADR-101 (oracle answers the don't-care QUEUE_INSERT_TIE canonically) — needs human review; see docs/decisions.

## Blocked (docs/blocked/)
none

## Pending escalations
- ADR-101 (canonical QUEUE_INSERT_TIE in the oracle) — review
- ADR-102 (G-core coverage: RANDOM_TARGET via crafted scenarios, FIELD_EVENT waiver) — approve/reject; edit
  docs/coverage/waivers.json if approved
- ADR-103 (extend projection with hidden engine state; record SELECT ties of replacement switches) — approve/reject
- docs/decisions/contract-hashes.md must be created by the human (hashes listed in the gate report)
- Gate checkpoint: G-core report awaiting acknowledgement; M3 content waits (infrastructure continues)

## Contract hashes (must match docs/decisions/contract-hashes.md)
not yet created (harness/contract/* is written in M2.8)

## Next 3 tasks
1. Infrastructure while the gate checkpoint is pending: M3.9 reduction loop (shrinker), M3.8 coverage steering +
   `--require`, M3.8b random-weighted/heuristic policies, M5.1-M5.4 env API
2. After acknowledgement: M3.2 handler classification, then M3.3-M3.7 content (conditions, statuses, Protect, Mega,
   mid-turn switch family) using effects/hooks.py
3. MUL-hook tie records (hooks.py TODO) with the first MUL content
