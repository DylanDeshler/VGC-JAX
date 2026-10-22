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

M2 — Harness core + core state machine (O, E) — in progress, core engine conformant
- done: M2.1 state/init/preview, M2.2 queue (re-key, SPEED_TIE_SELECT), M2.3 blocks A/B/C + tail + faint/win + end_turn,
  M2.4 switching (batched runSwitch, SPEED_TIE_SWITCHIN, dragIn), M2.5 core move pipeline (targeting incl. Dragon Darts,
  accuracy/OHKO/multiaccuracy, crit/roll, multihit, spread, drain/recoil/heal, boosts, secondaries, pivots, drags,
  Memento, Struggle), M2.6 masks, M2.9 core teams (oracle/lib/stageteams.js), M2.10 canary (tests/harness/test_canary.py)
- partial: M2.0/M2.8 harness — host-side single-battle replay (harness/replay_one.py) + parallel CLI
  (`python -m harness.cli replay <dir>`); still todo: on-device batched diff, triage/report/gates.py, protected
  harness/contract/* files and harness/gates.yaml (human must create docs/decisions/contract-hashes.md)
- todo: M2.7 from_ps_snapshot
- compile time of `step` (core, CPU): ~10 s after converting heavy unrolled loops to fori

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
- SPEED_TIE_SELECT ties whose members include inactive/fainted actors are not recorded (contract §4 filter); seen in
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
none

## Contract hashes (must match docs/decisions/contract-hashes.md)
not yet created (harness/contract/* is written in M2.8)

## Next 3 tasks
1. M2.7 from_ps_snapshot; M2.8 contract files + gates.py + report; then the G-core run (50k fresh-seed core battles,
   --legal on 100%) and the gate checkpoint report to the human
2. M3.1 hook framework (effects/hooks.py), M3.8 coverage counters, M3.9 shrinker/fixtures
3. M5 can start after G-core (env API)
