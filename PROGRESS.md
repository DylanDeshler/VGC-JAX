# PROGRESS
## Milestone / task
M0 — Foundations (O, E) — in progress
- M0.1 repo layout — done (research/dex/raw *copied*, not moved, to data/raw/ so research/ stays reproducible)
- M0.2 PS submodule @ d601443, `make oracle` — done
- M0.3 POC oracle port — done (POC scripts 01–05, 08 reproduce research verdicts byte-for-byte except timings; check-callsites passes)
- M0.4 `oracle gen` + DrawKey + tie records + serve RPC — core done: byte-identical shards for 1 vs 12 workers; 4k battles
  (chaos + randdoubles, both classifiers) with 0 unmapped draws, 0 classifier disagreements, 0 forced-key collisions;
  RPC round trips (run/snapshot/from_snapshot/validate_team) tested at every decision point. Pending: `--legal`, `--stage`
  (needs M0.5), coverage counters (M3.8).
- M0.5 datagen — todo
- M0.6 Python env / Makefile — partial (pins + Makefile + lint + merge.sh; launchd agents todo)
- M0.7 team banks — todo
- M0.8 test-support RPCs — todo

## Last T2/T3 run
none yet

## Top 5 buckets
none yet (no engine)

## Watch items
- SPEED_TIE_SELECT ties whose members include inactive/fainted actors are not recorded (contract §4 filter); seen in
  ~0.8–2% of battles (`final.diag.unrecordedSelectTies`). If these move state (e.g. instaswitch order replacing two
  fainted mons), it will surface as STATE buckets and needs a contract ADR.
- Trace size ≈ 95 KB/battle uncompressed, ~7 KB zstd (requests dominate); design estimate was 35 KB.

## Blocked (docs/blocked/)
none

## Pending escalations
none

## Contract hashes (must match docs/decisions/contract-hashes.md)
not yet created (harness/contract/* is written in M2.8)

## Next 3 tasks
1. M0.5 datagen (export_dex.js, legal_pool.js, census.js, dex_stages.py)
2. M0.8 test-support RPCs (fixedpoint, stats, sort_actions, legal, replay-by-DrawKey, craft, damage)
3. M1.2–M1.4 fixedpoint / stats / typechart with RPC-driven tests
