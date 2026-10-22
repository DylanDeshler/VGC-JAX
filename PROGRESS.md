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

## Last T2/T3 run
none yet

## Top 5 buckets
none yet (no engine)

## Watch items
- SPEED_TIE_SELECT ties whose members include inactive/fainted actors are not recorded (contract §4 filter); seen in
  ~0.8–2% of battles (`final.diag.unrecordedSelectTies`). If these move state (e.g. instaswitch order replacing two
  fainted mons), it will surface as STATE buckets and needs a contract ADR.
- Trace size ≈ 95 KB/battle uncompressed, ~7 KB zstd (requests dominate); design estimate was 35 KB.

- ADR-101 (oracle answers the don't-care QUEUE_INSERT_TIE canonically) — needs human review; see docs/decisions.

## Blocked (docs/blocked/)
none

## Pending escalations
none

## Contract hashes (must match docs/decisions/contract-hashes.md)
not yet created (harness/contract/* is written in M2.8)

## Next 3 tasks
1. M1.1 codegen (ids.py, tables.npz) from data/export + stages
2. M1.2–M1.4, M1.6: fixedpoint, stats, typechart, rng with RPC-driven tests
3. M2.0 harness core (oracle_client with cache, project.py)
