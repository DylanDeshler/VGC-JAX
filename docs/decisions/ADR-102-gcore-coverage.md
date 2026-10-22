# ADR-102 — G-core coverage: RANDOM_TARGET from crafted scenarios; waive SPEED_TIE_FIELD_EVENT

- **Status:** proposed 2026-10-22 — **needs human approval** (changes a protected file: `docs/coverage/waivers.json`,
  and how G-core coverage is counted; ADR-029).
- **Affects:** `harness/gates.yaml` G-core coverage, `docs/coverage/waivers.json`.

## Problem
The G-core gate (07 §12.1) requires ≥ 50 hits of each core RNG kind, including RANDOM_TARGET and the tie kind
SPEED_TIE_FIELD_EVENT. The gate corpus (50,000 fresh-seed core battles, `gcore-2026-10-21-a`) had **0** of each:

- **RANDOM_TARGET.** A *non-degenerate* random target needs ≥ 2 living foes and a move that retargets. In the core
  stage the only path is `randomNormal` Struggle (core has no Outrage/Thrash), i.e. a mon with every move at 0 PP while
  both foes live. Retargets after a faint always have ≤ 1 living foe (degenerate, dropped on both sides by contract §7).
- **SPEED_TIE_FIELD_EVENT.** Needs ≥ 2 equal-key handlers in `fieldEvent('Residual'|'SwitchIn')`. Probe
  `oracle/tools/probes/core_field_event_handlers.js`: every candidate callback belongs to an engine-always condition
  (statuses, lockedmove, partiallytrapped, …) that **no core entity can create** (core moves are DATA_ONLY with no
  condition refs, there are no items, the six core abilities are inert). So no core battle ever has a field-event
  handler, and the tie cannot occur. It becomes reachable in v0 (Leftovers, weather, screens, …).

## Evidence
- Gate run: `reports/conformance/gcore-2026-10-21-a/gate_report.md`: CBR = TMR = MPR = DPR = 100%, error rate 0,
  every other coverage item ≥ 15,810 hits.
- RANDOM_TARGET via crafted scenarios (07 §9.2 "crafted templates"): `tests/scenarios/core/test_struggle_random_target.py`
  crafts core states with PP 0 (craft RPC), continues them in PS under random play and replays them through the engine
  with forced outcomes: **400/400 battles clean, 360 non-degenerate RANDOM_TARGET draws** matched by key and args.

## Proposal
1. Count RANDOM_TARGET coverage for G-core from the crafted-scenario corpus above (G-core has no
   "coverage-steered + crafted" column; this adds one for this kind only).
2. Waive SPEED_TIE_FIELD_EVENT for G-core only, with the probe above; it stays required for G-v0 and later.

Suggested `docs/coverage/waivers.json` entry (to be added by the human):
```json
{"gate": "G-core", "target": "SPEED_TIE_FIELD_EVENT", "adr": "ADR-102",
 "probe": "oracle/tools/probes/core_field_event_handlers.js",
 "reason": "no core-stage entity can create a fieldEvent handler"}
```

## Consequences
With both, G-core passes on the existing run. Content work for M3 waits for the gate checkpoint acknowledgement
(ADR-027); infrastructure lanes (M3.1 hook framework, M3.8 coverage, M3.9 fixtures, M5 env API) continue.
