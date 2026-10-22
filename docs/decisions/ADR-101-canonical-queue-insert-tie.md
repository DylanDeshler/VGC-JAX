# ADR-101 — The oracle answers QUEUE_INSERT_TIE canonically

- **Status:** accepted by the implementer 2026-10-21; **flagged for human review** (touches how a don't-care kind is
  produced, though not the protected comparison contract).
- **Supersedes:** nothing. Refines ADR-012 / `contract/drawkey.md` §6 (QUEUE_INSERT_TIE is don't-care) on the oracle side.

## Problem
`BattleQueue.insertChoice` (PS `sim/battle-queue.ts:375-406`) inserts an action at `random(firstIndex, lastIndex+1)`
among queued actions that tie with it on `comparePriority`. The contract lists this draw as don't-care, and the
harness leaves it free. But the insertion position leaks into a forced key: lead and replacement switch-ins insert one
`runSwitch` action per mon, PS runs them as one batch attributed to the **head** entry, and the head's pokemon is the
`step_actor` of every draw in that SWITCHIN step (Trace targets, etc.). With equal-speed switch-ins, a free
QUEUE_INSERT_TIE changes `step_actor` and the engine cannot reproduce it.

Evidence: forced replay with a different PS seed (`scratch/replay_check.js`), 200 chaos-team battles: 1 forced draw
(ABILITY_TRACE_TARGET, turn 0) missed its key because `step_actor` was p1a instead of p2a. PS also runs the
EmergencyExit check of a batched runSwitch only for the head mon (`sim/battle.ts:2961-2963`), so the head is not
strictly state-irrelevant.

## Decision
The oracle answers every QUEUE_INSERT_TIE with `lastIndex` (the top of PS's range): an inserted action goes **after**
the queued actions it ties with. This is one of the outcomes PS itself can produce, so every trace remains a valid PS
trajectory. The engine inserts the same way (stable append order among equal keys). Trace headers list the
canonicalised kinds in `canonical_dontcare`.

## Consequences
- After the change: 600/600 battles (300 chaos, 300 random-doubles) replay exactly with a different PS seed, forcing
  only DrawKeys and tie records; 0 missed forced keys.
- Traces are no longer byte-identical to an uninstrumented PS run with the same seed when an insert tie occurs (the
  PRNG is not consumed for that draw). Nothing depends on that property.
- The weekly don't-care re-validation (07 §4.3) must run with this canonicalisation disabled for QUEUE_INSERT_TIE
  (`new InstrumentedPRNG({canonical: {}})`) to keep measuring its state impact.
