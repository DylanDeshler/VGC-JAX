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

## Re-validation evidence (2026-10-22, oracle/tools/revalidate_dontcare.js, 07 §4.3)
4,000 battles (2,000 chaos-stats teams, 2,000 random-doubles teams), replayed in PS with a different seed, forcing all
compared draws and ties, leaving one don't-care kind free at a time (other don't-care draws forced from the
recording):

| kind | battles containing it | state changed |
|---|---|---|
| RANDOM_TARGET_NOMINAL | 3,101 | 0 |
| RANDOM_TARGET_NOEFFECT | 3,449 | 0 |
| SELF_EFFECT_ROLL (chance undefined) | 2,522 | 0 |
| SECONDARY (chance undefined) | 46 | 0 |
| **QUEUE_INSERT_TIE** | **424** | **1** (chaos battle 969: a different ability on a mon at decision 1) |

So QUEUE_INSERT_TIE is **not** strictly state-irrelevant: the canonical answer of this ADR is required for the engine
to reproduce the oracle at all, and PS's natural distribution (random insertion) is not reproduced by the engine in
the rare cases where it matters. Options for the human: (a) keep this ADR (canonical insertion everywhere; real PS
games may very rarely diverge from engine physics), or (b) promote QUEUE_INSERT_TIE to a forced kind in contract §5
(engine keys and draws the insertion position). Recommendation: (a) for now; revisit if v0 re-validation shows a
higher rate.
