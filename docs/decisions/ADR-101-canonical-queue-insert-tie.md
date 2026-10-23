# ADR-101 — The oracle answers QUEUE_INSERT_TIE canonically

- **Status:** diagnosed 2026-10-23; revised recommendation (record the tie, option b) **accepted by the human 2026-10-23**; ships with contract_version 2.
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

## Diagnosis (2026-10-23, at the human's request)
**The 1/424 "state change" was a harness artifact, not mechanics.** Chaos battle 969 (seed reval-2): both leads tie
on Speed, the free insertion puts p2's runSwitch first, so the batched SwitchIn step's `step_actor` becomes 4
instead of 0. The recorded ABILITY_TRACE_TARGET draw (Gardevoir's Trace) is keyed with step_actor 0, misses
(`forcing.missed`), and re-rolls naturally: Trace copies Emergency Exit instead of Rock Head. Insertion order itself
changed nothing in that battle.

**But insertion order is mechanically relevant in content after core.** Every `insertChoice` site (PS
sim/battle-queue.ts:375-406 draws `random(firstIndex, lastIndex + 1)` among equal-priority actions):
1. `runSwitch` (sim/battle-actions.ts:211): the head of a batched runSwitch is the only mon PS runs
   `EmergencyExit` for afterwards (sim/battle.ts:2932-2934). Two equal-Speed switch-ins, one with Emergency Exit /
   Wimp Out dropped below half by switch-in damage: whether it activates depends on the insertion.
2. Mega Evolution re-inserts the mon's move at its new Speed (sim/battle.ts:2913-2918): an equal-Speed tie with
   another queued move decides move order.
3. Encore in Champions replaces the target's action via changeAction -> insertChoice
   (data/mods/champions/moves.ts:330): same, move order.
4. `beforeTurn` (sim/battle.ts:3002): no pokemon; ordering against other order-class entries has no effect.
Core has none of 1-3 (no Emergency Exit damage on switch-in, no Mega, no Encore), which is why the core-only
re-validation saw no real impact. M3.6 (Mega) makes case 2 reachable immediately.

## Revised recommendation: option (b), scoped
Make the insertion a **recorded tie** (like SPEED_TIE_*): new tie kind `QUEUE_INSERT_TIE` recorded when
firstIndex < lastIndex, keyed by (turn, step_tag, sub = per-turn insertion ordinal, ord = 0) with members = the tied
queue entries' actor refs plus the inserted one, value = the insertion offset. The engine draws it naturally with
PS's distribution (uniform over the range) and replays it forced; the key does not use step_actor, so it cannot
leak. The oracle stops canonicalising (this ADR's original decision is withdrawn). `beforeTurn` insertions have
no actor and stay don't-care. Contract change (drawkey.md §4/§6, dontcare.json): fold into contract_version 2
with ADR-103. **Needs human approval.**
