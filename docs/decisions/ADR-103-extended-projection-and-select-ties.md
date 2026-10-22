# ADR-103 — Extend the projection with hidden engine state; record SELECT ties of replacement switches

- **Status:** proposed 2026-10-22 — **needs human approval** (changes protected `harness/contract/projection_fields.json`,
  `normalisations.json` and `docs/design/contract/drawkey.md` §4; ADR-029). Bumps `contract_version` to 2.

## Problem 1: state outside the protected projection is unchecked
07 §5.1 compares HP, stats, boosts, status, moves, … but not engine state that later mechanics read:
`activeTurns`, `activeMoveActions` ([C] Fake Out mask), cached `pokemon.speed` (handler and switch-in ordering),
`moveThisTurn`, `lastMoveUsed`, `lastMoveTargetLoc`, `moveThisTurnResult` / `moveLastTurnResult` (Stomping Tantrum),
`hurtThisTurn`, `timesAttacked` ([C] Rage Fist), switch flags, `isStarted`, `newlySwitched`, `previouslySwitchedIn`,
the per-turn stat/item flags, ability/item `effectOrder` (RedirectTarget, SwitchIn ties), side `totalFainted` /
`faintedThisTurn` / `faintedLastTurn`, and battle `effectOrder` / `midTurn` / `lastSuccessfulMoveThisTurn`.
A divergence there stays invisible until it changes a compared field, possibly many turns later.

### Evidence
The oracle and harness now emit these as non-gating `ext` fields (oracle/lib/project.js, harness/project.py) and
report mismatches separately (`ext_mismatch_battles` in summaries). The first run (300 core battles) found two real
engine bugs the protected projection could not see:
- `lastMoveTargetLoc` for no-target actions: PS's resolveAction assigns getLocOf(getRandomTarget()) at commit
  (sim/battle-queue.ts:268-272); the engine kept 0 (266/300 battles). Fixed.
- `lastSuccessfulMoveThisTurn` is reset to null by a failed move (sim/battle-actions.ts:366); the engine kept the
  previous value (16/300). Fixed.
After the fixes: 3,000 fresh core battles, all protected checks clean, extended fields equal except the 4 battles
of Problem 2.

### Proposal
Add the `ext` fields to `projection_fields.json` with one normalisation: `lastMoveTargetLoc` is not compared when
the mon's last move has a random-resolved target (randomNormal / spread / side / allies / scripted: a don't-care foe
pick, contract §6) or when `lastMove` is null (the location survives clearVolatile, its origin is unknown).

## Problem 2: unrecorded SELECT ties are not state-irrelevant
Contract §4 records SPEED_TIE_SELECT only among entries whose actor is active and not fainted. End-of-turn
replacement `instaswitch`es have fainted outgoing actors, so a tie between two of them (one side, both slots
fainted, equal stored Speed) is shuffled by PS but never recorded; the engine uses insertion order.

### Evidence
4 of 3,000 core battles (`reports/conformance/ext2`): the two incoming mons' ability `effectOrder` values are swapped
(e.g. 13/14 vs 14/13). effectOrder breaks Lightning Rod / Storm Drain / Follow Me RedirectTarget ties and SwitchIn
hazard order (04 §6.4 rule 2), so it will become state-relevant in v0. `final.diag.unrecordedSelectTies` in traces
counts these: 0.8–2% of battles. Under the heuristic policy (more faints per game): 7 of 2,000 core battles
(`reports/conformance/heur1`), including ties between replacements on *different* sides.

### Proposal
In contract §4 SPEED_TIE_SELECT: for `switch` / `instaswitch` actions, do not filter tie members by actor
liveness (the switch executes regardless of the outgoing mon's state, sim/battle-actions.ts:79-215). Keep the filter
for move / beforeTurnMove / priorityChargeMove (those early-return). Members are the outgoing actors' refs, as now.

## Migration
contract_version 2; regenerate corpora (fixtures: none yet). Engine: queue.select uses the new membership rule;
oracle: instrument.js selectTied likewise.
