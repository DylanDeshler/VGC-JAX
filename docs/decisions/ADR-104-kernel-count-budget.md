# ADR-104 — The engine is 12× over the 09 §1.1a kernel-count budget: options

- **Status:** option 3 approved and implemented 2026-10-23 (see Update); GPU measurement pending (human, L40S);
  option 4 decided after that. Originally: proposed 2026-10-22 — **needs a human decision** (the levers change 04 §12 bounds / the 04 §2 control
  structure, or the budget itself).
- **Affects:** 09 §1.1a (CPU proxy budget (a), enforced at G-v0 and G-v1 without a GPU measurement), 04 §2, 04 §12.

## Measurement
`bench/hlo_kernels.py` (port of research/jax-bench/hlo_kernels.py): kernel-like ops per env step in the optimised
HLO of the vmapped auto-reset rollout step (B = 256, core stage, train-like config: forced / draw_log off).

| build | kernel-like ops / env step |
|---|---|
| core, `EngineConfig.events=False` | **119,789** |
| core, events on (PublicEvents, 03 §4.7) | 138,172 |
| budget (09 §1.1a) | 10,000 |

Attribution (events off), by loop nest:

| part | trip | per trip | total | share |
|---|---|---|---|---|
| Block B (moves, NB = 8) | 8 | ≈ 11,170 | 89,336 | 75% |
| — Champions hit loop inside one move (`hit_loop`, 10 hits) | 10 | ≈ 700 | ≈ 6,980 per B trip (≈ 55,800 total) | 47% |
| — straight-line move / light / post_action code | | | ≈ 4,190 per B trip | 28% |
| Block A (light actions, NA = 10) | 10 | ≈ 2,600 | 25,960 | 22% |
| Block C, init/auto-reset, masks | | | ≈ 4,500 | 3% |

Every loop runs its full trip count under vmap (masked execution, J6), so kernels ≈ Σ body size × trip count. The
toy that calibrated the budget (≈ 980 ops) had no multihit loop and one action loop.

Tried and rejected: one-hot `mget`/`mset` over the 8-mon axis (−4%: fewer fusions, more copies; J12 also measured
it slower on CPU).

## Options
1. **Keep the structure, take the budget from a GPU measurement.** The CPU count is a proxy; XLA:GPU fuses
   differently. Rent an H100 for a day now (09 says request at M4.3) and measure env-steps/s directly. Cheapest;
   the risk is that the GPU result is also short of 3×10⁴ steps/s.
2. **Shrink the hit loop.** Hit 1 stays the full body; hits 2..N run a reduced body (damage, crit/roll draws,
   per-hit secondaries and contact effects only), with N = the stage's largest multihit count (10 in core:
   Population Bomb). Estimated −30 to −40%. Moderate risk; conformance (50k-battle runs) catches mistakes.
3. **Vectorise the target axis (09 J5).** The move body loops over 3 targets with `fori` (getDamage, spreadDamage,
   runMoveEffects, secondaries); J5 measured vmap 1.6× faster with one HLO copy. Estimated −20 to −30% of the move
   body. Moderate effort.
4. **Micro-step control structure.** Replace Blocks A/B/C with one `fori` over a single "execute the head entry"
   body (light and move merged, NA + NB trips) and make the hit loop a resumable micro-state, so each body
   evaluation executes one hit. Largest change (04 §2 rewrite); only this option plausibly reaches ~10k.
5. **Revise the budget** to what a full engine can reach (e.g. ≤ 40k after options 2 + 3) and gate on GPU
   throughput.

## Recommendation
Option 1 now (it decides whether anything else is needed), and options 2 + 3 as infrastructure work before G-v0,
since both stay within the existing 04 semantics and are verified by the conformance harness. Option 4 only if the
GPU measurement demands it.

## Update 2026-10-23: option 3 implemented
Target axis vectorised in the move body (accuracy, getDamage, spreadDamage, runMoveEffects, secondaries, forceSwitch,
immunity, timesAttacked): batched gathers, batched draws (`rng.chance_v` / `randint_v`), one masked select per field
(`util.mset_many`, `faint.faint_many`), PS message order kept by emitting afterwards; drain and secondary
self-boosts on the user are scalar recurrences. Conformance unchanged (2,000 core battles 100%, events identical
except the ADR-103 tie); distribution tests pass.

| build | before | after |
|---|---|---|
| events off | 119,789 | **74,684** (−38%) |
| events on | 138,172 | **86,317** (−38%) |

## Update 2026-10-23: GPU measurement (human's NVIDIA L40S, 46 GB; core stage, after option 3)
| batch | env-steps/s (events on) | events off | compile |
|---|---|---|---|
| 4,096 | 49,724 | 53,344 | 63–72 s |
| 16,384 | 40,026 | 42,533 | 71–82 s |
| 65,536 | 40,685 | 45,517 | 103–118 s |

GPU HLO: 55,553 kernel-like ops per step (29,624 fusions, 25,929 copies).
Reading: at B = 4,096 a batched step takes ≈ 82 ms ≈ 1.5 µs per kernel, i.e. launch-bound; from 16k up throughput is
flat (B = 65,536: 1.6 s per batched step), i.e. bound by the masked worst-case work itself (every battle pays for
8 move slots × 10 hits per step) and by loop-carried state copies. Above the 3×10⁴ minimum (stated for an H100),
below the 10⁵ target; compile time is near the 120 s budget with core content only, and every later stage adds
handler code inside the same loops.

## Update 2026-10-23: option 4 implemented (human approved after the L40S run)
Preparatory vectorisation (same technique as option 3): resolveFaints / clearVolatile over the whole faint queue,
`start` run once after the preview commit instead of inside every loop pass, phazing bounded to 2 flagged slots
(ERR_OVERFLOW_PHAZE): 76,414 → 32,243 kernel-like ops per step (events off).

Then the control structure: Blocks A/B/C replaced by one `fori` of `EngineConfig.micro_steps` (P, default 8)
micro-steps (vgcjax/battle/step.py `micro`). A micro-step selects the next queued action and runs it (light action,
residual, or a move's preludes + first hit), or runs one more hit of the move in progress, which lives in
`st.exec` between micro-steps (move_pipeline.py `move_begin` / `move_hit` / `move_end`); then the move tails, the
runAction tail and end of turn. A turn that needs more than P micro-steps continues in the next `step` call with
the new phase **CONT** (no decision for either player, legal masks pass-only, actions ignored); 08's API is
unchanged otherwise (`has_decision` is already False there). NA / NB and their overflow flags are gone.

| P | kernel-like ops per step (events off) | step calls ending at a decision | ops per decision |
|---|---|---|---|
| 1 | 5,356 | — | — |
| 4 | 9,691 | 0.61 | ≈ 15,900 |
| 6 | ≈ 12,400 | 0.81 | ≈ 15,300 |
| 8 (default) | 15,231 | 0.99 | ≈ 15,400 |

Per decision: 119,789 (original) → 32,243 (vectorised blocks) → ≈ 15,400 (micro-steps). CPU (B = 256, machine
load 70–120 from other users' jobs, so indicative only): 5,215 decision-steps/s at P = 8 vs 2,930 before.
Conformance: contract v2 corpus replayed at P = 1, 2 and 8: 100% CBR/TMR/MPR, events identical (spill-over is not
observable; pinned by tests/harness/test_microsteps.py). Distribution tests pass.

## Update 2026-10-23: L40S after option 4 (human's run of bench/gpu_check.sh)
| batch | micro_steps | decision-steps/s, events on | events off |
|---|---|---|---|
| 4,096 | 8 | 408,963 | 436,029 |
| 16,384 | 8 | 786,656 | 965,077 |
| 65,536 | 8 | 942,405 | 1,351,429 |
| 65,536 | 4 | 865,993 | 1,130,854 |

GPU HLO 5,871 kernel-like ops per step (budget 10,000); compile 54–66 s. ≈ 20–30× the pre-option-4 L40S result
(40–45k), ≈ 10× the 09 §1 target, core stage, random policy, short timing windows. Default micro_steps = 8.
G-core re-run under contract v2 on this engine (`gcore-2026-10-23-v2`, 50,000 battles): PASS, contract verified.
