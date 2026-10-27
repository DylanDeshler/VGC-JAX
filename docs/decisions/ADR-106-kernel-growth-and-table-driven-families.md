# ADR-106: kernel-count growth with content; table-driven handler families

Status: **Pilot approved** (human, 2026-10-26; performance, no contract impact). Informs ADR-104's budget.

## Measurements (impl stage, bench/hlo_kernels.py, events on unless noted)
| point | kernel-like ops / step |
|---|---|
| after Protect family (9e39bf5) | 82.6k |
| + weather / terrain (78f84f5) | 102.7k |
| + ability batch 1 (eca6bd3) | 199k |
| + ability batch 2 (cae3816) | 315k |
| undispatched families (7edc8f3) | 135.8k |
| + ability batch 3 (4cba98b) | 154.7k |
| + ability batch 4, reaction boosts / statuses batched | 191.6k (167.5k events off) |

## Findings
1. `lax.switch` under `vmap` lowers to "run every branch, select": a dispatched handler costs its whole body for
   every env, and dispatch nested inside dispatch multiplies (the 315k -> 136k fix).
2. The masked-pass replacement removes the multiplication but still costs **one copy of the logic per entity**
   (a Python loop over abilities / items emits ops per entry). Cost grows linearly with implemented content:
   roughly +0.6-1.2k ops per ability so far. Extrapolated to all of v0 (~210 abilities, ~170 items, ~470 moves'
   bespoke code) this lands in the 400-600k range.
3. Events cost ~13% (167k vs 191k); the logic dominates.

## GPU measurement (L40S 46 GB, commit 1f54497, bench/gpu_check.sh, run by the human 2026-10-26)
| batch | micro_steps | env-steps/s (events on / off) | decision-steps/s (on / off) | compile + first call |
|---|---|---|---|---|
| 4,096 | 4 | 23,307 / 25,970 | 15,089 / 16,814 | 784 s / 583 s |
| 4,096 | 8 | 12,452 / 13,807 | 12,394 / 13,743 | 809 s / 589 s |
| 16,384 | 4 | 18,398 / 19,558 | 11,908 / 12,659 | 828 s / 602 s |
| 16,384 | 8 | 9,624 / 10,190 | 9,580 / 10,144 | 854 s / 629 s |
| 65,536 | 4 | 17,869 / 19,201 | 11,566 / 12,428 | 898 s / 683 s |
| 65,536 | 8 | 9,318 / 4,300 (outlier) | 9,274 / 4,279 | 1,021 s / 1,113 s |

GPU HLO: 90,488 kernel-like ops per env step (B = 16k, micro_steps 8, events off).

Reading: time per step call grows linearly with the batch (4k -> 64k: 0.18 s -> 3.7 s per call), so the GPU is
saturated by per-env work at B = 4k — the step is work-bound, not launch-bound as 09 §1 assumed. Throughput is
therefore set by the total work every env executes per step, which is exactly what grows with each implemented
handler (finding 1). Against 08 §6 / 09 §1: the B = 16k-64k minimum (3x10^4 env-steps/s, H100) is missed on the
L40S (~1.9x10^4); the search-batch minimum (2x10^4 at B = 1k-4k) is met (2.3-2.6x10^4); compile (10-15 min) is
5-8x the R4 trigger (120 s). With ~1/3 of the format implemented, more content will lower throughput further.

## Options
1. **Table-driven families** (recommended): express each handler family as tables indexed by the entity id plus
   one generic implementation (e.g. on-hit status proc: `proc_status[ability]`, `proc_chance[ability]`;
   status immunity: a bitmask `immune_statuses[ability]`; type-boost items: `boost_type[item]`; absorb abilities:
   `absorb_type[ability]`, `absorb_kind[ability]`). Cost per family becomes O(1) in the number of entities. Most
   v0 abilities / items fall into ~20 such families; the genuinely bespoke rest stays as masked code.
2. Accept linear growth; re-measure GPU throughput on the L40S at each milestone.
3. Environment bucketing by active handler family (gather / scatter subsets of the batch): data-dependent shapes,
   fights jit; not recommended.

## Decision
Pending. Until decided, new families are written with option 1 in mind (tables where the shape fits) and the
existing ability / item families will be converted in one pass after the v0 content lands.

## Pilot (approved 2026-10-26): profile first
`bench/hlo_attrib.py` charges every counted HLO op to its source stack (the optimised module's stack-frame table)
and reports self and inclusive cost per engine function. Impl stage, CPU, B = 256, events on, 300,594 ops:

| inclusive area | share |
|---|---|
| fieldEvent('SwitchIn') via switch_in / drag_in (run_switch compiled ~7x: run_light, and 2 drags in each of 3 post_action copies) | ~25% |
| move_begin / move_hit (spread_move_hit) | ~20% each |
| boost / boost_v / try_boost / emit_boost (many call sites) | ~13% |
| resolve_faints (many call sites) | ~9% |
| encore_start alone (copied into every add_volatile Start dispatch) | 7.9% |
| mget / mset / mset_many (self) | ~29% |

Finding: the cost is dominated by **generic machinery duplicated at every call site** (every call site of a heavy
function is inlined, and under vmap every copy executes every step), not by per-entity logic. The originally
proposed families (type-boost items, resist berries, status immunities) are already table-driven and cost ~0.1%.

Change 1 (pilot): `add_volatile` dispatches Start / Restart / durationCallback only over the conditions its call
site can name (a python-int condition: itself; move-data callers: the values present in move_volatile /
move_self_volatile / move_sec_volatile, `CO.vol_candidates`). **300,594 -> 255,226 ops (-15%)**; targeted and
random corpora unchanged (100%).

Next candidates, by measured share: (1) one switch-in execution per micro-step (drag-in resumes through the shared
run_light switch path instead of an inline runSwitch in every post_action copy); (2) fewer boost / resolve_faints
call sites (batch per action); (3) the same static-dispatch narrowing for other single_event call sites.
