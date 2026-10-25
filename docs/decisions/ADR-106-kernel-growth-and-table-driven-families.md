# ADR-106: kernel-count growth with content; table-driven handler families

Status: **Proposed** (performance; no contract impact). Informs ADR-104's budget.

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
