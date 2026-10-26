# ADR-105: fractional randomChance numerators in draw-argument comparison

Status: **Accepted** (human-approved 2026-10-26; touches the draw comparison of contract §5/§9).

## Context
Multi-accuracy moves (Population Bomb, Triple Axel) re-roll accuracy on hits ≥ 2 with PS's float stage table
([C] `scripts.ts:480-503`), so PS calls `randomChance(67.5, 100)`. The trace records `args: [67.5, 100]`;
`harness/forced.py compared_draws` truncates with `int()` to `(67, 100)`. The engine draws with the exact integer
bound `ceil(67.5) = 68` (`random(100) < 67.5 ⇔ r < 68`) and logs `(68, 100)`. Outcome sets are identical; only the
logged argument differs, giving a `DRAW_ARGS:ACCURACY` bucket (first seen: 1 of 300 impl battles, ab1 #137).

## Options
1. **Canonicalise PS's fractional numerator to `ceil`** in `compared_draws` for `randomChance` (exactly equivalent:
   same outcome for every draw value). Recommended.
2. Make the engine log `floor` while drawing with `ceil` (hides the true argument; not recommended).
3. Leave as a known bucket.

## Decision
Option 1, approved by the human on 2026-10-26. PS's `randomChance(n, d)` is `random(d) < n` with integer rolls
(`sim/prng.ts:91-117`), so `n = 67.5` succeeds for rolls 0-67, exactly `< ceil(n) = 68`; `floor` (67) would differ
at roll 67. `harness/forced.py compared_draws` now compares `randomChance` numerators as `ceil(n)`; the engine
already draws with that bound. `random` / `sample` arguments are unchanged (PS floors them itself).
