# JAX architecture patterns for the VGC engine (measured)

Companion to `jax-vgc-engine-scope.md` §3 and §7. Every number here comes from scripts in
`research/jax-bench/`. Raw JSON and logs are in `research/jax-bench/results/`, and `summarize.py` regenerates the tables.

## 0. Setup and method

- **Hardware:** Apple M4 Max (10 performance + 4 efficiency cores, 36 GB).
- **Software:**
  - JAX/jaxlib **0.12.0.1** on the XLA:CPU backend, Python 3.12, flax 0.12.10, chex 0.1.92.
  - jax-metal 0.1.1 installs trivially, but only next to jax 0.4.34, so it lives in a separate venv `.venv-metal` (§8.4).
- **Method:**
  - Each variant is a fresh closure.
  - Timing is split into `jax.jit(f).lower(...)` (tracing), `.compile()` (XLA), and steady state: the median of ≥3 runs after warm-up, with `block_until_ready`.
  - "ns/elem" is wall time divided by the batch size B. Memory is `compiled.memory_analysis()` plus peak RSS.
  - Benchmarks ran one at a time. Configurations that can hang compile in a subprocess with a timeout.
- **What CPU numbers mean:**
  - XLA:CPU splits large fused loops across threads: cpu/wall ≈ 6 at B ≥ 64k, about 70 G simple int32 ops/s, and about 200 GB/s for streaming copies.
  - Fixed costs are about 7 µs per jit call, about 3 µs per XLA:CPU thunk, and about 30 µs per loop iteration with a realistic body. At B ≈ 4k they dominate, which is why some small-B rows look "slow".
  - Ratios between variants transfer better than absolute numbers (§8).

---

## TL;DR: recommendations

1. **Dispatch effects mostly through tables, not `lax.switch`.**
   - Under `vmap`, a switch with a batched index runs **every branch** and then a `select_n`.
   - Measured cost is linear in total branch work: 3 → 8 → 24 → 105 ns/elem for N = 8 → 512 handlers.
   - A parameter-row "template interpreter" plus N/8 bespoke branches costs 5 → 29 ns at N = 512 (3.7× faster) and compiles 5× faster.
2. **Never use one-hot-multiply or `jnp.take` over stacked branch outputs.**
   - Both materialise an [N, B] intermediate: 1.2–1.5 GB of temp at N = 512, and 3–50× slower than `switch`.
3. **One small switch per hook, holding only the effects that have that hook.** Put a **`lax.optimization_barrier` after each hook dispatch.**
   - Without barriers, XLA fuses consecutive select trees into one giant fusion.
   - 8 hooks × 65 branches: compile **24.5 s → 1.7 s**, runtime **795 → 93 ns/elem**.
   - A naive "switch over every effect at every hook" costs 2,528 ns/elem (27× worse).
4. **Never put a big switch in an unrolled loop.** Keep the action-queue loop rolled (`fori_loop` = `scan` with static bounds). Vectorise the 3-target axis with `vmap`; don't unroll it.
   - A 128-way switch in fori(8) × unrolled(3): **37.8 s** compile.
   - Fully unrolled 8 × 3 with a 32-way switch: **> 90 s**.
   - Kept rolled, the N = 128 nest compiles in 1.2 s, or 0.65 s with barriers.
   - In the toy battle step, unrolling the queue gave **no** runtime gain and cost **3× compile time**.
5. **`while_loop` buys nothing under vmap.** With per-battle trip counts 4–8 it ran at max-trip cost plus select overhead (459 ns vs 392 ns for a masked fori). Use fixed-trip loops plus masks.
6. **State:**
   - Store int8/int16 and compute in int32. Native int8/int16 arithmetic was no faster on CPU (5.3–5.9 vs 2.7–5.1 ns per damage calc) and overflows PS math.
   - Keep the pytree around **≤ 100 leaves.** Per-leaf costs show up in standalone `tree_where` (1,024 leaves is 7× slower than 1 packed array) and in Python dispatch (≈ 1 µs/leaf/call).
   - Inside one jit, `tree_where(mask, new, old)` is nearly free for untouched leaves, because XLA folds `select(p, x, x)`.
   - Use a NamedTuple or `flax.struct.dataclass`; they are identical under jit.
   - Store volatiles as a `uint32[2]` bitfield per mon: 8 B and 2.1 ns/mon, vs 64 B and 41 ns for `bool[64]`.
7. **RNG:**
   - Threefry is expensive on CPU: 12–22 ns per draw, about 2–3 µs per battle-turn for 144 draws, as much as the whole toy step.
     - The cause is JAX's CPU-only *rolled-loop* threefry lowering, so this cost does **not** transfer to GPU (§8).
     - Switching the default impl to `philox2x32` made the whole CPU rollout 1.8–2.0× faster.
   - Generate one bulk `uniform(fold_in(key, turn), (A, T, E))` per turn, or use a counter-based integer hash of (seed, turn, actor, target, event).
   - **Do not use `rbg` / `unsafe_rbg` with per-battle keys.** Under `vmap` they generate the whole batch from the *first* key (JAX docs, verified here), which breaks per-battle replay.
   - Bulk philox2x32 was the fastest library PRNG at 0.72 ns/draw, and a 2-round integer hash cost 0.21 ns/draw. But philox with 144 `fold_in`s did not finish compiling in 150 s.
8. **Rollout:** one `jit(scan(step))` with an auto-reset gather from a pre-built initial-state bank, `donate_argnums` on the state, and the persistent compilation cache on.
   - The toy rollout (6-action queue × 3 targets × 200-op damage calc + 64-way switch, T = 128) compiles in **1.2 s** (0.1 s from a warm cache).
   - It runs at **0.43 M env-steps/s** on one XLA:CPU device at B = 32k.
   - Sharded over 10 virtual CPU devices it reaches **1.38 M env-steps/s**.
   - Memory is about 1.5 KB of XLA temp per battle.
   - With 8 extra 64-way hook points per action, compile goes **28.7 s → 2.5 s** with barriers and runtime gets 2.1× faster.
   - The H100 estimate is launch-bound: about 1,000 kernels per step, about 10–40 M steps/s for the toy at B = 64k, and about 1–10 M for the real engine (§8).
9. **Dev workflow:** see §6:
   - config-gated handler subsets, so a dev build only compiles the effects in the test teams;
   - barriers;
   - the persistent cache;
   - `disable_jit` for stepping one battle;
   - `checkify` only in test builds;
   - differential tests run *jitted* over batches of scenarios.

---

## 1. Effect dispatch under vmap  (`b1_dispatch.py`, `handlers.py`)

Handlers are synthetic but PS-shaped. Each one maps a 9-field int32 context to a new context, using 6 templates with per-handler constants, 10–40 int ops each:
- Technician-like BP gate;
- Sand Force-like weather + type multiplier;
- Tough Claws-like flag gate;
- Blaze-like pinch;
- Gyro Ball-like stat ratio;
- Leftovers/poison-like residual.

All arithmetic is PS `modify`/`chainModify` 4096-based integer math.

### 1A. One hook call, N handlers (B = 65,536; baseline copy of the context = 0.9 ns/elem)

Runtime, ns/elem:

| variant | N=8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| (a) `lax.switch` (batched idx → select over all branches) | 3.0 | 7.9 | 24.4 | 105.2 |
| (b1) compute all + one-hot multiply-sum | 3.3 | 34.2 | 694.4 | 5,492 |
| (b2) compute all + `stacked[idx]` (gather) | 9.0 | 21.2 | 73.9 | 337.5 |
| (c) table rows (template id + 4 consts) + N/8 special branches | 5.4 | 5.4 | 8.5 | 28.8 |

Compile time, s:

| variant | N=8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| switch | 0.05 | 0.14 | 0.55 | 2.70 |
| one-hot | 0.06 | 0.14 | 0.42 | 2.14 |
| take | 0.05 | 0.12 | 0.43 | 2.54 |
| table + N/8 specials | 0.08 | 0.08 | 0.15 | 0.50 |

Temp memory, MB:

| variant | N=8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| switch | 0 | 2 | 7 | 11 |
| one-hot / take | 0–1 | 75 | 300–384 | 1,216–1,536 |
| table | 2 | 2 | 4 | 7 |

- **`switch` under vmap is "compute everything, then select", but fused.** XLA temp memory stays at 2–11 MB even at N = 512, so the branch bodies and the `select_n` tree evidently stay fused and nothing [N, B]-sized is materialised. Cost is about 0.2 ns per branch per element here, i.e. linear in total branch FLOPs.
  - The JAX docstring for `lax.switch`/`cond` says: "when transformed with `vmap` to operate over a batch of predicates, `cond` is converted to `select`."
- **One-hot and take are strictly worse.** Stacking the branch outputs forces the [N, B] intermediate into memory.
- **Tables flatten the curve.**
  - The interpreter evaluates the 6 templates with *gathered* constants and selects by template id. Its cost depends on the number of templates, not on N.
  - Only the residue that can't be expressed as a row (here N/8) pays per-branch cost.
  - This mirrors minimon's `ModifierTable` and effect-kind enum (§7), and the scope doc's §3.4.

### 1B. Eight hook points per turn (B = 65,536)

N effects are spread over 8 hooks (on-BP, modify-atk, on-hit, residual, …). Each hook is invoked once.

Runtime, ns/elem:

| variant | N=8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| naive: switch over all N effect ids at each hook (identity if the effect lacks that hook) | 4.3 | 10.7 | 83.6 | 2,528 |
| grouped: per-hook remap table → (N/8 + 1)-way switch | 4.9 | 16.2 | 87.1 | 795 |
| **grouped + `lax.optimization_barrier` after each hook** | 6.7 | 13.0 | **32.8** | **93.2** |
| table rows per hook + per-hook special switch | 23.6 | 31.1 | 42.9 | 97.1 |

Compile time, s:

| variant | N=8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| naive | 0.06 | 0.27 | 2.59 | 27.1 |
| grouped | 0.06 | 0.23 | 1.91 | 24.5 |
| grouped + barrier | 0.06 | 0.13 | 0.44 | 1.73 |
| table per hook | 0.45 | 0.63 | 0.73 | 1.95 |

**Key finding: chained batched switches are superlinear unless fenced.**
- The output of one hook's select tree feeds every branch of the next hook's switch.
- XLA's fusion pass merges them into one huge elementwise fusion. The LLVM backend then struggles, which shows as compile time, and the generated code most likely duplicates work or spills registers, which shows as runtime. That is my reading; I did not inspect the generated assembly. The optimised-HLO text for the rollout grows from 6 MB to 21 MB between 4 and 8 un-fenced hooks (§5).
- A standalone check (`b1_barrier_chain.py`, B = 16,384) with k chained 65-way switches gave:

  | k | no barrier: compile / runtime | barrier: compile / runtime |
  |---:|---|---|
  | 1 | 0.26 s / 20 ns | 0.25 s / 19 ns |
  | 2 | 0.84 s / 39 ns | 0.39 s / 34 ns |
  | 3 | 1.95 s / 75 ns | 0.54 s / 46 ns |
  | 4 | 3.67 s / 164 ns | 0.69 s / 61 ns |

  Without barriers, compile time and runtime both grow faster than k.

- `lax.optimization_barrier(ctx)` after each hook forces materialisation of the 9 small context fields, a few bytes per battle, and restores linear scaling.
- On GPU the same fence splits one mega-kernel into a few kernels. That is cheaper than register spills at this size, but **re-measure on the H100** (§8).

### 1C. Switch nested in the turn loops: fori(8 actions) × (3 targets)

B = 4,096. Effect id per (action, target).

Compile time, s:

| loop nest | N=8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| fori8 / fori3 | 0.11 | 0.28 | 1.17 | 5.10 |
| fori8 / fori3 + barrier | 0.06 | 0.20 | 0.65 | 2.61 |
| fori8 / unroll3 | 0.26 | 2.63 | **37.8** | **> 90 (timeout)** |
| fori8 / unroll3 + barrier | 0.14 | 0.44 | 1.86 | 9.75 |
| unroll8 / unroll3 | 3.82 | **> 90** | > 90 | > 90 |
| unroll8 / unroll3 + barrier | 0.72 | 1.81 | 9.23 | 75.2 |

Runtime, ns/elem (B = 4,096, so XLA:CPU loop overhead of ≈ 30 µs per iteration dominates at small N):

| loop nest | N=8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| fori8 / fori3 | 239 | 739 | 2,409 | 5,925 |
| fori8 / unroll3 + barrier | 256 | 640 | 1,983 | 5,520 |
| unroll8 / unroll3 + barrier | 445 | 588 | 956 | 2,355 |

- **Rolled loops compile each switch once.** Compile time grows linearly in N: 5 s at N = 512, 2.6 s with barriers.
- **Every unrolled copy of a switch multiplies compile time,** and without barriers the copies chain superlinearly.
- **Full unrolling with barriers is the fastest at runtime** (2.5× at N = 512, because it removes 24 loop iterations), but it costs 75 s of compile. It is not worth it in development.
  - On GPU the loop-overhead side of this trade grows (each iteration is several kernel launches), so a **final "release" build might unroll the 3-target axis.** Measure there.

### Recommendations for dispatch

1. **Table first.** Export everything expressible as a row from `@pkmn/dex` into dense int16 arrays:
   - BP / priority / flags / target / secondary columns for moves;
   - modifier rows for items and abilities, as (hook, condition-kind, condition-arg, multiplier, …).
   
   Write a handful of generic "template" evaluators per hook. Expect 70–85% of Champions effects to fit; measure as the dex is exported.
2. **Residue goes into per-hook switches** (`HOOK_ON_BP`, `HOOK_MODIFY_ATK`, `HOOK_ON_HIT`, `HOOK_RESIDUAL`, `HOOK_SWITCH_IN`, …).
   - Index through a static remap table `local_idx[hook, effect_id]`, with 0 = no handler.
   - Size each switch by the effects that actually have that hook.
3. **Fence each hook dispatch with `lax.optimization_barrier`.** Don't let selects from different hooks chain.
4. **Budget.**
   - A fenced 64-way hook invocation costs ≈ 12–17 ns per battle on CPU. This was measured both in 1B and in the full rollout (+8 hooks per action: 369k → 285k steps/s).
   - It adds ≈ +0.15 s of compile per hook point.
   - With N_total ≈ 100–300 bespoke branches over about 8–12 hooks, each invoked 6–24× per turn, that is ≈ 1–3 µs per battle-step on CPU (≈ 100× less on H100) and a few seconds of compile. That is fine.
5. **The 3-target axis should be a vectorised axis inside the action body,** not an unrolled loop around the switch. In the vectorised form the switch appears once in HLO.

---

## 2. Control-flow structure  (`b2_control.py`, `toy_battle.py`)

### 2A. Synthetic: 8 actions × 3 targets × ~200-op integer damage calc (B = 16,384)

The HP update uses `.at[tgt].add` (scatter) unless noted. "while" uses a per-battle trip count in 4–8.

| outer (8) | inner (3) | compile s | ns/elem |
|---|---|---:|---:|
| fori | fori | 0.11 | 619 |
| fori | unroll | 0.15 | 391 |
| fori | **vector (vmap over 3 targets)** | **0.09** | **392** |
| scan | fori / unroll / vector | 0.10 / 0.15 / 0.09 | 601 / 386 / 407 |
| unroll | fori / unroll / vector | 0.37 / 0.66 / 0.31 | 523 / 326 / **291** |
| while (variable trip, max 8) | fori / unroll / vector | 0.14 / 0.19 / 0.15 | 536 / 476 / 459 |
| fori | scan | 0.13 | 677 |
| fori (no masking) | vector | 0.10 | 412 |
| fori / unroll | vector, **one-hot** HP update | 0.21 / 0.58 | 1,895 / 470 |

- **`fori_loop` and `scan` are the same thing.**
  - With static bounds `fori_loop` *is* a `scan`. The docstring says: "If the trip count is static … the `fori_loop` is implemented in terms of `scan`."
  - Runtime and compile time match within noise. Use whichever reads better; `scan` when you want stacked per-iteration outputs.
- **Vectorising the target axis ≈ unrolling it,** and it is 1.6× faster than an inner `fori`. It also keeps one copy of the body in HLO, which matters with switches (§1C).
  - Semantics: all three targets see the pre-move HP, which is what PS does for spread moves (damage is computed per target and applied together). Sequential per-target effects (e.g. a Rocky Helmet chain) go in a small post-pass.
- **Unrolling the 8-action loop** saved 25% on this small body but tripled compile time. On the realistic step (2B) it saved nothing.
- **`while_loop` under vmap runs to the max trip count** of the batch and selects per lane. It was slower than a masked fixed-trip fori.
  - It is also not reverse-differentiable. We don't need that, but a fori with static bounds keeps the option.
  - On GPU a data-dependent `while` predicate also needs a device→host sync per iteration (§8).
- **Scatter vs one-hot:** one-hot masks over the target axis were 1.6–4.8× slower than scatter in the synthetic, and within ±12% in the toy step (§2B, §5). Keep `.at[idx]` updates with batched indices.

### 2B. The toy battle step (B = 8,192)

The step has:
- a 6-entry queue with dynamic-speed argmax;
- the 3-target axis with a ~200-op damage calc;
- a 64-way secondary-effect switch;
- a 16-way item residual switch;
- faint replacement and terminal check.

| queue loop | targets | indexing | trace s | compile s | ns per battle-step |
|---|---|---|---:|---:|---:|
| fori | vector | gather/scatter | 0.43 | 0.90 | 2,493 |
| fori | vector | one-hot | 0.34 | 0.73 | 2,561 |
| fori | unroll | gather/scatter | 0.36 | 0.83 | 2,035 |
| fori | unroll | one-hot | 0.38 | 0.85 | 1,900 |
| unroll | vector | gather/scatter | 0.88 | 2.40 | 2,224 |
| unroll | unroll | gather/scatter | 1.07 | 3.05 | 2,519 |

- Differences are within about ±15%, which is near the noise floor for this machine.
- **Unrolling the queue triples compile time for no gain.** Keep the queue rolled.
- The target-axis choice is a wash at this size. Prefer `vector` because it composes with switches (§1C).

---

## 3. State representation  (`b3_state.py`)

### 3A. `tree_where(mask, new, old)` standalone vs leaf count

Setup: a 2 KB/battle int16 state split into F leaves, B = 16,384, so each call reads 2 states and writes 1 (96 MB).

| leaves F | 1 | 4 | 16 | 64 | 256 | 1,024 |
|---|---:|---:|---:|---:|---:|---:|
| ns/battle | 28.5 | 42.7 | 41.5 | 43.8 | 69.9 | 208 |
| effective GB/s | 216 | 144 | 148 | 140 | 88 | 30 |
| compile s | 0.02 | 0.02 | 0.04 | 0.03 | 0.06 | 0.19 |

Up to about 64 leaves the cost is bandwidth. Beyond that, a per-leaf fixed cost of about 3 µs per leaf per call shows up: separate loop/fusion thunks on CPU, separate kernels on GPU unless XLA horizontally fuses them.

### 3B. Masked merge inside a step (F = 256 leaves, k modified)

| k leaves modified | 1 | 8 | 64 | 256 |
|---|---:|---:|---:|---:|
| ns/battle | 38 | 41 | 55 | 100 |

- **XLA folds `select(p, x, x) → x`.** So `state = tree_where(mask, new_state, state)` over the *whole* tree costs only the leaves the step actually changed, provided untouched leaves are the same traced value (`_replace` / `.replace` keep them).
- **Write handlers so they only touch the fields they change.** Don't round-trip every field through arithmetic.

### 3C/3F. Python-side cost per jit call (B = 1)

Python dispatch overhead per call, by leaf count:

| leaves | 1 | 16 | 64 | 256 | 1,024 |
|---|---:|---:|---:|---:|---:|
| µs per jit call | 6 | 23 | 74 | 268 | 1,160 |

Container type, 64 leaves:

| container | flatten + unflatten µs | jit call µs |
|---|---:|---:|
| NamedTuple | 4.5 | 65 |
| dict | 7.5 | 67 |
| `jax.tree_util.register_dataclass` | 14.1 | 75 |
| `flax.struct.dataclass` | 14.2 | 75 |
| `chex.dataclass` | 17.7 | 79 |

- **Under `jit` all containers are identical.** They only differ at the Python boundary.
- **The per-leaf call overhead (≈ 1 µs/leaf) only matters when `step` is called from Python.** In training the rollout is one `scan` (§5), so it is paid once per rollout.
- It does argue for **≈ 50–100 leaves**: group per-mon scalars into a few `[2, 6, k]` int16 arrays by dtype and access pattern rather than having 70 separate `[2, 6]` leaves. Keep named accessors (properties or index constants) for readability.
- **Recommendation:** use a `NamedTuple` or `flax.struct.dataclass` (`.replace`, defaults, `pytree_node=False` for static metadata). Avoid `chex.dataclass` in hot Python paths.

### 3D. dtypes on CPU (B = 16,384)

Bandwidth-bound masked copy of [B, 1,024]:

| dtype | int8 | int16 | int32 | float32 |
|---|---:|---:|---:|---:|
| ns/battle | 18 | 29 | 58 | 54 |
| GB/s | 167 | 209 | 212 | 228 |

Compute-bound ~200-op damage chain (64 calcs per battle). Two runs are shown; there is about ±30% run-to-run noise.

| storage → compute | ns/calc |
|---|---:|
| int8 → int32 | 4.1 / 2.7 |
| int16 → int32 | 3.9 / 5.1 |
| int32 → int32 | 4.2 / 4.5 |
| int8 → native int8 | 5.9 / 5.8 |
| int16 → native int16 | 5.3 / 5.5 |
| float32 → int32 | 5.8 / 5.7 |

- **Small storage dtypes are pure wins for bandwidth-bound work** (copies, selects, the scan carry): the cost is proportional to bytes.
- **For arithmetic, upcast to int32.** Native int8/int16 arithmetic is no faster on CPU and silently overflows PS 4096-based math (`x * 6144` needs 32 bits).
- **float32 is not an option for exact PS math.** Integers are exact only to 2^24, and `modify(x, m)` intermediates reach about 3e7.
- On GPU, 8/16-bit values are widened to 32-bit registers anyway, so the same rule applies (§8).

### 3E. Volatile flags (per mon, B × 12 mons)

Operations timed: test a batched-index flag, set a static flag, clear a group.

| representation | bytes/mon | ns/mon |
|---|---:|---:|
| `bool[64]` | 64 | 41.4 |
| `int8[64]` | 64 | 35.4 |
| **`uint32[2]` bitfield** | **8** | **2.1** |

- **Use a bitfield of 2 × uint32 words.** Avoid `uint64`: it needs `jax_enable_x64`, which changes default dtypes everywhere.
- Use static bit constants (`VOL_PROTECT = (0, 5)` for word and bit). Testing a static bit is one AND plus a compare.
- Keep the small *counters* (`vol_ctr[16]: int8`) as a separate array.

---

## 4. Random numbers  (`b4_rng.py`)

Setup: per battle-turn, 8 actions × 3 targets × 6 events = **144 uniforms**, under `vmap` over per-battle typed keys, B = 16,384. The no-RNG baseline is 1.4 ns/battle.

| impl | method | compile s | ns per battle-turn | ns/draw | temp MB |
|---|---|---:|---:|---:|---:|
| threefry2x32 (default) | `fold_in` chain ×144 | 3.01 | 3,069 | 21.3 | 45 |
| threefry2x32 | `split(k_turn, 144)` | 0.08 | 2,545 | 17.7 | 90 |
| threefry2x32 | **bulk `uniform(fold_in(key, turn), (8,3,6))`** | 0.50 | 1,774 | 12.3 | 36 |
| philox4x32 | fold_in ×144 | **> 150 (timeout)** | – | – | – |
| philox4x32 | split 144 / bulk | 0.09 / 0.13 | 282 / 207 | 1.96 / 1.44 | 18 / 0.1 |
| philox2x32 | split 144 / bulk | 0.07 / 0.06 | 225 / **104** | 1.57 / **0.72** | 18 / 0.06 |
| rbg | fold_in / split / bulk | 2.46 / 0.18 / 0.12 | 6,500 / 2,266 / 480 | 45 / 15.7 / 3.3 | – |
| unsafe_rbg | fold_in / split / bulk | 0.89 / 0.14 / 0.12 | 728 / 10,973 / 552 | 5.1 / 76 / 3.8 | – |
| **custom counter hash** (2× murmur3 `fmix32` of seed ⊕ (turn, a, t, e)) | – | 0.04 | **30** | **0.21** | 0 |

Correctness checks (in `b4_rng.py`):
- **`vmap(uniform)(keys) == [uniform(k) for k in keys]`:** True for threefry and philox, **False for rbg and unsafe_rbg.**
  - The JAX docs say that under `vmap` "the entire batch of output random numbers is generated from only the first key in the input key batch."
  - So rbg/unsafe_rbg with per-battle keys **breaks per-battle determinism and replay.** Don't use them for the engine.
- **A `fold_in` draw is reproducible standalone,** i.e. independent of every other draw in the turn. That is the property the scope doc's forced-outcome and replay design needs.

Findings:
- **On CPU, threefry is the single most expensive thing in a turn.** 144 draws (1.8–3.1 µs) cost about as much as the entire toy battle step (2–2.5 µs).
- **A bulk draw per turn is 1.7× cheaper than per-event `fold_in` chains,** and compiles in 0.5 s instead of 3 s.
- **Per-event fold_in chains are dangerous for compile time:** philox times out, and threefry takes 3 s for this alone.

Recommendations:
1. **Per battle, keep `seed: uint32[2]` in the state, or pass the key into `step` (pgx v2 style).** Each turn, draw **one bulk tensor** `u = uniform(fold_in(key, turn), (QUEUE, 3, N_EVENTS))` with fixed event slots: ACC, CRIT, ROLL, SECONDARY, TIEBREAK, MULTIHIT, …. Index it as `u[action_slot, target, EVENT]`.
   - The address of each draw is static, so draws don't depend on masked branches, and replay is deterministic.
   - The forced-outcome override for tests is `u = where(force_mask, forced_u, u)`.
2. **On CPU (tests, data generation), set `jax_default_prng_impl='philox2x32'`.**
   - It is a one-line change that measured 1.8–2.0× faster on the whole rollout (§5).
   - It is vmap-exact over keys (JAX docs table; checked here for philox4x32).
   - Use it **only with bulk draws or `split`**: philox `fold_in` chains did not compile within 150 s.
   - Note that it changes the random stream, so pin the impl per experiment for replay.
3. **If profiling on GPU still shows RNG as hot, swap the generator** behind the same `(seed, turn, slot, target, event) → u32` interface:
   - a 2-round integer hash (0.21 ns/draw here, roughly 60× cheaper than threefry); or
   - `philox2x32` bulk.
   
   Validate the hash's statistics once with a quick PractRand/BigCrush-style check, or a χ² on damage-roll histograms. Game RNG does not need cryptographic quality.
4. **Leave `jax_threefry_partitionable` at its default (True since JAX 0.5.0).** It matters only when sharding across devices.

---

## 5. Whole-rollout compile and throughput  (`b5_rollout.py`, `b5_driver.py`)

**Toy battle step** (`toy_battle.py`). It is not a correct engine, but it has the shape of one.
- **State:** NamedTuple with 22 leaves, per-mon `[2,6]` int16/int8 arrays plus `[2,6,7]` boosts, `[2,6,16]` vol counters, a uint32 vol bitfield, side and field scalars, and a key. That is 755 B/battle against the ≈ 2 KB target.
- **Queue:** 6 entries; each iteration picks the next actor by dynamic-speed argmax (Trick Room, Tailwind and paralysis aware).
- **Per action:**
  - 3-target axis (vectorised);
  - accuracy / crit / roll draws;
  - ~200-op PS-order integer damage calc;
  - spread modifier;
  - scatter HP update;
  - 64-way batched secondary-effect switch;
  - PP decrement.
- **End of turn:** 4 × 16-way item-residual switches, weather chip, counters, faint replacement, terminal check.

**Rollout:** `jit(lax.scan(body, T=128))`. Each step runs:
- `observe` ×2 players (212 floats each);
- legal masks;
- masked random-categorical "policy";
- `step`;
- auto-reset via `tree_where(done, fresh, state)`, where `fresh` is a gather from a 4,096-entry bank of pre-built initial states.

The state is donated (`donate_argnums=(0,)`). Episodes average about 27 turns.

| config | B | compile s (+ trace) | env-steps/s | µs per step call | XLA temp MB | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1,024 | 1.2 (+0.5) | 263k | 3,887 | 2 | 450 |
| baseline | 8,192 | 1.2 (+0.5) | 369k | 22,183 | 12 | 482 |
| baseline | 32,768 | 1.2 (+0.5) | 432k | 75,843 | 50 | 538 |
| baseline | 65,536 | 1.3 (+0.5) | 436k | 150,000 | 100 | 616 |
| **batch sharded over 8 virtual CPU devices** | 32,768 | 1.3 | **1.27 M** | – | 6/device | 581 |
| **… 10 devices (= performance cores)** | 32,000 | 1.4 | **1.38 M** | – | 5/device | 588 |
| … 14 devices | 32,760 | 1.5 | 1.26 M | – | 4/device | 618 |
| reset = full `init()` every step (gymnax style) | 8,192 | 1.3 | 263k (−29%) | | 13 | 495 |
| action queue Python-unrolled | 8,192 | 3.0 (+1.1) | 374k (±0) | | 14 | 859 |
| one-hot instead of gather/scatter | 8,192 | 1.3 | 326k (−12%) | | 12 | 475 |
| targets unrolled instead of vmapped | 8,192 | 1.4 | 395k (+7%) | | 12 | 499 |
| store obs (bf16) + actions + rewards for the whole rollout | 8,192 | 1.3 | 349k (−6%) | | 26 (+862 MB output) | 2,209 |
| **+4 extra hook points × 64 handlers per action** | 8,192 | 4.6 (+1.1) | 278k | | 12 | 1,009 |
| … same with `optimization_barrier` per hook | 8,192 | **1.8** | **326k** | | 12 | 628 |
| **+8 extra hook points × 64 handlers per action** (512 handlers) | 8,192 | **28.7** (+1.7) | **136k** | | 12 | **2,932** |
| … same with barrier | 8,192 | **2.5** | **285k** | | 12 | 783 |
| +16 hook points × 64 handlers, barrier (1,024 handlers) | 8,192 | 4.4 (+3.0) | 229k | | 12 | 1,099 |

Findings:
- **Compile time is small for a realistic-sized step,** about 1.2 s plus 0.5 s trace.
  - It grows **linearly** with handlers when hooks are fenced: +0.15 s per 64-handler hook point.
  - It grows **superlinearly** without fences: 28.7 s, a 21 MB optimised-HLO text, and 2.9 GB of compiler RSS at 8 × 64 handlers.
  - This is §1B's finding reproduced inside the full rollout. The barrier also makes the rollout **2.1× faster** at runtime.
- **Throughput saturates at B ≈ 32k on one XLA:CPU device,** at about 0.43 M env-steps/s.
  - Sharding the batch across `--xla_force_host_platform_device_count=10` virtual devices (one per performance core; `NamedSharding(mesh, P('b'))` on the state; no code change) gives **3.2× more: 1.38 M env-steps/s**.
  - XLA:CPU's intra-op threading leaves most cores idle on this kind of many-small-ops program.
  - **For CPU-side work (tests, PS differential runs, data generation), shard over cores.**
- **Memory is a non-issue.** XLA temp is about 1.5 KB per battle (100 MB at B = 64k) and the donated state about 0.75 KB per battle.
  - Storing per-step observations dominates: 212 bf16 × 2 players × T × B = 862 MB at B = 8k, T = 128.
  - On GPU, size T × B for the obs buffer (and use bf16 or uint8 features), not for the engine.
- **The bank-gather auto-reset is 29% cheaper than calling `init` every step** (gymnax style). Even the cheap reset is paid every step under vmap, so keep reset trivial.

**Where the time goes on CPU:**
- Changing only `JAX_DEFAULT_PRNG_IMPL` from `threefry2x32` to `philox2x32` raises rollout throughput from 278–308k to **562–572k env-steps/s (1.8–2.0×)**. These are paired runs at B = 8,192 (`results/b5_prng.log`); philox4x32 gave 461k.
- So **about half of the CPU rollout is threefry.** That covers about 130 draws per step, counting the step's per-action draws, policy sampling and reset.
- The rest is gathers/scatters with batched indices into the per-mon arrays, the switches, and loop overhead.
- On GPU the RNG share should shrink a lot (§8).
- Note: these late paired runs shared the machine with an unrelated single-threaded process (about 1 of 14 cores busy), which explains why the threefry baseline is about 20% below the 369k measured earlier. Compare within a row group, not across groups.

---

## 6. Testing and debugging ergonomics  (`b6_ergonomics.py`)

### Measurements

**`jax.disable_jit()`**
- `lax.cond` and `lax.switch` with concrete predicates run **only the taken branch** (instrumented branch counters: `cond_true=1, cond_false=0`, `switch=[0,0,1,0]`), and `fori_loop` runs as a Python loop (8 body calls). You can therefore put Python `print`/`breakpoint()` inside handlers and step through one battle.
- Cost: the toy step runs eagerly in **318 ms** vs **78 µs** jitted at B = 1, about 4,000× slower. Fine for stepping one battle, useless for test suites.
- `vmap` still batches under `disable_jit`, so step an **unbatched** state.

**Eager vs jit:** `step` under `disable_jit` equals the jitted `step` **bit-for-bit** on every leaf. Integer math has no float re-association issues.

**`checkify`** (B = 8,192, scan of 16 steps; no checks = 2,058 ns/step, 0.80 s compile):

| checks | placement | compile s | ns/step | overhead |
|---|---|---:|---:|---:|
| `user_checks` (hp ≥ 0, boosts in [-6, 6]) | `checkify(scan(vmap(step)))` | 0.85 | 2,088 | +1% |
| `user_checks` | `scan(vmap(checkify(step)))` | 0.80 | 2,048 | ≈ 0 |
| `user_checks \| index_checks` | outside vmap | – | – | **crashes**: `IndexError` in JAX's checkify rule for batched scatter (jax 0.12.0.1) |
| `user_checks \| index_checks` | inside vmap | 2.40 | 2,743 | +33%, 3× compile |
| `all_checks` | inside vmap | 2.76 | 2,762 | +34% |

- A failing check surfaces as `negative hp -5 (check failed)`.
- Put the battle index in the message yourself: `checkify.check(ok.all(), "hp<0 in battle {i}", i=jnp.argmin(ok))`. With the vmap-inside placement the error object is batched per battle.

**`jax.debug.print`:** one host callback per scan step (an aggregate, not per battle) had **no measurable overhead** at B = 8,192 (2,068 vs 2,058 ns/step). Per-element prints under `vmap` print B lines, so print aggregates or a slice.

**Compile-time levers on the toy rollout** (B = 8,192; baseline 1.26 s compile + 0.53 s trace, 369k steps/s):

| lever | compile s | runtime effect |
|---|---:|---|
| **persistent cache, warm** (`jax_compilation_cache_dir`, min_compile_time_secs = 0) | **0.10** (trace still 0.53) | none |
| `XLA_FLAGS=--xla_backend_optimization_level=0` | 0.65 | **3.6× slower** (102k steps/s) |
| `JAX_DISABLE_MOST_OPTIMIZATIONS=1` | 0.66 | 3.6× slower |
| `--xla_llvm_disable_expensive_passes=true` | 1.26 | none (no gain) |

The persistent cache **works on XLA:CPU** in jax 0.12. Its key includes the un-optimised HLO, the jaxlib version, XLA flags and the device topology (per the JAX docs), so any edit to a traced function means a recompile. That is fine for re-running tests and sweeps, not for an edit-compile loop.

### Recommended dev workflow

1. **Static `Config` controls what gets compiled.**
   - `Config(dex=frozenset(effect ids in the test teams), hooks=..., checks=False, barrier=True)` is hashable and passed as `static_argnums`, or closed over.
   - Handler tables and switch branch lists are built from `config.dex`, so a unit test touching 6 effects compiles 6 branches, not 300.
   - This is the biggest lever. Compile time is linear in branches with barriers and rolled loops (§1), and superlinear without them.
2. **Three build profiles:**
   - *unit* (one mechanic, tiny dex, `disable_jit` available);
   - *conformance* (full v1 dex, `checkify(user_checks)` inside vmap, B ≈ 1–4k scenarios, persistent cache);
   - *train* (full dex, no checks, `donate_argnums=(0,)` on the state, one `jit(scan(...))` per rollout).
3. **Write the engine as pure functions of an un-batched state; `vmap` only at the API boundary** (pgx/minimon style). The same `step` then runs in all three modes: unbatched with `disable_jit` for stepping, `jit` for B = 1 replays, and `jit(vmap)` for batches.
4. **Differential tests vs Showdown:**
   - Run the jitted engine over a batch of scenarios, with forced RNG outcomes via the per-turn uniform tensor (§4).
   - Pull results to NumPy with `np.asarray` / `jax.device_get` and diff in Python.
   - Don't try to run the engine *on NumPy*. `lax.*` control flow, `.at[]` updates and `vmap` have no NumPy equivalent, and `disable_jit` already gives NumPy-like eager semantics on concrete arrays. Its results are bit-identical to jit (measured above), so it is the "reference interpreter" when debugging.
   - Keep a `numpy_ref.py` only for a few formula oracles (damage calc, stat calc), cross-checked against `@smogon/calc`.
5. **Keep compiles under about 1 minute as handlers grow:**
   - **(a)** tables first;
   - **(b)** per-hook switches;
   - **(c)** an `optimization_barrier` after each hook;
   - **(d)** rolled `fori` for the queue, vectorised targets, no Python-unrolled loops around switches;
   - **(e)** persistent cache on everywhere (`JAX_COMPILATION_CACHE_DIR`, `jax_persistent_cache_min_compile_time_secs=0`);
   - **(f)** split jits in development (`init`, `step`, `observe` as separate jitted functions; a hook-level jit to test one handler family). Note that splitting doesn't help the final training jit, because everything inlines;
   - **(g)** measure: log `lower()` and `compile()` time per build in CI and fail when compile time regresses by more than 30%;
   - **(h)** don't use `xla_backend_optimization_level=0` to save compile time. It halves compile and makes runtime 3.6× slower.
6. **Debugging NaN/garbage:**
   - `jax.debug.print` aggregates inside the scan;
   - `jax.debug.callback` to dump the state of the first failing battle (by index);
   - `checkify(user_checks)` invariants (hp ∈ [0, maxhp], boosts ∈ [-6, 6], at most one Protect per turn, …) in the conformance build;
   - `JAX_TRACEBACK_FILTERING=off` when an error points into JAX internals.

---

## 7. Prior art: pgx, minimon, jumanji, gymnax

Shallow clones (read, not executed) are in `research/jax-bench/ext/`:
- pgx `87278d2` (v2.6.0);
- minimon `d7955e7` (v0.1.0);
- jumanji `be60de6` (v1.1.2);
- gymnax `main` @ `4292f01` (v1.0.1; the default branch `env-dev` is stale).

**All four keep one un-batched state per env and `vmap` from outside.** None batches inside the env.

| | pgx | minimon (JAX Pokémon analogue, 1v1) | jumanji | gymnax |
|---|---|---|---|---|
| **State container** | vendored `flax.struct.dataclass` (`pgx/_src/struct.py`); explicit dtypes in defaults; chess nests a `NamedTuple` | own frozen dataclass on `jax.tree_util.register_dataclass`; player axis first (`hp f32[2,N]`, `stats [2,N,6]`) | `chex.dataclass`, must contain `key` | `flax.struct.dataclass` `EnvState` + separate `EnvParams` |
| **Static vs traced config** | n/a | **3 layers:** hashable `EnvConfig` (Python `if` at trace time) → traced `EnvParams` tables (vmappable) → `State` | n/a | `EnvParams` pytree (vmappable) |
| **step signature** | `step(state, action, key)` (v2 moved the key *out* of the state so planners can't peek at future randomness) | `step(key, state, actions i32[2], params) → StepOutput(state, obs f32[2,D], reward f32[2], done, info)` | `step(state, action) → (state, TimeStep)`; key inside state | `step(key, state, action, params)` |
| **Terminal handling** | `lax.cond(terminated \| truncated, no-op with zero rewards, _step)`; at terminal `legal_action_mask` = all-ones (avoids softmax NaN) | `select_tree(state.done, state, advanced)` | none in env; done in wrapper | n/a (auto-reset always) |
| **Auto-reset** | `pgx/experimental/wrappers.py:auto_reset`: `lax.cond(done, init_fn(key2).replace(...), state)`; drops the terminal obs (the docstring admits truncation is unsupported) | in `rollout.py` `lax.scan`: `env.reset(k)` **every step**, then `select_tree(done, fresh, next)` | `AutoResetWrapper`: `lax.cond(timestep.last(), _auto_reset, …)`; terminal obs kept in `extras["next_obs"]` | `step` = `step_env` + **always** `reset_env`, then `tree.map(lax.select(done, reset, step))`; `info["final_observation"]` |
| **Observation** | `observe(state)` from the current player's view; recomputed at the end of every step | `observe` vmapped over player → `f32[2,D]` = [global \| own \| opp]; hidden info via "revealed" masks; entities encoded by properties, not one-hot ids | `observe(state)`; specs as cached properties | `get_obs(state, params)` |
| **Multi-agent** | turn-based only; `current_player`, randomised seat order (`_player_order`) | **simultaneous**: joint `actions i32[2]`, `reward f32[2]`, no-op action legal only when the player has nothing to decide | `num_agents`-shaped TimeStep | single-agent |
| **Content / effects** | NumPy lookup tables precomputed at import | moves as columns (power/acc/priority/effect/effect_arg/effect_mag/chance); 8-kind effect enum, **every kind computed then masked**; items/abilities as `ModifierTable` rows (`stat_mult, damage_mult, turn_heal, entry_foe_boost, …`); **no `lax.switch` / `fori_loop` / `while_loop` in step**; float damage (not PS-exact) | — | — |
| **Control flow** | `lax.switch` in a few games, `fori_loop` common, `while_loop` in Go flood-fill and in mahjong init rejection sampling; fixed-size compaction via `jnp.nonzero(..., size=K)` | one `lax.cond` on phase (TURN vs FORCED_SWITCH) | `lax.switch` for TimeStep type | one `while_loop` (asterix) |

**Patterns to copy:**
1. **Three layers:**
   - static hashable `Config` (format, rules, dex subset, *which handlers to compile*), branched with Python `if` at trace time;
   - traced `Dex`/`Params` tables (moves, species, items, abilities, type chart as dense int arrays);
   - per-battle `State`.
   
   This is minimon's design, and it is also our compile-time lever (§6).
2. **Pass the key into `step`; don't store it in the state** (pgx v2, gymnax, minimon). Derive per-event randomness from (key, turn, actor, event) at fixed positions, so draws don't depend on which branches were masked.
3. **Player axis first:**
   - `[2, 6]` per mon, `[2, 2]` active slots;
   - joint action `i32[2, 2]`, `reward f32[2]`;
   - `obs [2, D]` via `vmap(observe, in_axes=(None, 0))(state, jnp.arange(2))`;
   - `legal bool[2, 2, A]`.
4. **Phase field plus a no-op action** that is legal only when the player has nothing to decide (minimon). This covers forced switches, the opponent-only replacement in EOT_REPLACE, and done states.
5. **Freeze terminal states with `where` and zero their rewards.** Use pgx's convention of an all-True legal mask at terminal so softmax never sees an all-masked row. Map illegal actions to a legal default rather than terminating (minimon), and count the replacements in `info`.
6. **Auto-reset costs one reset per step under vmap,** because `lax.cond` with a batched predicate becomes `select`. pgx's cond and jumanji's cond save nothing at B = 64k.
   - So make reset cheap: gather a pre-generated initial state from a bank by index (measured in §5).
   - Keep the terminal observation (`final_observation`) for truncation bootstrapping.
7. **Data-driven effects with "compute every kind then mask"** (minimon) is the table approach of §1. It is the right default; bespoke handlers are the exception.
8. **Fixed-size bounded work:** `jnp.nonzero(..., size=K, fill_value=...)`, `lax.top_k` / Gumbel tricks for sampling without replacement, static-bound loops.
9. **Shapes via `jax.eval_shape`** (minimon `obs_size`) rather than running `init`.

**Pitfalls seen:**
- **gymnax runs a full `reset_env` every step;** at our state size, use a bank gather.
- **minimon runs about 6 `select_tree`s over the whole state per turn.** That is fine inside one jit, because untouched leaves fold away (§3B), but expensive if the trees are actually different.
- **pgx's `auto_reset` drops the terminal observation.**
- **jumanji puts the key in the state.**
- **minimon uses float damage;** we need int32 PS-exact math.
- **`while_loop` in reset (pgx Sparrow Mahjong rejection sampling) runs to the slowest lane under vmap.**

---

## 8. Transfer to an H100

There is no GPU in this study, so this section is reasoning. Sources are cited inline; treat the estimates as ±2–3× until profiled on the real box.

### 8.1 Hardware ratios

| | this M4 Max under XLA:CPU (measured) | H100 SXM (spec) | ratio |
|---|---|---|---:|
| simple int32 ALU throughput, fused elementwise | ≈ 70 G ops/s (14 threads; §0) | 132 SMs × 64 int32 ops/clk × ~1.8 GHz ≈ 15 T ops/s; int32 add/mul/compare issue at about half the fp32 FMA rate on sm_80/sm_90 (CUDA C++ Programming Guide "Arithmetic Instructions" throughput table: 64 vs 128 results/clk/SM as I recall it; I could not re-fetch the table to verify) | ~200× |
| streaming bandwidth | ≈ 210 GB/s (§3D) | 3.35 TB/s HBM3, 50 MB L2 | ~16× |
| fixed cost per op | ≈ 3 µs per thunk or loop iteration; 6 µs + 1 µs/leaf per jit call | one kernel per fusion ("A fusion is always compiled to exactly one GPU kernel", openxla.org/xla/gpu_architecture), at ≈ 2–5 µs launch + tail each from the host stream; much less inside CUDA graphs (XLA "command buffers", `--xla_gpu_enable_command_buffer`) | ~1× |

**Consequence:** the GPU wins on per-battle work by 15–200×, but **not on the number of kernels per step.** The engine on GPU will be **launch/latency-bound unless B is large.** The scope doc's B = 16–64k is the right regime; 128–256k is affordable on memory (§5) if the learner can consume it.

### 8.2 Which findings transfer

| finding | transfers? | why / what changes on GPU |
|---|---|---|
| batched `switch` = all branches + select; cost ∝ total branch work (§1A) | **yes** | It's an HLO-level rewrite (JAX `cond`/`switch` docstring). XLA:GPU fuses the branches into one kernel, so there is no divergence (it's predicated), but every lane does all the work. The per-branch cost drops by about 200× with the ALU ratio, so **N ≈ 100–300 handlers is affordable on H100** as long as compile time is controlled. |
| one-hot / take materialise [N, B] (§1A) | **yes, worse** | HBM traffic instead of L2/registers; avoid. |
| table interpreter ≈ flat in N (§1A) | **yes** | Same instruction count; gathers from a small constant table hit L1/L2. |
| chained switches blow up without `optimization_barrier` (§1B, §5) | **expected yes; re-measure** | XLA:GPU's producer-consumer fusion also duplicates cheap elementwise producers into consumers. Giant select-tree kernels mean register spills and long LLVM/ptxas times. On GPU the barrier costs one extra kernel launch plus a few bytes per battle, which is cheap. |
| loops: rolled `fori`/`scan` ≈ unrolled at runtime, unrolling costs compile (§2) | **partly** | On GPU each loop iteration re-launches the body's kernels. With a **static trip count, XLA's `WhileThunk` skips the condition and never copies the predicate to the host.** With an unknown trip count it runs the condition, does a `Memcpy` of the predicate and `BlockHostUntilDone()` **every iteration** (xla `backends/gpu/runtime/while_thunk.cc`). So `lax.while_loop` with a data-dependent predicate is **strictly worse on GPU** than a masked `fori_loop`. Unrolling the 6–8 queue iterations doesn't reduce kernel count much, but it removes the loop machinery and lets XLA fuse across iterations, so try it in a release build. Command buffers can also capture whole loops. |
| vectorise the 3-target axis (§2) | **yes, more so** | Wider kernels and fewer launches. |
| small storage dtypes, int32 compute (§3D) | **yes** | GPU registers are 32-bit and 8/16-bit values are widened on load, so storage dtype only changes bytes moved. That matters for the `scan` carry, the masked merges and the obs buffer. Avoid `uint64`: 64-bit integer ops are multi-instruction on NVIDIA. |
| ≤ 100 leaves; per-leaf costs (§3A/C) | **yes** | Each separately-written leaf can be its own kernel unless XLA's horizontal/multi-output fusion merges them; fewer, larger arrays mean fewer launches. Python dispatch overhead is host-side and identical. |
| threefry is 40% of a CPU step (§4) | **no** | On CPU, JAX lowers threefry with **rolled loops** (`_threefry2x32_cpu_lowering_rule`, `use_rolled_loops=True`). That puts a 5-trip `while` with tiny thunks in every PRNG call; it accounts for 11 of the 13 while loops in the rollout HLO. On CUDA the default lowering is unrolled straight-line integer code that fuses (`jax_threefry_gpu_kernel_lowering=True` is an alternative FFI kernel). Expect RNG to be a small share of the step on H100. The bulk-per-turn pattern is still the right *structure*: fewer ops and kernels, and it supports forced outcomes. |
| sharding over cores gives 3.2× (§5) | n/a | One GPU = one device. Multi-GPU uses the same `NamedSharding(P('b'))` code. |
| compile times | **yes, larger** | XLA:GPU runs more passes plus LLVM→PTX→ptxas per kernel. Expect GPU compile times **≥ the CPU numbers here, roughly 1.5–3× in my experience** (unmeasured). The superlinear no-barrier cases would be minutes. The persistent cache supports GPU, including the per-fusion autotune cache (`jax_persistent_cache_enable_xla_caches`). |

### 8.3 Throughput estimate for H100

**Kernel count:** `hlo_kernels.py` on the optimised toy-rollout HLO (B = 8,192) counts **≈ 980 kernel-like ops per env step** (≈ 850 fusions) once the CPU-only rolled threefry loops are treated as single kernels; it is 2,238 as compiled for CPU. XLA:GPU fuses differently, so read this as ±2×.

**Launch floor:** 1,000 kernels × 2–5 µs is **≈ 2–5 ms per step** without command buffers, and ≈ 0.5–1.5 ms with them.

**Compute:**
- The CPU spends about 2.3 µs per battle-step (one device, B = 32k), of which about 1.4 µs is non-RNG.
- At a ~100–200× ALU ratio that is ≈ 7–15 ns per battle-step on H100, i.e. **≈ 0.5–1 ms per step at B = 64k**.
- Gathers/scatters into `[B, 2, 6]` arrays with per-battle indices are uncoalesced and are the likely dominant memory cost. They are still small: ≈ 64k × 32 B sectors per gather is about 2 MB, ≈ 0.6 µs.

| | estimate |
|---|---|
| **toy engine, B = 64k** | ≈ 1.5–6 ms/step → **≈ 10–40 M env-steps/s** (launch-bound) |
| **real v1 engine** (assume 3–10× more kernels per step and 10–30× more per-battle work) | ≈ 1–10 M env-steps/s at B = 64k |
| scope-doc target | ≥ 10⁵ env-steps/s |

The real-engine estimate clears the target by 10–100×. The risks are compile time and kernel count, not FLOPs.

**What to do on the H100 first:**
1. Profile one rollout with `jax.profiler` (Perfetto/XProf). Count kernels per step and the fraction of time spent in gaps between kernels.
2. Sweep B over 16k / 64k / 256k.
3. Toggle `--xla_gpu_enable_command_buffer` (default set vs `''`) and an unrolled-queue build.
4. Verify the barrier finding on the H100.

**Kernel-count hygiene matters more than op count on GPU:**
- merge per-field `.at[]` updates into one update of a packed array where natural;
- vectorise targets;
- avoid long chains of tiny scatters.

### 8.4 jax-metal (Apple GPU), for completeness

- jax-metal 0.1.1 installs only with jax 0.4.34 (separate `.venv-metal`) and runs simple programs.
- On §1A it was **slower than XLA:CPU**: switch N = 128 at 212 vs 24 ns/elem; table at 63 vs 8.5 ns/elem; even the context copy baseline was 10.5 vs 0.9 ns/elem.
- The toy rollout (B = 1,024, T = 8) lowered ("Donation is not implemented for METAL") but did not finish compile+run within 5 min, so it was abandoned.
- Not useful as a GPU proxy. Don't use it for this project.

---

## Appendix: reproduce

All scripts are in `research/jax-bench/`. Run them one at a time, because they share all cores.

```sh
uv venv .venv && uv pip install --python .venv/bin/python jax numpy chex flax
.venv/bin/python b1_dispatch.py --part AB --B 65536   # 1A/1B dispatch (+ grouped+barrier)
.venv/bin/python b1_dispatch.py --part C --timeout 90 # 1C loop-nest compile (subprocess per config)
.venv/bin/python b1_barrier_chain.py                  # chained switches +/- optimization_barrier
.venv/bin/python b2_control.py                        # 2A/2B control flow
.venv/bin/python b3_state.py                          # 3A-3F state representation
.venv/bin/python b4_rng.py                            # 4 RNG
.venv/bin/python b5_driver.py && .venv/bin/python b5_driver.py extras   # 5 rollout grid, sharding, hooks, HLO dump
.venv/bin/python hlo_kernels.py results/rollout_B8192.hlo.txt 128       # kernels per env step
.venv/bin/python b6_ergonomics.py                     # 6 disable_jit / checkify / debug.print / eager==jit
.venv/bin/python summarize.py                         # markdown tables from results/*.json
./metal_try.sh                                        # jax-metal attempt (.venv-metal, jax 0.4.34)
```

Notes:
- `toy_battle.py` is the shared toy engine; `handlers.py` holds the synthetic PS-like handlers.
- `ext/` holds read-only shallow clones of pgx, minimon, jumanji and gymnax. They were read, never executed.
