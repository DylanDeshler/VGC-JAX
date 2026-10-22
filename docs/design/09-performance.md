# 09 — Performance: targets, JAX rules, build profiles, benchmarks

Evidence: `research/jax-patterns.md` (**JP**). It was measured on an Apple M4 Max under XLA:CPU, with jax 0.12.0.1. The H100 transfer analysis is in JP §8.

## 1. Targets (from `research/rl-training-design.md` §8; ADR / 08 §11)

| Metric | Minimum | Target | Where measured |
|---|---|---|---|
| Engine env-steps/s, 1× H100, B = 16k–64k (standalone rollout with random legal policy) | 3×10⁴ | 10⁵ | `bench/throughput.py --device gpu` at G-v1, M9 |
| Env-steps/s at B = 1k–4k (search) | 2×10⁴ | — | same |
| Compile time, full stage build (train profile) | — | < 120 s | CI T1 (CPU) and GPU |
| Compile time, dev build (handler subset) | — | < 60 s | CI T1 |
| Memory at B = 64k (state + temps, excluding rollout storage) | — | < 15 GB | GPU |
| CPU harness replay throughput | ≥ oracle generation rate (≈ 450 battles/s) | — | `bench/replay.py` |

### 1.1 CPU proxy budgets
These are enforced at G-v0 and G-v1 whenever no GPU measurement exists. Calibrate them once a GPU is available.
- **(a) Kernel count.** Kernel-like ops per env step in the optimised train-profile HLO ≤ 10,000, measured by `bench/hlo_kernels.py` (ported from `research/jax-bench/hlo_kernels.py`). The toy is about 980. At the 2–5 µs launch floor (JP §8.3) that bounds a step at ≤ 50 ms, i.e. ≥ 1.3 M env-steps/s at B = 64k.
- **(b) CPU throughput.** Train profile, sharded over 10 P-core virtual devices, B = 32k: ≥ 5×10⁴ env-steps/s. The toy reached 1.38 M.
- **(c) Compile.** ≤ 120 s, cold cache.

**GPU escalation.**
- Request GPU access at **M4.3**, not M9. A one-day cloud H100 rental is enough for 09 §5 steps 1–3.
- If no GPU exists by G-v1, M9 becomes "CPU-only hardening" against (a)–(c), and every gate report states "GPU gate pending".
- Do not use jax-metal: it needs jax 0.4.34, was slower than CPU and didn't finish the toy rollout (JP §0, §8).

**Context.** The JP toy rollout ran at 0.43 M env-steps/s on one CPU device and 1.38 M sharded over 10 virtual CPU devices. The H100 estimate for a real engine is 1–10 M env-steps/s at B = 64k (JP §8.3), launch-bound. The risks are **compile time and kernel count, not FLOPs**.

## 2. Mandatory JAX rules (with measured justification)

| # | Rule | Evidence |
|---|---|---|
| J1 | Effects: table rows first, then per-hook `lax.switch` over only that hook's handlers | switch cost is linear in total branches; a table at N=512 is 29 vs 105 ns, with 5× faster compile (JP §1A) |
| J2 | `lax.optimization_barrier` after every hook dispatch | chained switches: compile 24.5 s → 1.7 s, runtime 795 → 93 ns (JP §1B). In the rollout: 28.7 s → 2.5 s compile, 2.1× faster runtime (JP §5) |
| J3 | Never one-hot or `take` over stacked branch outputs | materialises [N, B]: 1.2–1.5 GB at N=512, 3–50× slower (JP §1A) |
| J4 | Rolled `fori_loop`/`scan` for the action blocks; never Python-unroll loops containing switches | fori8 × unroll3 with a 128-way switch compiles in 37.8 s; fully unrolled > 90 s (JP §1C) |
| J5 | Vectorise the 3-target axis with `vmap` inside the move body | 1.6× faster than an inner fori, and one HLO copy of switches (JP §2A) |
| J6 | No `while_loop` with a data-dependent trip count | runs at max trip count under vmap; on GPU it adds a host sync per iteration (JP §2A, §8.2) |
| J7 | int8/int16 storage, int32 compute; no uint64 (needs x64); no float math in mechanics | native small-int arithmetic is 30–45% slower and overflows (JP §3D) |
| J8 | Bitfields (2+ × uint32) for presence flags | 20× cheaper than `bool[64]` (JP §3E) |
| J9 | Handlers touch only the fields they change; use `.replace` | XLA folds `select(p, x, x)`, so untouched leaves are free (JP §3B) |
| J10 | RNG: custom counter hash (`contract/drawkey.md` §2), never `rbg`/`unsafe_rbg` with per-battle keys | hash 0.21 ns/draw vs threefry 12–21 ns on CPU; rbg breaks per-battle determinism under vmap (JP §4) |
| J11 | Auto-reset = cheap `init` (bank gather + stat compute) | a full init every step costs 29% (JP §5) |
| J12 | Scatter/gather with batched indices (`.at[idx]`) rather than one-hot masks over small axes | one-hot was 4.8× slower in rolled loops (JP §2A) |

The T0 lint checks what it can mechanically:
- `jnp.where` on a Python `bool` from a tracer;
- `if` on traced values (best-effort AST check);
- missing `optimization_barrier` after a `run_hook` call;
- `while_loop` usage.

## 3. Build profiles (static `EngineConfig`; ADR-006)

| Profile | Handlers compiled | Checks | Modes | Use |
|---|---|---|---|---|
| `unit` | the entities in the test teams only (`cfg.handlers`) | `checkify(user_checks)` inside vmap | `forced`, `draw_log` on | scenario tests; `jax.disable_jit()` stepping of one unbatched battle |
| `conformance` | the full stage | `checkify(user_checks)` inside vmap (≈ +1%) | `forced`, `draw_log` on | harness replay. On CPU, shard the batch over `--xla_force_host_platform_device_count=<P-cores>` with `NamedSharding(P('b'))` (3.2× in JP §5) |
| `train` | the full stage | none | off | `jit(scan(...))` rollouts, `donate_argnums` on the state |

**Scenario tests don't compile per test.** The runner collects all scenarios in a session, builds **one** `conformance`-profile executable per stage, and replays them as a single vmapped batch, with scenarios padded to a common decision count. The `unit` profile is only for `jax.disable_jit()` debugging. This keeps T0/T1 inside budget as scenarios grow into the hundreds.

**Debugging facts** (JP §6):
- **`disable_jit`.** Under `jax.disable_jit()`, `lax.cond` and `lax.switch` take **only the taken branch** and `fori_loop` becomes a Python loop, so prints and breakpoints inside handlers work. It is about 4,000× slower and the results are bit-identical to jit. Step **unbatched** states.
- **`checkify`.** Put `user_checks` *inside* vmap. `index_checks` outside vmap crashes on batched scatter in jax 0.12.
- **`jax.debug.print`.** Use it on aggregates only; per-element printing under vmap prints B lines.
- **Persistent compilation cache.** Enable it everywhere (`JAX_COMPILATION_CACHE_DIR`, `jax_persistent_cache_min_compile_time_secs=0`). A warm cache brings compile from 1.26 s to 0.10 s on the toy rollout. Any edit to traced code invalidates the entry.
- **Never use `xla_backend_optimization_level=0` to save compile time.** It makes runtime 3.6× slower.

## 4. Benchmarks to maintain (`bench/`)

| Script | Measures | Run |
|---|---|---|
| `compile_time.py --profile {unit,conformance,train} --stage vN` | trace + lower + compile seconds; HLO size; kernel-like op count (`bench/hlo_kernels.py`, ported from `research/jax-bench/hlo_kernels.py`) | T1 (CPU), budgets enforced: fail on > 30% regression or over-budget |
| `throughput.py --B {1k,4k,16k,64k} --device {cpu,gpu}` | env-steps/s with a random legal policy inside `jit(scan)`, auto-reset on | T3 weekly; G-v0, G-v1, M9 |
| `replay.py` | harness replay battles/s (conformance profile, sharded CPU) | T3 |
| `memory.py` | `compiled.memory_analysis()` + peak RSS/HBM at B | T3 |
| `hook_cost.py` | per-hook dispatch cost vs handler count (regression guard for J1/J2) | when the hook framework changes |

Store results in `reports/perf/<date>.json` with a commit hash, and plot trends (optional).

## 5. GPU playbook (M9; JP §8)

1. **Profile.** `jax.profiler` trace of one rollout: kernels per step, time in inter-kernel gaps vs inside kernels.
2. **Sweep batch size.** B ∈ {16k, 64k, 256k}. Launch-bound programs scale with B until memory binds.
3. **Command buffers.** Toggle `--xla_gpu_enable_command_buffer` (CUDA graphs) on and off.
4. **Try release-build options:**
   - unroll the 3-target vmap into explicit lanes (JP §1C: 2.5× on CPU at a high compile cost);
   - unroll Block B's 6 iterations;
   - compact `vol_state`.
5. **Re-verify the barrier finding** on GPU (J2). It is expected to hold.
6. **Reduce kernel count:**
   - fuse per-mon scalar fields into dtype-grouped arrays if leaf count drives kernels (JP §3A, §8.2);
   - hoist hook candidate construction (04 §6.2) out of repeated calls where state hasn't changed.

## 6. If targets are missed (decision tree)

- **Compile > budget:**
  1. check barrier placement (J2);
  2. count branches per hook and move more handlers to rows (J1);
  3. check for accidental unrolling (J4);
  4. split the stage build only as a last resort, by gating rarely used handler families behind a config with a fallback error flag.
- **Throughput < minimum on GPU:**
  1. profile;
  2. if launch-bound, use larger B, command buffers and kernel-count reduction;
  3. if compute-bound, the cost of the all-branches hooks dominates, so move them to rows and shrink the largest switches (`onStart`, `onHit`, `onTryHit` are the biggest: 56/50/36 templates in v1, `research/dex-scope.md` §5);
  4. only then consider splitting the move pipeline into "common path" and "rare path" with per-battle predication.
- **Memory:** compact `vol_state`; store masks as `slot_mask` + `pair_ok`; reduce `PublicEvents` capacity.
