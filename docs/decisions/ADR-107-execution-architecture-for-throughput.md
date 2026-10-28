# ADR-107: execution architecture for throughput — flat stage pipeline, effects as data, training build

Status: **Accepted** (human, 2026-10-27) (performance; no change to the conformance contract,
engine semantics or PS fidelity). Supersedes the per-call-site tactics in ADR-106 as the main plan; ADR-106's
pilot change (static add_volatile dispatch) stays.

## 1. Why

### 1.1 Where we are (measured)
| | value | source |
|---|---|---|
| Complete battles/s, standalone, 1× L40S, random legal policy | **~650–820** (B=16k–64k: ~630–650; B=4k: ~740–820) | `bench/gpu_check.sh`, 2026-10-26 (episodes finished / wall time) |
| Decision-steps/s, same run | 9k–17k | same |
| Compile + first call | 10–15 min (R4 trigger: 120 s) | same |
| Kernel-like ops / env step (CPU proxy, B=256) | 255k after the ADR-106 pilot (was 300.6k) | `bench/hlo_kernels.py` |
| Scaling | time per step call grows linearly with batch from B=4k: the GPU is saturated by per-env work (not launch-bound) | ADR-106 |

### 1.2 What "fast enough" means (external reference points)
- **mikumiku37** (Reg M-C ladder #1; Smogon thread 3789199, read 2026-10-27): a **Rust engine on CPU**, "~7k
  random battles/s on a single core"; training on one RTX 5090: 8,192 battles in parallel, "collection runs at
  ~100k steps/s including network inference", PPO update ~70% of wall time, ~330M games in 48.3 h (≈1,900
  games/s averaged over the whole run, so ≈6k games/s during collection). No engine details or fidelity
  method published.
- **Jaxcalibur** (singles): ~165 games/s including learning on 1 H100 (docs/research/rl-training-design.md §1).
- **PokeJAX** (singles, LLM-translated, all-branches `lax.switch`): claims 5×10⁸ random-action SPS; fidelity
  checked only by win rate.

**Targets proposed here** (standalone, random legal policy, one GPU, B = 8k–16k):
- **T1 ≥ 3,000 battles/s** — enough to train at the mikumiku37 rate with the engine under ~half the wall time.
- **T2 ≥ 10,000 battles/s** — the engine stays under ~20% of wall time (08 §6 intent).
- **Compile ≤ 3 min** for the training build.
These replace the env-steps/s targets as the headline metric (env-steps depend on micro-step count;
battles/s does not). `bench/throughput.py` gains a battles/s line.

## 2. Diagnosis: why the engine is heavy (measured with `bench/hlo_attrib.py`)
Under `vmap`, every env executes every instruction of the step; work per step = the size of the compiled step.
Three structural properties make that size large; none of them is required by PS fidelity.

1. **Every micro-step contains every phase.** The micro-step body runs switch-in, move begin, one hit, move end,
   the runAction tail, end of turn and residual for every env, though each env needs one of them. A decision
   takes ~4–8 micro-steps.
2. **Heavy helpers are inlined at every call site.** Each Python call of a JAX function is a separate copy in
   the compiled step, and every copy executes. Measured (impl stage, 300.6k ops before the pilot):
   - fieldEvent('SwitchIn') ≈ 25% — `run_switch` exists ~7 times: once in `run_light`, and inside two forced-switch
     slots of each of three `post_action` copies;
   - boost machinery ≈ 13% (many call sites); `resolve_faints` ≈ 9% (many call sites);
   - `encore_start` 7.9% — one handler copied into every `add_volatile` dispatch (fixed by the ADR-106 pilot);
   - `move_begin` / `move_hit` ≈ 20% each.
3. **Per-mon access by dynamic index.** `mget` / `mset` / `mset_many` (gather / scatter on the 8-mon arrays by a
   traced ref) are ~29% of ops by themselves, called thousands of times per step.

The per-entity logic (abilities, items) is small: the type-boost items, resist berries and status-immunity
abilities that ADR-106 targeted cost ~0.1% (already table-driven).

## 3. Design

### 3.1 Principle
A small, fixed set of generic stages that every env runs **once per micro-step**, each consuming data (requests,
tables) rather than being called from many places. Variation lives in data; code size stops growing with call
sites or content.

### 3.2 Flat stage pipeline with an explicit continuation (the core change)
Today control flow is a nested call tree (runAction → useMove → hit → secondaries → boost …) that mirrors PS.
The proposal is the standard conversion of a recursive interpreter to a GPU-friendly one:

- Each env carries a small **continuation**: a phase id (`pc`) plus a few registers (current actor, move, target
  set, hit number, pending forced switch, …). ADR-104's resumable move execution (`st.exec`) is already this for
  the hit loop; it generalises to the rest.
- The micro-step body becomes a **fixed sequence of stages**, each compiled once:
  `select-action → switch-in → move-begin → hit → boost → status → damage-apply → faint → residual → end-turn`.
  A stage does real work only for envs whose `pc` (or pending-request flag) selects it, then sets the next `pc`.
- Where PS interleaves (e.g. a forced switch inside the runAction tail, before faints resolve), the stage that
  needs the interleaved work **stops and sets `pc` to resume after it**. The work then runs in the single shared
  stage, and the interrupted stage resumes where PS would. Order is PS's order by construction; only where in the
  compiled program the work sits changes.
- Each stage being single-instance removes the duplication in §2.2 by design, not by case-by-case tuning.

Expected effect (estimate, to be measured): switch-in ~25% → one copy (~4–5%); boost / faint / status sites
collapse to one copy each. Together a 2–3× smaller step. The cost: some decisions need more micro-steps.
`micro_steps` per call is tuned on battles/s, not on step count.

### 3.3 Requests instead of calls for the shared effect stages
Boost, status, volatile add and damage applications that today are called inline from many handlers become
**requests** written into small fixed-size buffers in the state (e.g. `pending_boost[k] = (target, deltas,
source, effect, order)`). The stage drains them in PS order (the order PS would have run the inline calls).
Rules for when a request must be drained before continuing (PS reads its result, e.g. `if (!this.boost(...))`)
are explicit: such sites end the stage and resume after the drain (§3.2). Buffer overflow raises a flag like
every other static bound.

### 3.4 Fixed-layout per-slot computation instead of dynamic indexing
Most work concerns at most 4 active mons. Instead of `mget(arr, ref)` gathers, stages compute over all 8 mons (or
the 4 active slots) with one-hot masks. On 8-element axes this is cheaper than gathers and scatters and fuses
into neighbouring ops. Mechanical, local change; applied first to the hottest accessors from the profiler.

### 3.5 Effects as data
Move, ability and item effects that share logic become table rows read by generic kernels. Already done:
damage modifiers (`mod_family`), on-hit secondaries, type-boost items, resist berries. Next families, from the
catalogue: switch-in setters (weather / terrain / Intimidate-like), on-contact punishers, absorb / immunity
abilities, residual heal / damage, status and volatile immunities, stat-stage reactions. Genuinely bespoke
mechanics stay as code, but run inside the single-instance stages.

### 3.6 Training build
A compile-time configuration that drops conformance-only machinery:
- the public event log (`events=False`, already measured: −6–12%);
- forced-draw lookup and draw logging (only replays against PS use them);
- harness-only fields.

Random draws are keyed by meaning, not stream position, so with no forced draws the outcomes are **identical**
between builds. A test runs the same seeded battles in both builds and requires bit-identical states.

## 4. Correctness: how outcomes are protected
1. **PS comparison, unchanged contract.** Every stage of the migration must keep every corpus at 100% (state
   projection, every random draw with its key, protocol events). No waivers.
2. **Engine-vs-engine differential test (new, the main safety net).** The refactor must not change any outcome,
   so the old engine (a pinned commit) and the new one must produce **bit-identical states** for the same seeds
   and actions. Random battles at scale (millions, cheap, on the GPU) cover far more paths than the PS corpora.
3. **Targeted corpora before touching a path.** Paths with thin coverage get dedicated PS corpora first:
   forced switches (Roar, Whirlwind, Dragon Tail, Circle Throw — currently none), multiple faints in one action,
   residual chains, switch-ins with several reacting abilities.
4. The profiler and kernel count are recorded per stage in this ADR, so every gain is attributable.

## 5. Plan (each step lands only with §4 green)
| step | content | expected gain (estimate) |
|---|---|---|
| A | Training build (§3.6) + battles/s metric + bit-identical cross-build test + engine-vs-engine differential harness | 5–15%; the safety net for B–F |
| B | Fixed-layout accessors for the hottest call paths (§3.4) | 10–20% |
| C | Continuation + single switch-in / faint / residual stages (§3.2): forced switches resume through the shared switch path | 20–30% |
| D | Request buffers + single boost / status / volatile stages (§3.3) | 10–20% |
| E | Move pipeline stages (begin / hit / end share one damage-apply and one secondary stage) | 10–20% |
| F | Effects-as-data families (§3.5), for the remaining content in the same pass as content lands | keeps cost flat as content grows |

GPU measurements (`bench/gpu_check.sh` reduced to the training build at B = 8k / 16k) after C and after E.
Content work pauses during A–E, so the migration is not chasing a moving target.

## 6. Alternatives considered
1. **A CPU engine (as mikumiku37: Rust, ~7k battles/s per core).** Would mean a second implementation, with its own
   conformance effort, and gives up fused GPU rollouts and search. Not proposed. Measuring this engine on CPU,
   sharded across cores, is cheap and is added to step A as a reference point.
2. **Phase bucketing** (sorting envs by phase and running phase-specific sub-batches): data-dependent shapes; the
   continuation design gets most of its benefit with static shapes. Revisit only if C–E fall short.
3. **Keep the current structure and tune call sites one by one** (ADR-106 next steps): smaller wins, with the
   duplication returning as content grows.

## 7. Open questions for the human
1. Targets T1 / T2 / compile ≤ 3 min — agree, or set different ones?
2. Pause content (step 2 of the plan) during steps A–E?
3. GPU runs at the end of C and E (~1 h each with the training build at two batch sizes)?

## Decision
Accepted by the human on 2026-10-27:
1. Targets T1 (>= 3,000 battles/s), T2 (>= 10,000 battles/s), compile <= 3 min; faster is always better.
2. Content work pauses during steps A-E.
3. GPU timing runs after C and E; keep them short (timing only, no PS comparison): the training build at the
   batch sizes that matter.

## Progress log
- **Step A (2026-10-27).** `scripts/diff/engine_diff.py`: engine-vs-engine differential test (per-field state
  hashes after every decision; step_counter pinned to the decision index so rolls depend only on draw keys).
  Validated: repeat recordings identical; micro_steps 4 vs 8 identical (layout-independent); a deliberate
  Leftovers 1/16 -> 1/15 change detected (7/64 battles, `mons.hp`). `bench/throughput.py` reports battles/s;
  `bench/gpu_time.sh` is the short timing-only GPU run (training build, B = 8k / 16k). The training build needs no
  new switch: forced draws and the draw log are already off by default, `events=False` drops the event log.
  Forced-switch PS corpora added (Roar, Whirlwind, Dragon Tail, Circle Throw, +Intimidate): baseline 100%.
- **Measurement rule (2026-10-27).** Wall-clock CPU throughput decides, not the kernel-op proxy: `bench/throughput.py
  --stage core --batch 1024 --steps 16 --micro-steps 4 --no-events` on this Mac (same settings every time).
  Pilot verified: 83 -> 113 battles/s, compile 190 -> 126 s.
- **Step B reverted.** Select-chain `mget` / `mset` cut ops 2.7% but ran 2.5x slower (113 -> 46 battles/s): each
  access touched all 8 rows (8x the data on wide fields such as vol_state). Gathers stay.
- **Step C (2026-10-27).** Forced switches no longer run inside the runAction tail: the tail records a continuation
  (`scratch.tail_pending / tail_kind / tail_pos`) and the next micro-step drags in through the single switch-in and
  runSwitch sites of `run_light`, then resumes the tail. runSwitch compiled once instead of ~7x.
  **113 -> 146 battles/s, compile 126 -> 106 s.** Differential test identical (92 fields x 512 battles x 120
  decisions); forced-switch and random PS corpora 100%.
- **Local timing is noisy (2026-10-27).** This Mac runs at load ~12 (system agents); best-of-N wall time and even
  process CPU time vary +-10-40% between rounds. `scripts/ab_time.sh` alternates baseline / tree; only large effects
  are resolvable locally. Changes are judged by: differential test (must be identical), op count, and GPU runs.
- **Narrowing (2026-10-27), safe by construction.** (1) runEvent dispatch tables exclude moves' own handlers (never
  candidates; TryHit 12 -> 7 branches). (2) `boost` / `boost_v(simple=True)` at 7 statically self-sourced or
  positive-only call sites: no TryBoost blockers, Defiant / Competitive or Rattled compiled. Ops 194k -> 173k;
  differential test identical; boost-related PS corpora 100% (a_sapsipper's one failure is pre-existing at e9951ed).
- **GPU checkpoint (L40S, 231fbef, training build).** B=8192: **2,117 / 1,930 / 1,694 battles/s** at micro_steps 2 / 3 /
  4 (34.1k / 33.6k / 29.3k decision-steps/s); B=16384 at 4: 1,431 battles/s (was ~650 at the 2026-10-26 run);
  compile + first call 352 s (was 583-630 s). Per-call time fits ~26 ms fixed + ~39 ms per micro-step (B=8192).
  `EngineConfig.micro_steps` default 8 -> 2 (outcomes independent of it: engine-vs-engine test).
- **Weather hoist (2026-10-28).** At micro_steps 2 the once-per-call residual block is ~28% of ops; inside the
  Residual walk every one of the 16 iterations carried the weathers' eachEvent code. Weathers' Residual order (1) is
  strictly below every other handler's (next: 3), so a listed weather row is always position 0: the weather code is
  now compiled into position 0 only (trace-time check from handler_order). Training-build ops 53.4k -> 48.4k.
  Broad differential test identical (92 fields x 1024 battles x 120 decisions, teams from 63 corpora); weather PS
  corpora 100%.
