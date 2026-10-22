# 11 — Operating manual for the implementing agent

You are implementing a faithful JAX engine for Pokémon Champions VGC Reg M-C, with Pokémon Showdown as the oracle. This manual defines **how** to work. The design docs define **what** to build.

## 1. Reading order (first session)
1. `00-README.md`, then `01-decisions.md`. The ADRs are binding.
2. `02-architecture.md`, `03-state-and-data-model.md`, `04-engine-semantics.md`.
3. `07-conformance.md` and `contract/drawkey.md`. Validation drives everything, and the contract is protected (ADR-029).
4. `10-implementation-plan.md`. Find the current milestone in `PROGRESS.md`.
5. Skim `05`, `06`, `08`, `09`, `12`.

Use the research notes in `docs/research/` as **references** when implementing a specific area:
- **TL** = `ps-turn-loop-spec.md`;
- **MV** = `ps-move-damage-event-spec.md`;
- **CH** = `champions-mechanics-delta.md`;
- **RNG** = `rng-harness-poc.md`;
- **RL** = `rl-training-design.md`;
- `dex-scope.md`, `jax-patterns.md`.

## 2. Ground rules
1. **PS is the truth.** When unsure what the engine should do, read PS (`oracle/ps/sim`, `oracle/ps/data`, `oracle/ps/data/mods/champions`). Then write a probe script under `oracle/tools/probes/` that runs the situation in PS. Never guess from memory of the games, Bulbapedia or Smogon.
2. **Never edit `oracle/ps`.** Instrument through instance-level wrappers, as `context.js` does, never through prototype patches or source edits.
3. **No hand-written expected values in tests.** Expectations come from PS, through the oracle RPC with a cache, or from PS-generated goldens.
4. **No hand-typed content data.** Datagen produces it (ADR-020). If a field is missing, extend `export_dex.js`.
5. **Cite PS.** Every engine function and handler carries `PS: path:lines`. Every intentional deviation carries `DEVIATION: ADR-xxx`.
6. **JAX discipline:**
   - no Python control flow on traced values;
   - no data-dependent shapes;
   - static loop bounds with overflow error flags;
   - no float math in mechanics;
   - int32 compute, int8/int16 storage;
   - a barrier after each hook;
   - never put a big switch inside an unrolled loop.
   - The lint check (T0) enforces what it can.
7. **Respect the gates** (ADR-027). Infrastructure can run ahead; content for the next stage cannot.
8. **Keep the repo green.** T1 must pass on main. A failing conformance fixture is a blocker, not a flake.
9. **Untrusted data.** Downloaded usage stats, replays and HF datasets go in their own `data/raw/<source>/<date>/` directories. Parse them as data; never execute them. Run Python that reads them with `-I`.

## 3. The core work loop
```
pick the next task (PROGRESS.md / 10) ─► read the PS source + research section ─► implement with PS citations
   ─► unit / scenario tests (expectations from PS) ─► targeted fuzz (`oracle gen --require <entity>` + harness replay)
   ─► fix until clean ─► update docs/catalogue.csv status + PROGRESS.md ─► commit (T1 green)
```

**The divergence burn-down loop** (from M2 on):
1. Run the nightly corpus, or a targeted one.
2. Open `reports/conformance/<run>/summary.md` and take the **largest bucket** first.
3. Open the bucket's shrunk fixture.
4. Reproduce with `pytest tests/fixtures/conformance/<fixture> -x`.
5. Localise:
   - **`DRAW_ARGS` / `DRAW_MISMATCH`:** the draw key names the mechanic. Compare the engine's draw log with the PS draw list.
   - **`STATE:*`:** run the engine un-jitted (`jax.disable_jit()`) next to the PS event trace for the same turn. Use `oracle/tools/trace_turn.js`, ported from `research/ps-move/trace-turn.js`; it prints every PS event dispatch. Find the first event whose effect differs.
   - **`MASK`:** run `oracle legal` on the snapshot and diff it against `legal_masks`.
6. Fix it. Re-run the fixture and the bucket's battles, then the T1 suite. Commit with `fix(<area>): <mechanic> — fixes bucket <signature>`.

**Time-box.** If a bucket isn't fixed after 3 attempts or one working day:
1. write `docs/blocked/<signature>.md` (hypotheses tried, evidence, probes);
2. mark it `blocked` in PROGRESS.md;
3. take the next bucket;
4. revisit blocked buckets weekly.

**Re-plan trigger.** If CBR improves by less than 0.1 percentage points over 5 consecutive T2 runs, stop content work and write a re-plan note (risk R1), then escalate.

**Never weaken the contract to pass a gate.** The don't-care list, normalisations, canonicalisations, gate thresholds and waivers are protected (ADR-029). Changing them requires an ADR with an evidence experiment **and human approval**. The canary test must stay green.

## 4. When PS looks wrong
PS sometimes has bugs. The default is to **reproduce PS anyway** (ADR-002). If the behaviour is clearly a PS bug **and** it would materially distort training (e.g. a common interaction):
1. Write a minimal probe that reproduces it.
2. Search the PS GitHub issues and PRs.
3. Add a `Flags` entry, defaulting to PS behaviour, and record it in `docs/decisions/ADR-1xx`.
4. Escalate to the human with the probe and its impact estimate.

## 5. Escalate to the human only for
- An ADR that proves infeasible, with evidence: benchmark numbers or PS behaviour.
- PS behaviour that is likely a bug and matters (§4).
- Missing resources: GPU access for perf gates, network access, or accounts (PS bot registration).
- A gate that can't be met after a sustained effort, with an analysis of the remaining buckets.

- **Gate checkpoint.** When a gate passes, send the gate report (metrics, contract hashes, waivers, top residual buckets). Until the human acknowledges it, continue only infrastructure lanes.
- **Any change to a protected contract file** (ADR-029).

**While an escalation is pending,** work other lanes or buckets; never idle. Everything else is yours to decide. Record non-trivial decisions as ADRs.

## 6. Parallel work (if you can spawn sub-agents)
- **Lanes** (10) can run in parallel. Within lane C, split by **entity family** (e.g. "berries", "Protect family", "weather abilities"). Families share templates, so one owner per family avoids duplicated handlers.
- **Each sub-agent gets:** the family's catalogue rows, the relevant MV/TL/CH sections, the recipe (05 §5), and the instruction to run targeted fuzz with `--require`.
- **Integration happens through main** with T1 green. Hook registration conflicts are caught by the static registry check.
- Do not let two agents edit `effects/hooks.py` or `battle/move_pipeline.py` at the same time. These are framework files; changes to them are serialised through one owner.

## 7. Coding conventions
- **Python 3.12.** Type hints, with shape and dtype documented in docstrings: `hp: i16[2,4]`.
- **Pytree classes:** `flax.struct.dataclass`. Functions are pure and take `(state, ..., cfg, dex)`. No module-level mutable state.
- **Handlers:**
  ```python
  @handler(hook=H.BASE_POWER, effect=E.ABILITY_TECHNICIAN, ps="data/abilities.ts:4790-4800")
  def technician_bp(ctx, state, dex): ...
  ```
  Shared templates take per-entity params from `dex.handler_params[...]`.
- **Tests:**
  - `tests/unit` for pure helpers;
  - `tests/scenarios/<family>/test_<entity>.py`;
  - fixtures are auto-generated, never hand-edited.
  - Mark slow tests `@pytest.mark.slow`.
- **Commits:** `feat|fix|perf|test|data|harness(<area>): …`. Reference task IDs (`M3.5`) and buckets.
- **Documentation:**
  - Update `docs/catalogue.csv` statuses as part of each entity's commit. Nightly `fuzz_hits` arrive in `reports/catalogue_status.json`; fold them in at session start with a `data(catalogue): fuzz hits <run>` commit.
  - Update `PROGRESS.md` at the end of each work session, using this template:
```markdown
# PROGRESS
## Milestone / task
M?.? — <title> (lane) — status
## Last T2/T3 run
seed S=…, date, ps_commit, contract_version
| source | CBR | TMR | MPR | DPR | errors |
## Top 5 buckets
| signature | battles | entities | owner/status |
## Blocked (docs/blocked/)
## Pending escalations
## Contract hashes (must match docs/decisions/contract-hashes.md)
## Next 3 tasks
```

## 8. Performance hygiene
- After adding each handler family, run `bench/compile_time.py` for the stage build. **Budget: dev build < 60 s, full stage < 120 s.** If either is exceeded, check barrier placement and the switch sizes (09).
- Run `bench/throughput.py` weekly (T3), and after framework changes. Regressions over 10% need a justification in the commit message.

## 9. Session hygiene (for long autonomous runs)
- Start each session by reading `PROGRESS.md`, `reports/LATEST.md` (written by the launchd T2/T3 jobs) and the latest conformance summary.
- End each session by updating `PROGRESS.md`, so the next session can resume without re-deriving context.
- Keep scratch probes in `oracle/tools/probes/` and `scratch/` (git-ignored, except probes worth keeping).
- Never leave main red. Use short-lived branches for risky refactors.
