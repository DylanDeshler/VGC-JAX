# Implementability review of the vgc-jax design package

**Reviewer's lens:** I am the agent who must implement this alone, starting tomorrow. The review covers implementability, completeness and process.
**Scope:** `design/00–12`, `CLAUDE.md`, `catalogue.csv` and `tools/make_catalogue.py`, checked against `research/poc`, `research/dex`, `rl-training-design.md` (RL) and `jax-patterns.md` (JP).
**Date:** 2026-10-21. Every number below was checked against files on disk.

## Verdict

- **M0 can start tomorrow, with two caveats.**
  - M0.1, done literally, produces a broken and legally questionable repo (item 1).
  - M0.5 and M0.7 hide decisions about stage lists and data sources (items 5 and 6).
- **M1 cannot reach its "Done when" criteria as written.**
  - Its golden tests depend on oracle RPCs that no task builds (item 3).
  - The damage goldens conflict with "no hooks" (item 21).
- **G-core, the first gate on the critical path, cannot be measured as written** (item 2).
- **The RL API is close.** Before M5 it needs `final_obs`, a coherent truncation rule, a decision on auto-resolve and a defined `Masks` pytree.
- **There is no performance plan that works without a GPU** (item 19).
- **The biggest process risk is that gates are self-graded against contract files the agent can edit** (item 14).

## How to read the items

Each item gives a severity, the file and section, the problem, and a fix (with exact text where possible).

**Severity:**
- **BLOCKER:** following the design literally fails, or a gate cannot be evaluated.
- **MAJOR:** costs significant rework, or forces the implementer to make an unrecorded decision on a load-bearing point.
- **MINOR:** an inconsistency or a small gap.

---

## BLOCKER

### 1. BLOCKER · `10` M0.1, `00` §4, `02` §2 (`docs/research/`) — "copy `research/` → `docs/research`" would commit 1.35 GB of venvs, third-party code and third-party content

**Problem.** `research/` is about 1.35 GB:
- `jax-bench/.venv`: 432 MB;
- `jax-bench/.venv-metal`: 525 MB;
- `jax-bench/ext`: 242 MB of cloned gymnax, jumanji, minimon and pgx repos, each with its own licence;
- `dex/raw`: 110 MB of Smogon chaos JSON, the 26 MB VGC-Bench HF log file, and replay team files that contain **player usernames**;
- `champions/web`: downloaded Bulbapedia (CC BY-NC-SA), Serebii and official pages.

Copied literally, the repo can't be pushed to GitHub (the venvs contain single files over 100 MB) and it redistributes third-party content and personal data. The README also says `../jax-vgc-engine-scope.md` is copied, but it lives outside `research/`. ADR-016 cites it as evidence.

**Fix.** Replace the M0.1 task text with:

> M0.1 Create the repo layout (02 §2). Copy `design/` → `docs/design/`, `design/CLAUDE.md` → `CLAUDE.md`, `design/catalogue.csv` → `docs/catalogue.csv`, and `../jax-vgc-engine-scope.md` → `docs/research/`. Into `docs/research/` copy only: `*.md`; `poc/{lib,tools,scripts,results,rng_callsites.json}`; `dex/*.json`; `dex/scripts/` (minus `__pycache__`); `dex/out/`; `ps-loop/`; `ps-move/`; `champions/*.{js,py,json,txt}` and `champions/diff/`; `jax-bench/*.{py,sh}` and `jax-bench/results/`. **Never copy** `jax-bench/.venv*`, `jax-bench/ext/`, `**/__pycache__`, `champions/web/` or `dex/raw/`. Move `dex/raw/*` to `data/raw/` (git-ignored) per item 5. Add these paths to `.gitignore`. Done when the tree matches 02, `du -sh .git` < 50 MB, and `git ls-files | xargs du -ch | tail -1` < 50 MB.

### 2. BLOCKER · `07` §12.1 G-core, `10` M2.9 — the "core subset" can't be built as specified, and "curated inert" abilities can't be verified

**Evidence** (census `research/dex/handler_census_legal.json`, legal pool `research/dex/legal_pool.json`):

- **Moves.** 60 of the 260 DATA_ONLY moves have `data_refs` to status or volatile conditions that have handlers. M2 doesn't implement those conditions; they arrive in M3.3–M3.4. Examples: Heat Wave, Rock Slide, Thunderbolt, Ice Beam, Flamethrower, Scald, Thunder Wave, Will-O-Wisp, Sleep Powder, Swagger. The references are to flinch, brn, par, confusion, psn, slp, frz and tox.
  - 5 more DATA_ONLY moves have `core_refs` into `sim/`: explosion, selfdestruct, feint, ironhead and toxic.
  - Only **197** DATA_ONLY moves are truly hook-free.
- **Abilities.** Every mon needs an ability, but only **6 abilities are DATA_ONLY**: battlearmor, corrosion, earlybird, levitate, shellarmor and stall.
  - Four of them still need engine code: Levitate (grounding, `sim/pokemon.ts`), Stall (−0.1 fractional priority), and Battle Armor / Shell Armor (`onCriticalHit: false`).
  - They are legal on only **20 species**, about 15 after the Species Clause collapses the 6 Rotom formes.
- **No strictly inert alternative exists.** Abilities such as Swift Swim or Run Away have callbacks that *run* on every speed or trap event and return `undefined`. So the coverage counters (07 §9.1, "count whenever a callback actually runs") can't prove they are inert. Those counters don't exist until M3.8 anyway.
- **Wrong examples elsewhere.** 07 §10.2's "inert" example, Ball Fetch, is not legal in Reg M-C. Run Away is a [C] CUSTOM handler.
- **No `core` stage exists.** `EngineConfig.stage` and `oracle gen --stage` have no `core` value.

**Fix.** Replace G-core's "(DATA_ONLY entities only)" and M2.9 with:

> **Core subset (stage `core`, emitted by `dex_stages.py --stage core`):**
> - **Moves:** class DATA_ONLY with census `data_refs == {}` and `core_refs == {}` (197 at d601443).
> - **Items:** none (empty slot).
> - **Abilities:** the 6 DATA_ONLY abilities. M2 implements Levitate in the grounded check (H9), Stall in fractional priority (04 §5.2), and the Battle/Shell Armor crit veto. Corrosion and Early Bird are inert without status.
> - **Species:** those that can legally hold one of these abilities (20 at d601443). No Mega stones.
> - **Extending the set:** additional abilities may be added to `tools/datagen/core_inert.json` only through an ADR showing CBR = 100% on 10k core battles with that ability present and ignored by the engine.
> - **Wiring:** add `"core"` to `EngineConfig.stage` and `oracle gen --stage`.
> - **"Core RNG sites"** = the `rng_callsites.json` ids of ACCURACY, CRIT, DAMAGE_ROLL, MULTIHIT_COUNT, RANDOM_TARGET, SECONDARY, SPEED_TIE_ACTION_ORDER, SPEED_TIE_SWITCHIN_ORDER and SPEED_TIE_FIELD_EVENT, each with ≥ 50 hits in the gate corpus.
> - **Expectation:** this pool is narrow. G-core proves the state machine, not coverage.

---

## MAJOR

### 3. MAJOR · `10` M0.4 vs M1.2/M1.3/M1.5/M2.2/M2.6/M3.1/M3.9/M5.6 — eight oracle RPCs are required by "Done when" criteria but no task builds them

**Problem.** M0.4 builds only `run`, `snapshot`, `from_snapshot` and `validate_team`. These RPCs are needed but never assigned:

| RPC | First needed by |
|---|---|
| `fixedpoint` | M1.2 |
| `stats` | M1.3 |
| `damage` | M1.5 (needs crafted battles with forced draws, i.e. most of `craft`) |
| `sort_actions` | M2.2 |
| `legal` and `gen --legal on` | M2.6 (MPR) |
| `replay` (forced by DrawKey) | M2.8, M3.9, M5.6 |
| `craft` | M2.4 scenario tests (item 4) |
| `resolve_priority` | M3.1 |
| `trace_events` | the debugging loop (11 §3) |

M0.4 also hides real work. The 07 §3 trace needs the full DrawKey (`step_tag`, `action_ord`, `sub`) and semantic tie records (07 §4.3). The POC key is different: it uses step labels such as `move:p2a:<move>`, and its ties are shuffle draws. Converting ties to winner records means instrumenting `speedSort` callers. The 1.5 EW estimate for M0 doesn't cover this.

**Fix.** Add:

> **M0.8 Oracle test-support RPCs (lane O, about 1 EW).**
> - **RPCs:** `fixedpoint`, `stats`, `sort_actions`, `legal` (plus `gen --legal on`), `replay`, `craft` (minimal: teams, leads, hp, boosts, pp, turn, active_turns), `damage` (core-subset sets only until M3, see item 21) and `trace_events`. `resolve_priority` may slip to M3.1.
> - **Done when:** each RPC has a round-trip test, and `bench/oracle_legal_cost.js` records the per-decision cost of acceptance enumeration (the POC `08_masks` recorded no timing).
>
> Raise M0 to about 2.5–3 EW. In M0.4, add "DrawKey per 07 §4.1 and tie records per 07 §4.3 (new: instrument `speedSort` callers)".

### 4. MAJOR · `10` M0.5, M2.4/M2.5, M3 ordering — circular and hidden dependencies

**Problems.**
1. **M2 scenario tests come before their tooling.** M2.4 and M2.5 require scenario tests, but the crafted-scenario builder, the `craft` RPC, the scenario runner and the oracle cache are in M3.8. The projection comparison (`harness/project.py`) is in M2.8, after M2.4.
2. **M0.5's self-tests depend on later milestones** (06 §3):
   - self-test 1 compares PS against **engine tables**, which arrive in M1.1;
   - self-test 3 requires every handler to be classified, which happens in M3.2.
3. **M3's infrastructure comes after the content that needs it.** M3.8 (coverage, steering, `--require`, scenario runner) and M3.9 (shrinker, fixtures) are listed after M3.3–M3.7. Those content tasks need `oracle gen --require` (05 §5 step 5) and the shrinker.
4. **The projection map is placed too early.** 06 §1 lists `projection_map.json` as a codegen output (M1.1). 07 §5.3 says it is generated from a 50k-battle corpus *and* the engine field layout, which arrives in M2.1.

**Fix.**
- Insert **M2.0 Harness core** before M2.2:
  - `harness/project.py` (both directions, core fields);
  - `harness/oracle_client.py` with the content-hash cache;
  - the scenario-test runner using `craft` and `run` (from M0.8).
- Change M0.5's Done-when to: "self-tests 2 and 4 pass. Self-test 1 joins at M1.1. Self-test 3 joins at M3.2, scoped to the current stage."
- Reorder M3 as: M3.1 → M3.8 → M3.9 → M3.2 → M3.3–M3.7.
- In 06 §1, move `projection_map.json` to "generated in M2.8 by `harness.cli gen-projection-map` from a core corpus; re-run per stage".

### 5. MAJOR · `06` §4–5, `07` §12.1, `10` M0.5/M0.7 — the data needed for M0 is on disk, but nothing points to it, and the real-sheet corpus isn't frozen

**What exists on disk:**

| Need | On disk now | Notes |
|---|---|---|
| Chaos stats, Reg M-C 2026-09 | `research/dex/raw/stats_2026-09/` (8 files: `regmc` and `regmcbo3`, each at cutoffs 0/1500/1630/1760) | https://www.smogon.com/stats/2026-09/chaos/ |
| Chaos stats, Reg M-B 2026-08 | `research/dex/raw/stats_2026-08/` | used only for the robustness check |
| Real Bo3 sheets | `research/dex/raw/replays_2026-10/gen9championsvgc2026regmcbo3_teams.jsonl` | 3,850 rows → **3,211 unique** packed teams. **0 of 19,266 mons carry SP.** 977 mons have a blank gender. Rows contain usernames. Several fetches failed (`replays_bo3.log`) |
| Real Bo1 sheets | `..._regmc_teams.jsonl` | 146 rows |
| VGC-Bench Reg M-B logs | `research/dex/raw/vgcbench/logs_gen9championsvgc2026regmbbo3.json` (26 MB) | HF `cameronangliss/vgc-battle-logs`; licence not checked |

**Problems.**
- The gate text "all 3,211+ real teams" is a moving target: re-running the scraper changes it.
- The real sheets need deterministic SP and gender fills, which aren't specified.

**Fix.** Add 06 §4.0:

> **Seed data (M0.1/M0.7).**
> - **Location.** Move `research/dex/raw/*` to `data/raw/{smogon-chaos/2026-09, smogon-chaos/2026-08, ps-replays/2026-10, vgcbench/2026-10}/`, with a `SOURCES.md` (URL and retrieval date) and a `MANIFEST.sha256`. `data/raw/` is git-ignored; keep a tarball outside the repo.
> - **Frozen real-sheet corpus.** `data/teams/real_sheets_mc_v1.jsonl` = the 3,211 unique Bo3 packed teams with usernames removed and sha256 recorded. G-v0, G-v1 and G-v2 use this frozen file. New scrapes go to `real_sheets_mc_v2` and enter gates only at a milestone boundary, by ADR.
> - **Fills.** SP is filled per 06 §4.2 with seed `sha256('spfill', canonical_sheet)`. Blank genders are filled with the species' fixed gender, or `M` if the species allows it, else `F`.

### 6. MAJOR · `05` §1, `06` §1, `10` M0.5, `catalogue.csv` — stage lists are under-specified, the seed catalogue disagrees with the stated stage sizes, and monthly regeneration moves the gate target

**Problems.**
- **Only v1 is emitted, and its threshold is adaptive.** `dex_select.py` writes only `v1_*.json`. It picks v1's τ by `meets()` (the smallest τ meeting the coverage targets), not a fixed τ.
- **Undocumented upstream step.** `dex_select.py` depends on `usage_coverage.py` having written `out/synthetic_teams.json`. 06 doesn't mention this step.
- **Two different v0 definitions.** The catalogue's stage column (`make_catalogue.py`: non-CUSTOM in the v1 list, or p ≥ 0.01 *including replay p_team*) gives v0 = **387 moves / 107 abilities / 145 items**. 05 §1 and `dex_options.json` (`cost_aware_tau=0.01`) give **377 / 110 / 142** and 220 CUSTOM.
- **Moving target.** 05 says stages are re-run "on every new month of usage stats and on every re-pin". Mid-M4, that can move entities into v0.

**Fix.** Replace the 05 §1 "Authoritative lists" bullet with:

> **Authoritative lists.** `tools/datagen/dex_stages.py` ports `usage_coverage.py` (run first) and `dex_select.py`.
> - **Frozen inputs:** chaos `2026-09` `regmcbo3-1760` and `regmc-0`; `data/teams/real_sheets_mc_v1.jsonl`; census and legal pool at the pinned commit.
> - **Outputs:** `stages/{core,v0,v1,v2}.json`. v0 and v1 come from `cost_dex({'bo3_1760': τ, 'bo1_0': τ})` with τ = 0.01 and 0.001. v2 is the whole legal pool. `core` follows item 2.
> - **Self-check at d601443:** v0 = 377/110/142 and 220 CUSTOM; v1 = 454/172/160 and 376.
> - **Catalogue:** regenerate `docs/catalogue.csv`'s `stage` column from these lists. The seed catalogue's stage column is approximate.
> - **Freeze:** stage lists stay frozen for a milestone. Re-run them on new stats only at a gate boundary, recorded by ADR.

### 7. MAJOR · `01` ADR-021, `10` "Re-pin and regulation events", `12` R2 — Reg M-C ends before G-v0, monthly re-pins conflict with hard gates, and nobody decides what the target is

**Problem.**
- **Timing.** Today is 2026-10-21. Reg M-C ends **2026-12-02**, but G-v0 is 11–15 EW away (about February 2027). ADR-021 requires a re-pin "on every regulation change", which would land the new regulation's data in the middle of M3 or M4.
- **Gate conflict.** The merge rule ("re-pin lands only when gates are back at their previous level") has no previous level before G-v0. During a long burn-down, a monthly re-pin can block all progress.
- **Format removal.** A later PS commit may drop `gen9championsvgc2026regmc` entirely.

**Fix.** Add ADR-028:

> **ADR-028 — Target freeze until G-v0.**
> - Conformance targets `gen9championsvgc2026regmc` at `d601443` until G-v0 passes.
> - No scheduled re-pins happen during M0–M4. An unscheduled re-pin is allowed only when a PS fix removes a bucket we would otherwise have to emulate, by ADR.
> - When the next regulation is published, run datagen for its format id on a branch and record the delta in `docs/oracle-changelog.md`, but don't switch targets.
> - After G-v0, monthly re-pins resume. A regulation switch is its own mini-milestone: G-v0 is rerun on the new format, and the M-C engine remains available via its DATA_VERSION.
> - `export_dex.js` and `dex_stages.py` take `--format`. Nothing hard-codes `regmc`.

### 8. MAJOR · `07` §3/§7/§10.2, `06` §6 — no versioning or invalidation scheme for traces, fixtures or the oracle cache across re-pins and contract changes

**Problem.**
- **Partial versioning.** Traces have `trace_version: 1` and fixtures store `ps_commit`, but neither records the DrawKey or projection contract version.
- **Stale cache keys.** The oracle cache is "keyed by `ps_commit`" only. A fix to `classify.js` or `project.js` silently serves stale cached expectations.
- **No re-pin procedure for fixtures.** When PS's behaviour changes, a fixture can start "failing" for a legitimate reason. A new PS serializer can also break `deserializeBattle` on old snapshots. Re-pins are mandatory (ADR-021), yet nothing defines what to do with old fixtures.

**Fix.** Add 07 §10.3:

> **Versioning.**
> - Every trace, fixture and cache entry carries `ps_commit`, `oracle_git` (the hash of `oracle/lib/**`) and `contract_version`. `contract_version` is bumped on any change to the DrawKey, the don't-care list, the projection or the normalisations.
> - Oracle cache key = `sha256(ps_commit, oracle_git, contract_version, canonical_request)`. `tests/.oracle_cache/` is not committed; T1 rebuilds it.
> - Gates use only traces whose `contract_version` is current.
>
> **On re-pin**, `harness.cli refresh-fixtures` re-runs each fixture's snapshot, choices and forced draws on the new PS:
> - if the PS projection is unchanged, it updates `ps_commit`;
> - if the projection changed, it moves the fixture to `tests/fixtures/conformance/_repin/<old_commit>/` and opens a `PS_CHANGE` bucket with the PS commit range;
> - if the snapshot doesn't deserialize, it regenerates the fixture from its source trace (fixtures keep `source_battle_id`), or deletes it with a changelog line.
>
> Fixtures are never deleted by hand.

### 9. MAJOR · `07` §2–3, `02` §4.1 — no reproducibility or seeding policy for the oracle or the tests

**Problem.**
- **Non-reproducible ids and seeds.** `battle_id` is a `<uuid>`. Only a global `--seed 1` exists.
- **Undefined derivations.** Nothing says how per-battle PS seeds, team-sampler seeds, policy-RNG seeds or SP-fill seeds derive from the run seed.
- **Scheduler dependence.** With a worker pool, battle-to-worker assignment depends on scheduling, so two runs of `gen --seed 1` can differ.
- **Unseeded tests.** Hypothesis and natural-mode tests are also unseeded.

Without fixed seeds, "reproduce bucket X from last night" is impossible.

**Fix.** Add 07 §2.1:

> **Determinism.**
> - For run seed `S` and battle index `i`:
>   - PS seed = `sodium,` + `sha256('battle',S,i)[:32]`;
>   - team seed = `sha256('team',S,i)`;
>   - policy seed = `sha256('policy',S,i)`;
>   - `battle_id = sha256(S,i,ps_commit,oracle_git)[:16]` (replaces the uuid).
> - Worker w processes `i ≡ w (mod W)`. Each shard holds a fixed index range and is sorted by `i`. Outputs are byte-identical for any `--workers` value.
> - Every report records `S` and the full command line.
> - Engine natural-mode tests use `PRNGKey(i)`. Hypothesis runs with `derandomize=True` in T0/T1 and a random seed (printed) in T2.

### 10. MAJOR · `07` §12.2, `11` §2.8/§9, `CLAUDE.md` — CI infrastructure, repo location and scheduling are undefined

**Problem.**
- **Undefined tooling.** "CI tiers", "every PR", "merges" and "nightly" are used throughout, but no CI system, remote, repo path or scheduler is named.
- **Hosted CI doesn't fit.** T2 ("< 6 h on the dev box") and T3 ("< 24 h") exceed hosted-runner limits and need the dev box.
- **Nothing triggers "nightly".** An autonomous agent runs in sessions, so nothing would actually start a nightly run.
- **Submodule build gaps.**
  - The way the PS submodule is built and cached in CI is unspecified.
  - `node build` and `npm ci` write `dist/` and `node_modules/` into the submodule, so `git status` reports `oracle/ps` as dirty.
  - The POC used `npm install`; `npm ci` needs a lockfile in `oracle/ps`.

**Fix.** Add 02 §6:

> **Execution environment.**
> - **Repo.** `~/Documents/research/pokemon/vgc-jax`, git, branch `main`. A private remote is optional; there are no hosted runners.
> - **CI = local make targets.**
>   - `make t0` runs as a git pre-commit hook.
>   - `make t1` must pass before any merge to `main` (`scripts/merge.sh` enforces this).
>   - `make t2` (nightly, 02:00) and `make t3` (Sunday) run from launchd agents in `ops/launchd/` and write `reports/LATEST.md`, which every session reads first.
> - **Toolchain pins.** Node is pinned in `.nvmrc` (26.8.1, the version used in research). `make oracle` = `cd oracle/ps && npm ci && node build`; if `oracle/ps/package-lock.json` is absent, use `npm install --no-save` and record `npm ls --all` in `oracle/ps-deps.lock.txt`.
> - **Submodule hygiene.** `.gitmodules` sets `ignore = untracked` for `oracle/ps`. Cache `node_modules/` and `dist/` keyed by the submodule SHA.

### 11. MAJOR · `07` §1, §8, §12.1 — gate metrics are ambiguous after the first divergence and for sampled mask checks

**Problem.**
- **No post-divergence rule.** Replay "records the first failure per battle" (07 §8), but TMR, MPR and DPR are defined per decision. Nothing says whether decision points after a battle has diverged count.
- **MPR over sampled checks.** The mask check runs on only 10% of nightly battles, so it's unclear what MPR is computed over.
- **G-core's MPR requirement.** "100%" implicitly requires `--legal on` for all 50k core battles, at a cost nobody has measured.
- **Unpinned compile budget.** "compile < 2 min" names no hardware and no cache state.

**Fix.** Replace the 07 §1 metrics table with:

> For battle b, let **f(b)** be the first decision index at which state, recorded masks, draws or error flags fail, with f(b) = ∞ if the battle is clean. Only decision points k ≤ f(b) are counted.
> - **CBR** = #{b : f(b) = ∞} / #b.
> - **TMR** = Σ_b #{k < f(b)} / Σ_b #{k ≤ f(b)}.
> - **MPR** = over k ≤ f(b) with `legal` recorded, the fraction of equal sets.
> - **DPR** = over steps k ≤ f(b), the fraction of equal draw multisets.
> - **Error rate** = the fraction of battles with any error flag at k ≤ f(b).
>
> G-core runs `--legal on` for 100% of its corpus. Compile budgets are measured on the M4 Max with a cold persistent cache (`JAX_COMPILATION_CACHE_DIR` emptied).

### 12. MAJOR · `07` §12.1, `10` G-v0/G-v1 — gate corpus composition, sizes and policies are under-specified

**Problem.**
- **Corpus split unclear.** "≥ 200k battles" doesn't say whether it is per CBR column or in total.
- **Real-sheet corpus undefined.** "Real-sheet corpus, random and heuristic policies" doesn't say how many battles each team plays or against whom.
- **Heuristic policy missing.** The `heuristic` policy ("Protect when threatened") is undefined, isn't in the POC (`player.js` has only `randomChoice`) and is assigned to no task.
- **Overfitting risk.** Nothing forbids gating on the same corpus seeds used for debugging.

**Fix.** Add to 07 §12.1:

> **Gate run** (a fresh seed never used for debugging):
> - **(a)** 200k pool-B battles under the random policy → CBR column 1.
> - **(b)** 100k battles: 70% pool D, 30% crafted → CBR column 2.
> - **(c)** The frozen real-sheet corpus restricted to the stage (about 2,630 teams for v0). Each team plays 10 battles against uniformly drawn other real sheets under `random`, and 10 under `heuristic` (about 52k battles). It must meet column 1's threshold.
> - **Mask checks:** `--legal on` for 10% of (a)+(b) and 100% of (c). DPR and error rate are counted on all battles.
> - **G-v1/G-v2:** scale (a) to 1M.
>
> **`heuristic`** (oracle-side, deterministic given the policy seed):
> - Mega whenever it is legal.
> - On turn 1, Fake Out if legal.
> - Otherwise Protect with p = 0.25 if legal and not used last turn.
> - Otherwise prefer a spread move when both foes are active.
> - Otherwise choose uniformly among damaging moves.
> - Switch with p = 0.05.
>
> Implement it in `oracle/lib/player.js` (new task M3.8b).

### 13. MAJOR · `07` §9.3, `01` ADR-026, `05` §2 — coverage gate and "done" criteria are partly unmeetable and partly undefined

**Problems.**
1. **The pair gate can't be met as defined.** The pair list comes from census `mentions` (959 mention pairs in the legal pool). Many are spurious: for example, `protect` lists `mentions_ability: ['stall']`, but that's the `stall` *condition*. Others are code checks rather than co-dispatch. Pairs, however, are *counted* as "co-occurring in the same `runEvent` dispatch". Some listed pairs can never be hit.
2. **No waivers.** "Reachable" is undefined for handlers and sites, and there is no waiver process.
3. **Contradictory thresholds.** "0 open divergences" per entity, together with signatures that list the entities involved, effectively requires CBR ≈ 100%. That contradicts the CBR ≥ 99% threshold, and the precedence is unstated.
4. **Undefined "fuzz hit".** It could mean callback invocations, battles, or team presence. The difference is about 100×.

**Fix.** Replace the 07 §9.3 bullets with:

> - **Handler coverage.** Every BESPOKE handler and ROW entity in the stage has ≥ 50 **fuzz hits**. A fuzz hit is a distinct battle in the gate corpus in which the entity's callback ran at least once (handler entities) or its data effect applied (data entities).
> - **RNG sites.** Every RNG site with `reachable ∈ {yes, cond}` has ≥ 50 hits.
> - **Pairs.** Every pair in `docs/coverage/pairs_<stage>.json` has ≥ 10 hits. The list holds co-dispatch pairs observed at least once in the M3.8 1M-battle coverage run, plus census mentions confirmed reachable by a probe.
> - **Waivers.** A target can be waived only in `docs/coverage/waivers.json`, with a probe under `oracle/tools/probes/` and an ADR. Every gate report lists all waivers.
> - **Open divergence.** A bucket with ≥ 1 battle in the latest gate run. Per-entity 0-open-divergence and corpus CBR must **both** hold. Buckets whose signature lists no entity (SYSTEM) count only against CBR.

### 14. MAJOR · `07` §4.4/§5.4/§6, `11` §5, `01` ADR-027 — gates are self-graded against contract files the agent can edit

**Problem.** The agent owns:
- the don't-care list;
- the projection normalisations and dropped fields;
- the mask canonicalisations;
- the coverage goals and gate thresholds.

Under gate pressure, adding a kind to don't-care or a field to "bookkeeping" is the cheapest way to pass. Nothing detects this, and 11 §5 says "everything else is yours to decide". Nothing checks that the harness can catch bugs at all; a bug that silences diffs would look like success.

**Fix.** Add ADR-029 and an 11 §5 bullet:

> **ADR-029 — Protected conformance contract.**
> - **Protected files:**
>   - `harness/contract/{dontcare.json, degenerate.json, normalisations.json, canonicalisation.json, projection_fields.json}`;
>   - `harness/gates.yaml` (thresholds, corpus sizes, coverage goals);
>   - `docs/coverage/waivers.json`.
> - **Changes** need an ADR with the evidence experiment (for example, the 07 §4.4 0%-impact run) **and human approval**.
> - **Hashes.** `harness.gates` prints the sha256 of each protected file in every gate report. A pass counts only if the hashes match `docs/decisions/contract-hashes.md`, which only the human edits.
> - **Canary test (T1).** `tests/harness/test_canary.py` runs 200 core battles with the test-only flag `cfg.flags.canary_crit_2048` (crit = ×1.25) and asserts that the harness reports ≥ 1 `DRAW_ARGS` or `STATE` failure.
> - **Fixtures** are removed only by `refresh-fixtures` (item 8).

### 15. MAJOR · `03` §5, `04` §5.5/§8, `07` §3–4, `01` ADR-012 — the DrawKey contract is stated four different ways, and kind names don't match the inventory

**Inconsistencies:**
- **`step_tag` values.** 03 §5 lists 5 (COMMIT, A, B, RESIDUAL, SWITCHIN). 07 §4.1 lists 7 (A_SWITCH, A_MEGA, A_OTHER, B_MOVE, …).
- **Coordinate count.** ADR-012 has 8 coordinates, without `subject` or `sub`.
- **Tie ordinal.** 04 §5.5 keys SPEED_TIE_SELECT by `(turn, block, selection_index)`. 07 §4.3 uses `sub` = the executed-action ordinal in the turn, *across* blocks and excluding runSwitch.
- **Trace-only fields.** The 07 §3 trace key has `step_actor` and `step_subject`, which aren't in the DrawKey.
- **Generated names won't match.** 04 §8.3 says DrawKind names are generated from `rng_callsites.json`, but the inventory uses different names: `QUICK_CLAW` and `PRIORITY_PROC` (not FRAC_PRIO), `SPEED_TIE_ACTION_ORDER` (not SPEED_TIE_SELECT) and `SPEED_TIE_SWITCHIN_ORDER`.
- **Shared site.** `CONFUSION_SELFHIT_DAMAGE_ROLL` has no site of its own; it is `sim/battle.ts:2409` called from `getConfusionDamage`.
- **Missing kind.** `SPEED_TIE_ABILITY` at `data/abilities.ts:2483` (Magician, `reachable: yes`) is in neither the kind table nor the don't-care list, so it would surface as UNMAPPED.

**Fix.** Create a normative `docs/design/contract/drawkey.md` and make 03 §5, ADR-012, 04 §5.5/§8 and 07 §4 reference it instead of restating the key. It should contain:
- the 10-field key, with the 7-value `step_tag` (3 bits, matching 04 §8.1 packing);
- `sub` for SPEED_TIE_SELECT per 07 §4.3, which overrides 04 §5.5;
- an explicit **inventory-kind → DrawKind table** covering all 52 reachable sites, plus an "unreachable" list with assertions (GMAX, QUICK_CLAW_GEN2/3, MOVE_MAGNITUDE, ITEM_STARF_STAT, …);
- a note that `step_actor` and `step_subject` are informational.

`contract_version` (item 8) covers this file.

### 16. MAJOR · `08` §1–2 vs `04` §2 — auto-resolve is the training default, but `step` doesn't implement it, it would multiply step cost, and conformance never tests it

**Problem.**
- **Not specified.** `auto_resolve_forced=True` is the default, but 04 §2's `step` has no auto-resolve loop.
- **Cost.** Resolving a decision internally means running another decision cycle (Blocks A/B/C) inside `step`. Under vmap every battle pays for every iteration, so even a bound of 1 roughly doubles per-step cost and compile time.
- **Untested.** Conformance runs with it off, so the training-mode path is never checked against PS.
- **Small payoff.** Single-option decisions are a fraction of the 2.91 replacement phases per game (RL §8.1).

**Fix.** In 08 §1, change the default to `auto_resolve_forced: bool = False`. In 08 §2 add:

> Auto-resolve is not part of `step`. Trainers drop samples where `has_decision == [False, False]` from all losses. Implement auto-resolve only after a benchmark shows these steps exceed 5% of rollout steps, by ADR. Any implementation must pass `tests/property/test_autoresolve.py`: the trajectory equals the parity-mode trajectory with the single legal actions inserted.

### 17. MAJOR · `08` §5, `03` §5 StepOut, `08` §6 AutoResetEnv — `final_obs` is missing and the truncation semantics are contradictory

**Problem.**
- **`final_obs` doesn't exist.** 08 §5 says "`final_obs` is provided by the autoreset wrapper", but `StepOut` has no such field and `AutoResetEnv.step` returns `(EnvState, StepOut, EpisodeStats)`.
- **Contradictory turn cap.** At the cap the engine applies the tiebreak outcome to `reward` *and* sets `truncated=True`. A PPO trainer would then bootstrap V(final_obs) on top of a terminal outcome. "tiebreak" vs "draw" doesn't resolve it.

**Fix.** Replace 08 §5's truncation bullet with:

> `turn_cap_outcome ∈ {"tiebreak", "draw", "truncate"}`:
> - **`tiebreak`:** `terminated=True`, `truncated=False`, reward = the tiebreak result.
> - **`draw`:** `terminated=True`, reward 0.
> - **`truncate`:** `terminated=False`, `truncated=True`, reward 0. The trainer bootstraps from `final_obs`.
>
> `StepOut` gains `final_obs: Obs[2]`. It equals `obs` except on steps where `AutoResetEnv` reset the battle, where it holds the pre-reset observation.

### 18. MAJOR · `10` M5 — M5 is allowed to start after G-core, but M5.5 needs v0 teams, and its gate is undefined

**Problem.**
- **Blocked by ordering.** M5 "can start after G-core", but M5.5 trains on "v0 teams", which raise `ERR_UNSUPPORTED_EFFECT` until G-v0.
- **Undefined gate terms.** Nothing defines the "max-damage heuristic", "a short run", the number of evaluation games or seat handling.
- **Wrong lane for the heuristic.** A true max-damage policy needs `damage_calc`, which is M8.

**Fix.** Change M5.5's Done-when to:

> - **Teams:** core-subset teams before G-v0; v0 teams after.
> - **Training budget:** PPO for 5×10⁷ env steps (≤ 8 h on the dev CPU, sharded).
> - **Targets:** ≥ 95% win rate vs uniform random, and ≥ 70% vs `MaxBPPolicy`. `MaxBPPolicy` picks the legal move maximising base power × STAB × type effectiveness against the current foe target, using public info only, with ties going to the lowest index, and never switches.
> - **Evaluation:** 2,000 games with seats alternated; report the 95% CI lower bound.
> - **Re-run** on v0 teams after G-v0.

### 19. MAJOR · `09` §1/§5, `10` G-v1/M9, `11` §5 — no performance plan or proxy gate without a GPU

**Problem.**
- **The GPU gate never binds.** Every throughput gate is on an H100. G-v1 accepts "CPU proxy recorded with GPU gate pending", so it is never binding.
- **No fallback.** M9 is entirely GPU work. If no H100 ever appears, the project has no performance acceptance at all, and the agent learns about a launch-bound design too late.
- **Unused tooling.** `research/jax-bench/hlo_kernels.py` exists (09 says "if present") and gives a CPU-measurable proxy for the GPU's binding constraint, kernels per step (JP §8.3).

**Fix.** Add 09 §1.1:

> **CPU proxy budgets** (enforced at G-v0 and G-v1 whenever no GPU result exists; calibrate once a GPU is available):
> - **(a) Kernel count.** Kernel-like ops per env step in the optimised train-profile HLO ≤ 10,000, measured by `bench/hlo_kernels.py` (ported). The toy is 980. At JP §8.3's 2–5 µs launch floor this gives ≤ 50 ms per step, i.e. ≥ 1.3 M env-steps/s at B = 64k.
> - **(b) CPU throughput.** Sharded CPU (10 P-core devices, B = 32k), train profile: ≥ 5×10⁴ env-steps/s (the toy reaches 1.38 M).
> - **(c) Compile.** ≤ 120 s.
>
> **Escalation.** Request GPU access at **M4.3**, not M9: a one-day cloud H100 rental is enough to run 09 §5 steps 1–3. If no GPU exists by G-v1, M9 becomes "CPU-only hardening" against (a)–(c), and every gate report states "GPU gate pending". Do not use jax-metal (JP §8.4).

### 20. MAJOR · `11` §3/§5/§7, `12` R1 — weak feedback loops for a long autonomous run

**Problem.**
- **No time-box.** Nothing limits how long the agent works on a stuck bucket.
- **Re-plan trigger not in the manual.** R1's "re-plan if burn-down stalls 2 weeks" is in neither 11 nor PROGRESS.
- **No human checkpoint.** A passing gate doesn't stop for sign-off.
- **No `PROGRESS.md` template.**
- **No escalation behaviour.** Nothing says what to do while an escalation is pending.

All of these let the agent loop, or drift silently for weeks.

**Fix.** Add to 11 §3:

> **Time-box.** If a bucket isn't fixed after 3 attempts or one working day:
> 1. write `docs/blocked/<signature>.md` (hypotheses tried, evidence, probes);
> 2. mark it `blocked` in PROGRESS.md;
> 3. take the next bucket;
> 4. revisit blocked buckets weekly.
>
> **Re-plan trigger.** If CBR improves by less than 0.1 pp over 5 consecutive T2 runs, stop content work and write a re-plan note (R1).

Add to 11 §5:

> **Gate checkpoint.** When a gate passes, send the gate report to the human. Until it is acknowledged, continue only infrastructure lanes. While any escalation is pending, work other lanes; never idle.

Add to 11 §7 a `PROGRESS.md` template with these headings: Milestone/Task · Last T2/T3 run (`S`, date, CBR/TMR/MPR/DPR per source) · Top 5 buckets · Blocked · Pending escalations · Contract hashes · Next 3 tasks.

### 21. MAJOR · `10` M1.5, `02` §4.2 `damage` RPC — the "no hooks" damage core can't be validated against PS goldens on random sets

**Problem.**
- **Hooks fire in PS.** M1.5's goldens use "random attacker/defender/move/field". In PS, abilities, items and screens all act through hooks (screens are side conditions with `onAnyModifyDamage`), but the M1 engine has no hook framework.
- **Contradictory parameters.** "Screens as explicit parameters" in the engine can't match PS's screen semantics before M3.

**Fix.** Change M1.5 to:

> **Damage core.** No hooks. Explicit parameters for spread, crit, roll and burn. The ModifyDamage chain is the identity until M3.1.
>
> **Goldens:**
> - port the 4 verified examples from `research/ps-move/damage-examples.js` (all 16 rolls each);
> - plus 100k `damage` RPC cases restricted to the core subset (item 2): no items, core abilities, no side conditions, no weather or terrain;
> - screens, weather, items and abilities join the golden set at M3.3/M4, when their hooks exist.

### 22. MAJOR · `07` §12.2 T0/T1, `09` §3 `unit` profile — the test suite's compile cost blows the T0/T1 budgets

**Problem.**
- **One compile per scenario file.** The `unit` profile compiles each scenario's handler subset (`cfg.handlers`) separately: up to 60 s each (09 §1 dev budget).
- **Budgets can't hold.** T0 (< 2 min) runs "the affected scenario files". T1 (< 20 min) runs *all* scenario files, which will number in the hundreds by v1.
- **The cache doesn't help.** The persistent cache is invalidated by any edit to traced code.

**Fix.** Add to 09 §3:

> Scenario tests do not compile per test. The runner collects all scenarios in a session, builds **one** `conformance`-profile executable per stage, and replays them as a single vmapped batch (scenarios are padded to a common decision count). The `unit` profile is only for `jax.disable_jit()` debugging.
>
> T0 = lint + `tests/unit` + scenarios run under `disable_jit` for touched families only (≤ 20 scenarios). T1 = one compile per stage plus the batched scenarios and fixtures.

---

## MINOR

### 23. MINOR · `01` ADR-024, `02` §2, `10` M0.6 — exact pins aren't chosen

**Problem.** ADR-024 says "pin JAX exactly" but leaves the version open. Research used jax/jaxlib 0.12.0.1, flax 0.12.10, chex 0.1.92, Python 3.12 and Node v26.8.1. `uv` itself is unpinned, and there is no plan to pair a CUDA jaxlib with the CPU pin.

**Fix.** Add to M0.6:

> Pin `jax==jaxlib==0.12.0.1`, `flax==0.12.10`, `chex==0.1.92` and Python 3.12 (`.python-version`). Add a `gpu` extra pinning `jax[cuda12]==0.12.0.1`. Pin Node 26.8.1 in `.nvmrc` and `uv` in `README`. Bump these only via ADR, with a T3 run.

### 24. MINOR · `05` §3, `06` §1, `11` §7, `CLAUDE.md` — conflicting rules for handler stubs, handler signatures and generated file names

**Problem.**
- **Stub location.** 05 §3 says codegen emits stubs "that the implementer fills in", but they land in `vgcjax/data/handlers_stub.py`, and CLAUDE.md forbids editing `vgcjax/data/*`.
- **Handler signature.** 05 §3 uses `(ctx, state)`, while 11 §7 uses `(ctx, state, dex)`.
- **Rows file names.** The rows file is `hook_rows.npz` (06), `hook_rows_*.npz` (05) or `hooks_table.py` (02).

**Fix.**
- Stubs are written to `vgcjax/data/handlers_stub.py` as a reference only. `tools/datagen/scaffold.py <entity>` copies a stub into the right `vgcjax/effects/*.py` module.
- The signature is `(ctx, state, dex) -> (relay, ctx, state)`.
- The only rows file is `vgcjax/data/hook_rows.npz`. Fix 02 §2 and 05 §3 to match.

### 25. MINOR · `02` §4.3, `08` §1, `07` §5.2 — API and module names diverge

**Problem.**
- 02 uses `init_from_trace`; 08 uses `init_from_order`.
- 02 names `vgcjax.harness_api.project(state)`, which isn't in the 02 §2 layout; 07 uses `harness/project.py engine_to_json`.
- RL uses `legal`; 08 uses `legal_masks`.
- "Trace parsing never in `vgcjax/`" sits awkwardly next to `vgcjax.battle.state.from_ps_snapshot` parsing PS JSON.

**Fix.** Use `init_from_order`, `legal_masks` and `harness/project.py`, and delete `harness_api`. Add to 02 §4.3: "`from_ps_snapshot` is the single exception: PS *state* JSON may be parsed in `vgcjax/battle/snapshot.py`. Trace NDJSON is parsed only in `harness/`."

### 26. MINOR · `03`, `01`, `00`, `04` §14, `10` "Total" — numeric inconsistencies

**Problem.**
- **Leaf budget:** ≤ 100 (ADR-013) vs ≤ 150 (03).
- **Types:** `TypeId 0..18 + 19 = ???` (19 or 20 slots for 18 types) vs "19×19" (M1.4) vs `typechart [20,20]` (03 §2).
- **v0 size:** the README's "v0 (220 custom entities)" vs the catalogue's v0 CUSTOM count of 154 non-condition entities.
- **Champions deltas:** 04 §14 requires scenario tests for "CH §0 rows 1–20 and CH §12 P0/P1", but the catalogue has 16 ENGINE_DELTA rows. Rows 8, 10, 11 and 12 are entity-level, and P1 rows 13–16 aren't mapped.
- **Totals:** single-agent serial totals are M0–M4 = 13–15 EW, plus M5 = 15–18 EW, not "11–15 EW".

**Fix.**
- Leaf budget ≤ 150 (update ADR-013).
- `TypeId`: 0 = none, 1..18 = Normal..Fairy, 19 = `???`; `typechart [20,20]`; M1.4 reads "20×20 + pseudo".
- README: "v0: 220 CUSTOM entities including conditions".
- Add the missing ENGINE_DELTA rows to `make_catalogue.py`, or map them by id in a `ch_delta` column.
- Correct the totals, and state "with one agent, lanes are serial".

### 27. MINOR · `02` §4.1, `07` §2, `11` §3 — CLI flags and policies that are used but never specified or assigned

**Problem.**
- `--require <id>`, `--legal on`, `--stage core` and the `oracle legal` subcommand are used in 05, 07 and 11, but are missing from 02 §4.1.
- CLAUDE.md passes `--out traces/smoke` (a directory), while 02 uses a shard pattern.
- The `random-weighted`, `heuristic` and `engine-policy` policies aren't in the POC and have no task.

**Fix.** Add these flags to 02 §4.1, and state that `--out` is a directory with shards named `shard-%05d.ndjson.zst`. Assign `random-weighted` and `heuristic` to M3.8b (item 12), and `engine-policy` to M5.6.

### 28. MINOR · `07` §10.2 — the shrinker's "inert" abilities are wrong

**Problem.** Ball Fetch isn't legal in Reg M-C. Run Away is a [C] CUSTOM handler (trap immunity), so it isn't inert whenever trapping is present.

**Fix.** "Neutral ability: use one from `core_inert.json` (item 2) that the mon can legally have. If none exists, keep the original ability."

### 29. MINOR · `10` M0.3, `02` §2 — "POC scripts 01–08 run unchanged" isn't literally possible

**Problem.**
- `ps.js` defaults to `/tmp/psrepo/ps`, and `teams.js` defaults to `/tmp/stats_dl/...`; both need env vars.
- `classify.js` loads `path.join(__dirname,'..','rng_callsites.json')`, but 02 places the file at `oracle/data/`.
- `callsite.js` requires `${PS_ROOT}/node_modules/source-map`, but 02 says `oracle/package.json` provides `source-map`.

**Fix.** "M0.3 Done when `PS_ROOT=oracle/ps CHAOS_STATS=data/raw/smogon-chaos/2026-09/gen9championsvgc2026regmcbo3-1760.json node docs/research/poc/scripts/0N_*.js` reproduces the `results/0N_*.json` verdicts. In `oracle/lib`, change only the default paths: the inventory goes in `oracle/data/`, and `source-map` comes from `oracle/node_modules`."

### 30. MINOR · `10` M0.7, `06` §4.1 — M0.7 points at the wrong sampler

**Problem.** `research/poc/lib/teams.js` has no stage filter, no Teammates chain and no Mega-base-ability handling. The stage-aware sampler with the Teammates chain is `research/dex/scripts/team_sources.js`, which reported 2,000 of 2,000 valid within v1.

**Fix.** "M0.7: port the pool B sampler from `research/dex/scripts/team_sources.js`, parameterised by stage list; use `research/poc/lib/teams.js` only for `HarnessRNG` and pool C."

### 31. MINOR · `07` §4.4, `04` §7.3 — rules for SECONDARY draws aren't stated

**Problem.** PS draws `random(100)` for every secondary, even when `chance` is undefined (`rng_callsites.json`, `sim/battle-actions.ts:1399`, flagged "don't-care" there). Those draws aren't on 07's don't-care or degenerate lists. Chance-100 secondaries (Icy Wind) are also neither degenerate nor don't-care.

**Fix.** Add to 07 §4.4: "SECONDARY with `chance` undefined is don't-care (both sides). SECONDARY with chance ≥ 100 is drawn by the engine and compared, because PS draws it." Add the same text to 04 §7.3.

### 32. MINOR · `07` §3, `02` §2 — no storage budget, retention policy or parquet dependency

**Problem.**
- **Volume.** At 35 KB per battle uncompressed, T2 is about 7 GB of raw JSON a night and T3 about 35 GB a week.
- **Missing dependency.** `results.parquet` needs `pyarrow`, which isn't in the pyproject list.

**Fix.**
- Keep the last 7 nightly runs and every gate run; prune the rest.
- Write only failing battles' full traces to `reports/`; passing shards are deleted after the run unless `--keep`.
- Budget 200 GB of disk.
- Add `pyarrow` to `pyproject.toml`.

### 33. MINOR · `06` §5, `01` ADR-025, `11` §2.9 — supply chain, licences and personal data

**Problem.**
- `npm ci` runs install scripts (esbuild fetches a binary).
- The VGC-Bench HF licence hasn't been checked.
- Replay team files contain usernames.

**Fix.** Add to ADR-025:

> - Run `npm ci` only against the committed lockfile and record `npm ls --all` per pin.
> - Check the licence of each dataset before committing anything derived from it, and record it in `SOURCES.md`.
> - Strip usernames and ratings from committed team banks; keep only the rating bucket.
> - Never commit `champions/web` pages.

### 34. MINOR · `08` §1–3, §6 — the `Masks`/`Obs` pytrees and `has_decision` are underspecified

**Problem.**
- **Phase-dependent shapes.** The preview mask is `[360]` (or 90), while move and switch masks are `[2,29]` plus `pair_ok [29,29]`. Under vmap, `Masks` must be a single fixed pytree.
- **`has_decision` gap.** It isn't defined for a player whose only joint action is a non-pass forced action (a locked move).
- **Missing plumbing.** `opp_policy_id` has no path into `EnvState`; the `reset(key, sampler)` signature has no slot for it.
- **Undefined size.** `shaping`'s width `F` is undefined.

**Fix.**
- `Masks = {preview: bool[360], slot: bool[2,29], pair_ok: bool[29,29]}`, with all three always present and only the phase's fields meaningful.
- `has_decision[p] = (#legal joint actions for p) > 1`.
- `reset(key, sampler, opp_policy_id: i32[B])`, stored in `EnvState`.
- `F = 4` (HP fraction sum, mons remaining, fainted, mega used).

### 35. MINOR · `08` §7 — `sample_hidden` and `determinize` omit hidden counters and the belief helper

**Problem.** 08 §4 lists sleep and confusion turns remaining as hidden information, yet `sample_hidden` samples only SP spreads, the brought set and bench order. `determinize` can't build a consistent state without them. RL §7.9's `public_outcome_match` helper, needed for belief updates by re-simulation, was dropped.

**Fix.** "`HiddenSample` also includes the remaining counters for sleep (`time`), confusion, partial trap, lockedmove and Encore/Taunt/Disable, sampled from PS's distributions conditioned on the turns elapsed. Add `public_outcome_match(state', observed_events) -> bool` (MAY)."

### 36. MINOR · `07` §5.3, `03` §4.3 — `vol_state` column semantics are fixed in one place and per-volatile in another

**Problem.** 03 fixes `vol_state[NV,6]` columns as "duration, counter/time, seq, source, move/arg, extra". It also says "per-volatile field mapping generated (07 §5.3)".

**Fix.** "Columns 0 = duration, 2 = seq and 3 = source are fixed. Columns 1, 4 and 5 are assigned per condition by `gen-projection-map`, in first-seen key order, and frozen in `projection_map.json`. New keys append."

### 37. MINOR · `10` M2.7 — the round-trip test needs a scoped snapshot source

**Problem.** "10k snapshots" from full-pool battles contain conditions the M2 engine can't represent.

**Fix.** "M2.7: 10k snapshots from core-stage traces (3 random decision points per battle, including mid-turn SWITCH pauses from core battles with pivot-free teams). The test re-runs per stage at each gate."

### 38. MINOR · `07` §12.2 T2, `05` §2, `CLAUDE.md` — nightly automation edits `docs/catalogue.csv` against the "same commit as the entity" rule

**Problem.** T2's "catalogue update" (`fuzz_hits`, auto-`done`) creates commits outside any agent session, which conflicts with CLAUDE.md's "update it in the same commit as the entity".

**Fix.** "T2 writes `reports/catalogue_status.json` only. The agent folds it into `docs/catalogue.csv` at session start, in a `data(catalogue): fuzz hits <run>` commit."

### 39. MINOR · `08` §3 `preview_canonical` — the canonical 90 removes a policy-relevant choice

**Problem.** The canonical 90 fixes bench order ascending, but Illusion disguises as the *last* party member. Bench order is therefore a real choice for Zoroark teams from v1 onward, and the canonical wrapper removes it.

**Fix.** "`preview_canonical` canonicalises bench order only when no brought mon has Illusion. Otherwise it enables both bench orders: up to 180 entries, masked in the 360 table."

---

## Answers to the brief's eight questions

| # | Question | Short answer and items |
|---|---|---|
| 1 | Can M0/M1 start with no questions? | M0 yes, after fixing M0.1 (1). It still hides decisions on data sources, stage lists and pins (5, 6, 23) and needs eight unassigned RPCs (3). M1 is blocked on RPCs and the core-subset restriction (3, 21). |
| 2 | Are the gates measurable? | Not as written. The G-core subset is undefined and infeasible (2). Metric definitions after the first failure (11), corpus composition (12) and the coverage, done and waiver rules (13) are missing. The M5.5 gate is undefined (18). |
| 3 | Circular dependencies? | Yes: scenario tests vs `craft` (4); M0.5 self-tests vs M1.1 and M3.2 (4); M3.8/M3.9 after the content that needs them (4); M5.5 vs G-v0 (18); the projection map vs the engine layout (4). |
| 4 | Is the RL API sufficient? | Close. Missing `final_obs` and coherent truncation (17), a decision on auto-resolve (16), the `Masks` pytree and `has_decision` definition (34), and hidden counters for determinisation (35). Opponent pools and episode stats are adequate. |
| 5 | A perf plan without a GPU? | No. Add CPU proxy budgets and escalate early (19). |
| 6 | Process risks? | Self-graded gates (14), no time-boxing or checkpoints (20), CI and scheduling undefined (10), test compile cost (22). CLAUDE.md and 11 are otherwise consistent with 01 and 10. |
| 7 | Cross-document inconsistencies? | The DrawKey contract (15), names and paths (24, 25, 27), numbers (6, 26). |
| 8 | Missing bases? | Seeding (9), trace and fixture versioning (8), regulation and re-pin policy (7), storage (32), supply chain and personal data (33), the data manifest (5). |
