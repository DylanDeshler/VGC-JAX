# 12 — Risks, mitigations, and resolved/open questions

## 1. Risk register

| # | Risk | Likelihood / impact | Mitigation | Owner / trigger |
|---|---|---|---|---|
| R1 | **The mechanics long tail is slower than estimated.** vgc-engine was only 16.5% clean after significant effort. | High / High | Harness-first (ADR-016). Shrinking and bucketing turn the tail into a queue. Staged dex (v0 trainable early). Catalogue tracking. | Watch the weekly CBR trend. If v0 bucket burn-down stalls for 2 weeks, re-plan scope (e.g. narrower v0 team pool) |
| R2 | **The oracle moves.** About 16 mechanics commits in 3 months; regulation change expected 2026-12-02, **before** G-v0. | Certain / Medium | Target freeze until G-v0 (ADR-028); after that, monthly re-pins (ADR-021). Call-site drift alarm, generated data, catalogue diff, fixture refresh (07 §10.3) | Every re-pin; new regulation announcement |
| R3 | **PS differs from the real Champions game.** Example: Fire Mane is disputed. | Medium / Medium (policies evaluated on PS anyway) | Flags (ADR-019), PS-issue watch list in `docs/oracle-changelog.md`, re-pin picks up fixes | On new PS issues touching `champions` |
| R4 | **All-branches cost plus handler count make throughput or compile time miss targets.** | Medium / High | Tables-first dispatch, per-hook switches with barriers, Block A/B split, config-gated dev builds, persistent compile cache. Budgets enforced in CI. M9 perf lane. | Compile > 120 s, or GPU throughput < 3×10⁴ steps/s at G-v1 |
| R5 | **Forcing-contract gaps.** Some draws can't be keyed identically on both sides (e.g. new PS call sites, nested event draws). | Medium / Medium | `DRAW_MISMATCH` localises them; key-contract fixes go to both sides; weekly don't-care re-validation; stack classifier cross-check | `UNMAPPED` draw or classifier disagreement |
| R6 | **Hidden ordering semantics** (effectOrder, handler ties, cached speed) cause rare divergences. | Medium / Low–Medium | Mirror `effectOrder` (`seq`) and `speed_cached` update points exactly (ADR-007); crafted tie templates | Buckets with `STATE` diffs only under ties |
| R7 | **Memory.** State about 10 KB/battle (above the scope's 2 KB guess) plus rollout storage. | Medium / Medium | Compact `vol_state` via codegen offsets; store `slot_mask` + `pair_ok` instead of joint masks; recompute observations in the update | > 15 GB at B = 64k |
| R8 | **Model-free RL fails to learn VGC at scale** (VGC-Bench saw degradation; PokaiTrainer says search carries strength). | Unknown / High (for the user's goal, not the engine) | The engine API supports both regimes (search hooks in M8). Prior evidence is sample-starved (5M steps vs 10⁹) | Smoke-trainer results (M5.5) |
| R9 | **Team realism:** usage-sampled teams lose within-set correlations, and real sheets are low-ladder. | Medium / Medium | Mix pools A and B; continuous scraping with rating filters; held-out real sheets for evaluation | Pool statistics in the team-bank manifest |
| R10 | **Bot and ToS policy:** PS registry rules, Pokémon ToU. | Certain / Medium | Local server and opt-in bot pool only; no in-game automation; offline-analysis framing (08 §8) | Before any online play |
| R11 | **Environment fragility:** `/tmp` was wiped twice during research. | Happened / Low | Everything lives in the repo or durable data dirs; PS in a submodule | — |
| R12 | **Agent drift:** an implementer deviates from PS semantics "for simplicity", or weakens the contract to pass gates. | Medium / High | Binding ADRs, a PS citation on every function, no hand-written expectations, fixtures block merges, protected contract with hash check, canary test (ADR-029), time-boxing and gate checkpoints (11 §3, §5) | Lint for missing `PS:` tags; contract-hash mismatch |

## 2. Questions resolved by this design (so the implementer doesn't re-litigate them)

| Question | Resolution |
|---|---|
| SV or Champions? | Champions Reg M-C (ADR-001) |
| Tera? | No. Champions disables it; Mega Evolution replaces it |
| Which oracle? | PS pinned (ADR-002); real-game disputes are flags |
| Dex scope? | Staged v0 → v1 → v2; v2 = full legal pool (ADR-003) |
| Match PS's RNG stream? | No. Semantic keyed forcing (ADR-012, `contract/drawkey.md`) |
| Engine shape? | Pure single-battle functions, vmapped (ADR-005) |
| Step granularity? | One simultaneous decision point (ADR-008) |
| Per-slot actions? | 29 (move × target × mega, switch to brought b, pass); joint 841 (ADR-018) |
| Preview actions? | Engine 360 (exact PS); policy wrapper 90 canonical (08 §3) |
| Volatile storage? | Per mon, dense per kind, bitfield presence (03) |
| Effect dispatch? | Rows + per-hook switch + barriers (ADR-010) |
| Durations? | PS semantics verbatim (ADR-007, 04 §9.3) |
| Reward? | Terminal zero-sum; shaping features only (ADR-023) |
| Turn cap? | 40 in training with tiebreak; 1000 in conformance |
| Opponent HP in obs? | PS Champions floor-percent rule plus colour bits (ADR-022) |
| Damage calcs in obs? | No. Model-callable on hypotheses (ADR-022) |
| Bo1 vs Bo3? | Engine plays single games. Deploy and evaluate on Bo3 (forced OTS) |
| Language for the oracle? | Node JS (reuse the POC); engine Python/JAX (ADR-024) |
| Which regulation if Reg M-C ends mid-project? | Stay on Reg M-C until G-v0, then switch as a mini-milestone (ADR-028) |
| Auto-resolve forced decisions? | Not in `step`; the trainer masks the samples (08 §2) |
| Turn cap semantics? | `tiebreak` (terminated) by default; `truncate` available with `final_obs` (08 §5) |

## 3. Open questions (the implementer decides, records an ADR, and proceeds)

1. **Exact target-slot legality when a target slot is empty at choice time** (04 §4.3). Determine it from PS acceptance enumeration in M2.6.
2. **Residual tie grouping comparator.** Is `effectOrder` part of PS's fieldEvent tie comparator (MV §1.8)? Determine it in M3.3 with a probe. It affects whether `SPEED_TIE_FIELD_EVENT` groups ever have more than one member.
3. **Whether `speed_cached` must also refresh at other PS call sites** in edge cases (Instruct `insertChoice`, Mega, end-of-turn). Verify each against TL §5.7 with probes.
4. **Compact vs dense `vol_state`.** Decide at M9 by measurement.
5. **Whether the 3-target axis should be unrolled in the GPU release build.** Measure at M9 (`research/jax-patterns.md` §1C).
6. **Transform PP under Champions PP rules** (5 PP each in base; check the Champions override). Probe in M6.1.
7. **Hard-to-reach effects** whose PS behaviour depends on Gen-specific code paths not exercised by Champions (e.g. `Pursuit`-like leftovers in `sim/`). Treat them as unreachable unless coverage finds them.
8. **Closed team sheet support** for Bo1 play (85% of Bo1 games). Out of scope (ADR-004), but if the user later wants it, the reveal bits and `protocol_reducer` are the extension points.
