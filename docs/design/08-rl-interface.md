# 08 — RL-facing interface (env API, observations, actions, resets, search and deployment hooks)

Source research: `research/rl-training-design.md` (**RL**). This document fixes the API. The trainer itself (PPO, league, model) is out of scope, but the engine must support it without redesign. Requirements are tagged **MUST**, **SHOULD** or **MAY** following RL §7.12.

## 1. Core API (MUST) — `vgcjax/env/api.py`

```python
# static
@dataclass(frozen=True)
class EngineConfig:
    stage: Literal["v0","v1","v2"]
    handlers: frozenset[str] | None = None   # None = all handlers of the stage; subset for fast dev builds
    max_turns: int = 40                       # 1000 in conformance mode
    turn_cap_outcome: Literal["tiebreak","draw","truncate"] = "tiebreak"
    auto_resolve_forced: bool = False         # see §2; not implemented unless an ADR justifies it
    preview_canonical: bool = True            # policy sees canonical preview options (§3)
    forced: bool = False                      # compile forced-outcome lookup (07)
    draw_log: bool = False
    events_capacity: int = 32
    obs_history: int = 16
    flags: Flags = Flags()                    # ADR-019 disputed-mechanics flags (PS defaults)
    debug_checks: bool = False                # checkify invariants

def init(key, team: Team, cfg, dex) -> State                         # phase = PREVIEW
def init_from_order(key, team, order: i32[2,4], cfg, dex) -> State   # harness/replay: skip preview
def step(state, actions: i32[2], key, cfg, dex,
         forced: ForcedOutcomes | None = None) -> tuple[State, StepOut]
def observe(state, player: i32, cfg, dex) -> Obs                     # info-state only, canonical frame
def legal_masks(state, player: i32, cfg, dex) -> Masks
def observe_privileged(state, player, cfg, dex) -> Obs               # SHOULD: hidden fields filled + known masks
def hidden_targets(state, player, dex) -> HiddenTargets               # SHOULD: labels for auxiliary heads
```

**Rules:**
- All functions are pure and single-battle. Vmap them outside.
- `jax.eval_shape(init, …)` gives the shapes without running.
- `step` on a `DONE` state is a no-op: all-true masks and zero reward.

## 2. Phases and "both act every step" (MUST)

- **Phases:** `PREVIEW`, `MOVE`, `SWITCH`, `DONE`.
- **Every step takes `actions[2]`.** A player without a decision has a pass-only mask (`has_decision[p] = False`), and its action is ignored.
- **`has_decision`.** `has_decision[p] = (number of legal joint actions for p) > 1`. A forced locked move or a single forced replacement counts as no decision.
- **Auto-resolve is not part of `step`.** Resolving internally would mean running another decision cycle inside `step`, and under vmap every battle pays for it. Instead:
  - trainers drop samples where `has_decision == [False, False]` from all losses;
  - samples with exactly one `has_decision` False are dropped from that player's policy and entropy losses, but kept for the value loss.
  - Implement auto-resolve only after a benchmark shows such steps exceed 5% of rollout steps, by ADR. Any implementation must pass `tests/property/test_autoresolve.py`: the trajectory equals the parity-mode trajectory with the single legal actions inserted.

## 3. Actions and masks (MUST; ADR-018)

| Phase | Engine action | Mask shape | Notes |
|---|---|---|---|
| PREVIEW | `0..359`: `PREVIEW_ORDERS[i] = (s0,s1,s2,s3)` | `[360]` | With `preview_canonical`, only canonical entries are enabled: leads in ascending sheet order, bench in ascending sheet order (90 entries). **Exception:** if a brought mon has Illusion, both bench orders are enabled (up to 180), because Illusion disguises as the last party member. `env/actions.py` exports the index maps. |
| MOVE | joint `j = a0*29 + a1` | per slot `slot_mask[2,29]`, `pair_ok[29,29]`, `joint_mask[841]` (materialised on request) | Per slot, encoding per 04 §4.3: `a = 6·m + 2·t + mega` for a < 24; `24–27` switch to brought b; `28` pass. Locked / Struggle use `a = 0` |
| SWITCH | same encoding | same | Only `24–27` and `28`. Revival Blessing slots: `24–27` select fainted b |

- **The `Masks` pytree has a fixed shape** whatever the phase: `Masks = {preview: bool[360], slot: bool[2,29], pair_ok: bool[29,29]}`. All three fields are always present; only the current phase's fields are meaningful. `joint_mask(m) -> bool[841]` is a helper.
- **Illegal actions** are replaced with the first legal action, and `ERR_ILLEGAL_ACTION` is set (ADR-014).
- **Interop (MUST)** — `vgcjax/interop/ps_choice.py`:
  - `to_ps_choice(state, player, action) -> str`, for example `"move 1 +2 mega, switch 3"` or `"team 3125"`. It maps brought index → PS `switch N` through `ps_order`, and target t → PS signed location.
  - `parse_ps_choice(state, player, str) -> action`, the exact inverse on canonical strings, used by the harness.
  - `ps_request_to_masks(request_json)`, used for deployment and parity.

## 4. Observation (MUST) — `vgcjax/env/obs.py`

Canonical frame: index 0 is self, 1 is opponent. Integer-coded, with embeddings done in the model. Target size is about 4 KB per player.

| Group | Content |
|---|---|
| `meta` | phase, request kind, which own slots must act, turn, turns to cap, `mega_available` |
| `field` | weather / terrain id + turns elapsed + turns remaining; pseudo-weathers (Trick Room, Gravity, …) + remaining turns |
| `side[2]` | side conditions + remaining turns / layers, this-turn guards, `mega_used`, mons remaining, fainted count |
| `sheet[2,6]` | species, item, ability, nature, gender, moves[4], `mega_stone`, all public under OTS; plus `brought ∈ {unknown, yes, no}` and `revealed` |
| `own[6]` (self only) | exact stats[5], SP[6], exact HP / max HP, PP[4], `brought_order` |
| `mon_public[2,4 brought]` | HP as **PS Champions rule** `floor(100·hp/maxhp) or 1` plus colour bits (20/50) for the opponent; exact HP for self. Plus status + public counters (sleep turns elapsed, not remaining), boosts[7], public volatile bits, active slot, fainted, current species (Mega), item state (consumed / knocked off), ability state (revealed changes), last move, choice-lock move, Protect streak, turns active, observed PP use |
| `events[16]` | last 16 public event records (03 §4.7), using percentage HP |
| `masks` | the masks from §3, for the policy |

**Hidden information (never readable by `observe` for the opponent):**
- SP spread and exact stats; exact HP (only the percentage rule);
- unrevealed brought mons and bench order;
- sleep and confusion remaining counters;
- RNG;
- the opponent's chosen action.

**Damage calculations are not in the observation (ADR-022).** Provide `damage_calc(att: MonView, dfn: MonView, move, field: FieldView, hyp: SpreadHyp) -> DmgDist` (SHOULD) as a pure jittable function over views and hypotheses, sharing `battle/damage.py` internals. Also provide the convenience `calc_features(info, prior)`, which uses own exact stats and the opponent's prior spreads.

**Leak tests (MUST, `tests/property/test_obs_leak.py`):**
1. Perturbing hidden fields leaves `observe(state, 0)` bitwise identical. Perturbations: re-sample the opponent's SP within its candidates; swap the unrevealed bench; change exact HP within the same percentage and colour bucket; change the RNG key. The same perturbations on side 0 must change the observation where public.
2. **Perspective symmetry:** `observe(state, 1) == observe(swap_sides(state), 0)`.
3. **Protocol round trip** (M8):
   1. play an engine game;
   2. export it through PS re-simulation;
   3. reduce the PS protocol with `interop/protocol_reducer.py`;
   4. the result equals `infostate(state, p)` at every decision.

## 5. Reward, termination, truncation, shaping (MUST)

- `reward: f32[2]` is zero-sum and terminal only: +1 win, −1 loss, 0 draw. A `reward_01` flag selects 1/0.
- `terminated` is set on a KO win or loss, or a PS tie.
- **Turn cap.** `turn_cap_outcome` selects what happens at `max_turns`:
  - **`tiebreak`** (default): `terminated=True`, `truncated=False`, and the reward is the tiebreak result (remaining mons, then HP-fraction sum, then draw);
  - **`draw`**: `terminated=True`, reward 0;
  - **`truncate`**: `terminated=False`, `truncated=True`, reward 0. The trainer bootstraps from `final_obs`.
- **`StepOut.final_obs: Obs[2]`** equals `obs`, except on steps where `AutoResetEnv` reset the battle, where it holds the pre-reset observation.
- **`shaping: f32[2, 4]`** holds the HP-fraction sum, mons remaining, mons fainted and Mega used. These are potential features only; they are **not** added to the reward.

## 6. Auto-reset, team bank, episode stats (MUST) — `vgcjax/env/autoreset.py`, `teambank.py`, `stats.py`

```python
class TeamBank(PyTree):    # device-resident, validated offline (06 §4)
    species, item, ability, nature, gender: i16[N,6]; moves: i16[N,6,4]
    sp_cand: i8[N,6,K,6]; sp_logit: f32[N,6,K]; valid: bool[N]
def sample_team_pair(key, bank, pair_logits: f32[N*N] | None, opts) -> tuple[Team, i32[2]]
#   opts: no_mirror, skip_preview (random preview), mirror_only (debug)
class AutoResetEnv:
    reset(key, sampler, opp_policy_id: i32) -> (EnvState, StepOut)    # opp_policy_id stored in EnvState
    step(env_state, actions, key) -> (EnvState, StepOut, EpisodeStats)   # select-based reset when done
class EpisodeStats(NamedTuple):   # valid where done
    winner: i8; turns: i16; steps: i16; end_reason: i8
    team_ids: i32[2]; opp_policy_id: i32; preview_actions: i32[2]
    megas: i8[2]; remaining: i8[2]; hp_frac: f32[2]; error_flags: u32
```

**Reset cost.** Under vmap, a select-based auto-reset computes `init` every step. `init` must therefore be cheap:
- gather the team from the bank;
- compute stats;
- no hook calls until the PREVIEW step.

**Matchup tables.** `[N,N]` win and count tables are updated by scatter-add from EpisodeStats, inside jit, by the trainer.

**Seat randomisation** is the trainer's job. The engine must be side-symmetric. This is tested: swapping sides plus mirroring the forced outcomes gives a mirrored trajectory.

## 7. Search support (SHOULD, milestone M8)

| API | Purpose |
|---|---|
| clone | free (immutable pytrees) |
| `infostate(state, player) -> InfoState` | the player's knowledge; the same pytree the protocol reducer builds |
| `sample_hidden(key, info, belief=None) -> HiddenSample` | SP spreads from candidates (or a belief), the brought set, bench order, **and hidden counters**: sleep `time`, confusion, partial trap, lockedmove, Encore/Taunt/Disable remaining. Counters are sampled from PS's distributions conditioned on the public turns elapsed |
| `public_outcome_match(state', observed_events) -> bool` (MAY) | belief update by re-simulation (RL §7.9) |
| `determinize(info, hidden, cfg, dex) -> State` | a full state consistent with the info state. Round-trip property: `infostate(determinize(infostate(s), h)) == infostate(s)` |
| `hash_infostate(info, hp_buckets=10, world_id=None) -> u32[2]` | transposition and statistics keys |
| `apply_preview(state, a0, a1, key) -> State` | vmappable over 90×90 for preview matrix solves |
| `forced` outcomes | exact-expectation and belief-likelihood computations |
| `enumerate_outcomes(state, actions, max_branches)` (MAY) | KO-relevant roll buckets × hit/crit outcomes |

## 8. Deployment bridge (SHOULD, M8) — `vgcjax/interop/protocol_reducer.py`

- **Input:** a PS protocol stream from the player's perspective, plus requests. **Output:** an `InfoState`. Then `observe(determinize(info, prior_sample), 0)` gives the policy input.
- **Sources:** own side from request JSON; opponent from `|showteam|`, `|switch|`, `|-damage|` (percentage plus colour), `|-boost|`, `|-status|`, field messages and so on.
- **Mask parity:** on every live decision, compare engine masks with PS's request. On mismatch, log it and fall back to the PS legal set.
- **Client:** poke-env 0.16+ (`DoubleBattle`) is a thin transport only. Do not map poke-env's battle object field by field (RL §9.1).
- **Policy (documentation):**
  - register bots per PS policy (2026-10-09);
  - use the opt-in bot pool or a local server only;
  - never use it in tournaments or for live assistance in the Champions game, which the Pokémon Terms of Use prohibit;
  - the intended "Champions use" is offline analysis, practice and team evaluation.

## 9. Replay export and debugging (MUST)

- `cfg.draw_log=True` records the chance trace. `interop/trace_export.py` builds a PS input log from (teams with SP, actions, chance trace), runs it through the oracle with forced outcomes, and returns the genuine PS protocol log and HTML replay.
- **Any engine/PS divergence during export is a conformance failure.** There is no separate engine-native protocol emitter, so there is no second source of truth.
- Deterministic replay: (teams, initial key, actions) fully determines a game.

## 10. Opponent pools and populations (support only)

- The engine is agnostic to who acts. Requirements:
  - `opp_policy_id` passes through to EpisodeStats;
  - observations are canonical per player, so any policy can drive either seat.
- Batch partitioning by opponent group, and NeuPL-style policy-index conditioning, are trainer concerns.

## 11. Throughput contract (MUST; details in 09)

| Target | Value |
|---|---|
| Standalone engine, 1 H100, batch 16k–64k | ≥ 3×10⁴ env-steps/s minimum, **10⁵ target** |
| Batch 1k–4k (for search) | ≥ 2×10⁴ env-steps/s |
| Compile time, full stage | < 2 min |
| Memory at batch 64k | < 15 GB |

Measured game length: about 8 turns and about 12.5 env steps per game (RL §8.1).
