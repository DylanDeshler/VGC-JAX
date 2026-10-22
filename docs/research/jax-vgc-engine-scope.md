# Scoping: a pure-JAX VGC (doubles) battle engine for policy training

*Research date: 2026-10-21. "Verified" means I checked the primary source myself. Everything else is from research-agent reports and should be checked before you rely on it.*

---

> **Update 2026-10-21:** Target chosen: **Champions VGC Reg M (doubles)**. Champions has **no Terastallization**; it uses **Mega Evolution**. Tera references below have been changed to Mega. See §6a for the automated validation design.

## 0. TL;DR

- **Precedent.** Jaxcalibur showed the recipe works for Gen 9 random singles:
  - Engine: pure-JAX, every branch computed and then masked, `jit`/`vmap`.
  - Model: an 8.5M-parameter transformer trained with PPO + GAE self-play, no search during training.
  - Scale: about 100M games in about 1 week on 1 H100. The author spent 6+ months on and off.
  - Search: pUCT is used **only at test time** and adds about 100–150 Elo.
  - Code is **not released** and won't be.
- **No JAX doubles/VGC engine exists publicly.** The nearest projects are:
  - `colefoster/vgc-engine`: Rust, Gen 9 doubles plus Champions, pre-alpha.
  - `spktrm/minimon`: a JAX singles toy with procedurally generated mechanics, useful as an API template.
- **The hard part is mechanics fidelity, not RNG or JAX.** vgc-engine's own audit (verified) found:
  - Even with every random outcome forced to match Showdown, only **16.5%** of real Reg M-C battles match PS all the way through, and **78.7%** of turns match.
  - About **94%** of divergences are mechanics bugs.
  - Budget most of the effort for a differential-test harness against Showdown.
- **Pick the format first. The landscape has moved.** Official VGC now runs on **Pokémon Champions**:
  - Reg M-C has about 1.63M ladder battles per month, against about 9.9k for SV Reg G (Smogon stats 2026-09, verified).
  - Champions changes some mechanics (e.g. full paralysis is 1/8, not 1/4) and uses Stat Points instead of EVs.
- **The long tail is bounded.** At ≥1760 rating in Champions Reg M-C Bo3, 99% of usage is covered by **92 species, 158 moves, 65 abilities and 60 items**. That set is a realistic v1 dex.
- **Rough effort:** 3–6 engineer-months to reach a trainable, mostly faithful v1 limited to that dex, assuming heavy LLM-assisted codegen as Jaxcalibur used. Full PS parity is open-ended.

---

## 1. What Jaxcalibur tells us (the blueprint)

| Aspect | Jaxcalibur (Gen 9 randbats singles) | Implication for VGC |
|---|---|---|
| Engine | Pure JAX. Every possible step (moves, switches, end-of-turn) always runs and irrelevant results are discarded. This wastes FLOPs but gives every action the same computational structure. | Same approach. Doubles roughly doubles the per-turn work: 4 actors, a 3-target axis, spread resolution. |
| Network | 8.5M-param non-causal transformer with pre-RMSNorm and a "position-MoE" (separate MLPs for Pokémon, move and event tokens). Tokens: field, actives, candidate moves, 6+6 team, 16 recent events, 24 matchup tokens holding precomputed damage calcs. | Reuse the design. Add ally-relationship tokens and a joint 2-slot action head (§5). |
| Aux heads | Value; opponent item, ability, moves and Tera type; opponent's next action | With open team sheets, item/ability/moves heads become **Stat Point spread + brought-4 + lead + Mega** prediction. |
| RL | PPO + GAE, undiscounted 0/1 reward, entropy bonus plus a "zero-avoiding" KL-to-uniform term so situational moves like Encore don't die out | Same approach. Pure self-play risks cycles in VGC (team rock-paper-scissors). Budget for a league or PSRO (VGC-Bench saw exploitability grow with team-pool size). |
| Search | Test time only. pUCT modified for simultaneous moves (regret-matching-like edge sampling), worlds sampled from the network's beliefs, HP binned into 10 buckets, depth 4, 20,480 rollouts, 32 worlds, a few seconds per decision on GPU | Open team sheets shrink the hidden state a lot, so search should be *more* effective (the author makes this point). Doubles has a much larger joint action space (§5). |
| Compute | About 1 week on 1 H100 for about 100M games. Smaller networks trained longer beat bigger ones. | My derived figure: about 165 games/s end to end, including network inference and updates. Doubles plus team preview will be slower per game. Plan for several GPU-weeks. |
| Validation | Not described | You must build this (§6). |

Sources: https://jaxcalibur.github.io (verified), and the Smogon thread https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/. In the thread the author says doubles is possible "with more work", and that the narrow randbats pool is what made the engine tractable.

---

## 2. Decisions to make before writing code

1. **Target format.**
   - **Champions Reg M (recommended).** It's where the players, the data and the tournaments are. Downsides: mechanics are newer and PS support is less mature.
   - **SV Reg G/I.** Older and more stable in PS, and VGC-Bench's 700k logs are SV. Downside: about 9.9k ladder games per month, a dying meta.
   - Either way, keep mechanics constants behind a `Ruleset` config so you can switch.
2. **Team scope.**
   - (a) A fixed small team pool. Easiest, and VGC-Bench shows it's already hard.
   - (b) A sampled pool from usage stats or the `metamon-teams` / VGC-Bench team sets.
   - (c) Team building as a separate outer loop. Out of scope for the engine.
3. **Fidelity bar.**
   - "Good enough to train a strong policy that transfers to Showdown" is not the same as "bit-exact with PS".
   - Recommendation: exact damage formula and turn order for the v1 dex, a conformance metric tracked in CI, and accept a documented list of known gaps.
4. **Hidden information model.** With open team sheets only spreads, the brought 4, and counters/PP are hidden. Choose between:
   - asymmetric actor-critic (the critic sees everything);
   - an explicit belief over K spread archetypes.

---

## 3. Engine architecture

### 3.1 API (gymnax/pgx style, pure functions)
```python
state = init(key, teams: TeamBatch, ruleset: Ruleset)   # team-preview phase
obs_p0, obs_p1, mask_p0, mask_p1 = observe(state)
state = step(state, action_p0, action_p1, key)           # runs until the next decision point
```
- **Simultaneous moves.** Both players act in every call. A player with nothing to choose (e.g. only the opponent must replace a fainted mon) gets an all-pass mask.
- **Phase machine** (part of the state): `{TEAM_PREVIEW, CHOOSE, MIDTURN_SWITCH(side, slot), EOT_REPLACE, DONE}`. U-turn, Parting Shot, Eject Button, Eject Pack and Emergency Exit suspend the turn halfway. So the **action queue and its cursor live in the state**, and `step` resumes from them.
- **RNG.** Pass the key explicitly, as in pgx v2. Inside a turn, derive draws by `fold_in(key, turn, actor_idx, EVENT_ID)` with static event IDs (ACC, CRIT, ROLL, SECONDARY, …). Draws then don't depend on which branches are masked, and replay from (seed, actions) is deterministic.
  - Don't try to match PS's sequential ChaCha20 stream; masked branches make that impossible.
  - For testing, use a **forced-outcome hook**: an optional per-event override tensor.
- **Terminal handling** follows pgx's `core.step`: `lax.cond` no-ops on finished states, so `vmap`/`scan` keep a fixed shape. Add an auto-reset wrapper for training.

### 3.2 State layout (struct-of-arrays, leading batch dim B)
- **Per mon `[B, 2, 6]`** (about 70 scalars, roughly 150 B each):
  - identity and typing: species, types[2], is_mega (Champions has no Tera; Mega changes species, types, ability and stats);
  - stats: stats[6], hp, maxhp;
  - status and boosts: status plus status_ctr, boosts[7];
  - moveset and items: moves[4], pp[4], item, item_consumed, ability, ability_suppressed;
  - position and history: brought, slot, fainted, turns_active, last_move, choice_lock, protect_ctr, proto_stat, weight;
  - volatiles: `vol_bits: uint64`, `vol_ctr[16]: int8`, `vol_arg[8]: int16`.
- **Per side `[B, 2]`:** screen and Tailwind turns, hazards, one-turn guards (Wide Guard, Quick Guard), mega_used, fainted_last_turn, and per-slot Wish / Future Sight.
- **Field `[B]`:** weather and terrain (enum plus turns), Trick Room / Gravity / rooms turns, turn number, phase.
- **Turn scratch:** `queue[≤8]` of (side, slot, kind, move, target_slot, mega, prio, frac_prio, done).
- **Total:** about 2 KB per battle, so 64k+ parallel battles per GPU is memory-trivial.
- **Targets are stored as slot indices** and resolved at execution time, which gives correct behaviour for Ally Switch, switch-ins and retargeting.

### 3.3 Turn resolution (fixed-shape)
1. Switches (order 103), then Mega Evolution (104), then moves (200), then residuals (300). Queue order values are from PS `sim/battle-queue.ts`.
2. **Dynamic speed (Gen 8+).** Don't sort once. Run a fixed `fori_loop` of up to 6 iterations. Each iteration picks the next action by argmax over (order, priority, frac_prio, *current* effective speed, random tiebreak).
   - Trick Room flips the speed key.
   - This handles mid-turn Tailwind, Icy Wind and Trick Room, plus After You, Quash and Quick Claw.
3. **Per-move execution over a fixed 3-target axis.** Steps, in order:
   - resolve the target set (redirection, retargeting at random to a living foe, Commander untargetability);
   - per target: Protect / Wide Guard / Quick Guard / Psychic Terrain / priority-blocker checks, then immunity, then accuracy, then damage;
   - set `spreadHit` from the final target count, so spread moves take ×0.75 (3072/4096) only if more than one target remains;
   - secondaries, then items and abilities reacting.
4. **End of turn.** Residuals in PS `onResidualOrder` order, then the faint check, then the EOT_REPLACE phase if needed.
5. **Switch-in abilities** (Intimidate, weather/terrain setters, Ruin, Protosynthesis) trigger in **speed order** when several mons enter together. This decides weather wars.

### 3.4 Effects: data tables plus a small handler set
- PS effects are JS event callbacks (`onModifyDamage`, `onTryHit`, …). They **can't be transpiled mechanically**.
- Step 1: export the common fields to dense arrays from `@pkmn/data` / `@pkmn/dex`. Fields include BP, type, category, accuracy, priority, target type, flags, boosts, secondary chance and effect, drain/recoil, multihit, and BP/stat modifiers per item or ability.
- Step 2: represent only the residue as `lax.switch` handlers keyed by effect ID.
- **Cost warning.** Under `vmap`, `lax.switch` with a batched index executes *every* branch. Cost grows with the number of handlers, which is another reason to restrict the dex.
  - Group handlers by hook point (on-switch-in, modify-BP, modify-atk, on-hit, residual, …) so each hook's switch stays small.
  - Prefer multiplier tables to handlers wherever possible.
- **Damage formula** (verified against PS `battle-actions.ts`):
  - integer `trunc` math;
  - `modify()` rounds half down on a 4096 base;
  - order: spread → weather → crit ×1.5 → random roll 85–100 → STAB (1.5, or 2.0 with Adaptability; Champions has no Tera STAB) → type → burn → final chain (screens at **2732/4096 in doubles**, Life Orb 5324/4096, …).
  - Use `@smogon/calc` as an oracle for unit tests.

---

## 4. Mechanics scope for v1 (doubles-specific must-haves)

Beyond singles basics (damage, types, status, stat stages, the PP and choice system):

- **Team preview:** bring 4 of 6 and choose leads, 180 ordered options per side.
- **Targeting:** all PS target types (`normal`, `adjacentFoe`, `adjacentAlly`, `allAdjacentFoes`, `allAdjacent`, `allySide`, `foeSide`, `all`, `randomNormal`, `scripted`), retargeting, Ally Switch.
- **Redirection:** Follow Me, Rage Powder (a powder move: Grass types, Overcoat and Safety Goggles are immune), Lightning Rod, Storm Drain. Ignored by Stalwart, Propeller Tail and Snipe Shot.
- **Protection:**
  - Protect and its variants (Spiky Shield, Silk Trap, Burning Bulwark, Baneful Bunker), with consecutive-use success at (1/3)^n;
  - Wide Guard and Quick Guard;
  - bypasses: Feint and Unseen Fist.
- **Tempo:**
  - Fake Out (first turn only, so track `turns_active`), Tailwind (4 turns), Trick Room (5 turns, priority −7, toggles), Icy Wind and Electroweb;
  - Helping Hand (×1.5, stacks);
  - Quick Claw and other fractional priority.
- **Switch-in effects:** Intimidate and its full interaction list (Clear Amulet, Defiant/Competitive, Guard Dog, Mirror Armor, Inner Focus/Oblivious/Own Tempo/Scrappy, Rattled, Adrenaline Orb, Mirror Herb, White Herb).
- **Field:** weather and terrain wars (Drought, Drizzle, Sand Stream, Snow Warning, the Surges, Orichalcum Pulse, Hadron Engine), Protosynthesis / Quark Drive, Booster Energy, the Ruin abilities.
- **Special cases:** Commander (Tatsugiri + Dondozo). Parting Shot, U-turn, Volt Switch and Eject Pack / Eject Button need the mid-turn phase.
- **Items:** Choice items, Focus Sash, Assault Vest, Sitrus, Life Orb, Covert Cloak, Clear Amulet, Mirror Herb, Safety Goggles, Rocky Helmet, Booster Energy.
- **Champions-specific (verified in PS `data/mods/champions/`, 2026-10-21):**
  - Tera disabled (`canTerastallize` returns null); Mega Evolution via Mega Stones, once per side.
  - Level-50 stats are `base + SP + 20` (HP: `base + SP + 75`), with nature ×1.1 / ×0.9 and no IVs.
  - All move PP capped at 20.
  - Full paralysis is 1/8; freeze thaws on a 1/4 roll with a turn limit (`conditions.ts`).
  - Trick Room has no speed underflow.
  - Overrides for `modifyDamage`, `spreadMoveHit` and `hitStepMoveHitLoop`, plus about 1.1k lines of move tweaks and 1k lines of item tweaks.
  - The format is `[Gen 9 Champions] VGC 2026 Reg M-C` (doubles, Open Team Sheets; the Bo3 variant forces OTS).

The exact v1 dex should be generated from the Smogon chaos JSON: everything up to 99% usage at ≥1760 in the target format. The research agent computed these coverage numbers:

| Format | Species | Moves | Abilities | Items |
|---|---|---|---|---|
| Champions Reg M-C Bo3, ≥1760 (90/95/99%) | 39/52/92 of 191 | 77/100/158 of 473 | 31/40/65 of 167 | 29/36/60 of 146 |
| SV Reg G, ≥1760 | 38/49/79 of 151 | 84/112/169 of 370 | 26/34/55 of 116 | 18/25/39 of 103 |

Teams are sampled only from the supported dex, so unsupported effects never come up.

---

## 5. Action and observation spaces

- **Per slot (compact):**
  - 4 moves × 3 targets (foe L, foe R, ally; non-targeting moves use one canonical index) × Mega {0, 1} = 24;
  - plus 2 bench switches, plus pass;
  - **27 actions**. Add a Mega flag if Champions needs one.
- **Joint per side:** 27² = 729, masked for double Mega and both slots switching to the same bench mon. Mid-turn switch: 3 options. EOT replacement: ≤ 2 per slot. Preview: 180.
- **Recommended head.** A joint head with low-rank interaction: `logit[i,j] = a_i + b_j + <u_i, v_j>`.
  - It uses an exact joint mask.
  - It captures coordination like Protect + attack, or Fake Out + Trick Room.
  - Avoid independent per-slot heads followed by "fixing up" illegal pairs, which biases the PPO gradient.
  - The alternative is autoregressive (slot B conditioned on slot A).
- For reference, poke-env doubles uses 107 actions per slot as a MultiDiscrete pair, most of it dead space in Gen 9.
- **Observation:** Jaxcalibur-style tokens. Doubles additions:
  - 2+2 active tokens and an ally-pair token;
  - damage-calc matchup tokens for every (own active × foe active × move) combination, including spread-adjusted values;
  - brought/unrevealed masks, and the opponent's full open team sheet.
  - The engine computes the damage calcs, so expose `calc_matrix(state)` as a jitted helper.

---

## 6. Validation (the biggest work item)

Evidence that this dominates the effort:
- vgc-engine's 2026-09 audit (verified): only 16.5% of battles fully match even with RNG forced, and about 94% of divergences are mechanics bugs.
- PS itself reproduces only about 33% of real-log turns from reconstructed teams, because of hidden information.

Plan:
1. **Oracle:** pinned `smogon/pokemon-showdown` (or `@pkmn/sim`), driven by a Node harness.
2. **Unit tests:** damage against `@smogon/calc` across random (attacker, defender, move, field) samples drawn from the dex.
3. **Forced-outcome differential tests:**
   - generate random legal battles;
   - record PS's chance outcomes per event (rolls, crits, hits, ties);
   - replay them through the JAX engine with the same outcomes forced;
   - diff state after each turn (HP, status, boosts, field, positions).
   - pkmn/engine (`ExhaustiveRunner`) and `mfflanag1/pokemon-rl` both use this pattern.
4. **Distribution tests** for RNG-dependent mechanics (paralysis, multihit, confusion, Protect chain).
5. **Real-log replay** of VGC-Bench's 700k SV logs or spectated Champions logs, using the turn-level match rate as the tracked metric.
6. **Fuzzing in CI.** Every divergence becomes a regression fixture.

---

## 6a. Automated, effectively unlimited differential testing against PS

**Feasibility: yes.** The checks below were made against the PS source, commit `d601443`, 2026-10-21:
- **Every random draw goes through one object.** `battle.prng` handles `random`, `randomChance`, `sample` and `shuffle`. That is about 80 call sites across `sim/` + `data/` + the Champions mod, a finite list to classify.
- **Random draws can be overridden.** `forceRandomChance` exists in debug mode, and `battle.prng` can be swapped for an instrumented proxy.
- **Battles can be started from any state.** `sim/state.ts` provides `serializeBattle` and `deserializeBattle`.
- **Legal actions come with each request.** PS sends the legal choices with every decision request.
- **The target format exists.** `[Gen 9 Champions] VGC 2026 Reg M-C` is implemented in `data/mods/champions/`.

**Loop** (runs unattended; each divergence becomes a permanent regression test):
1. **Generate.** Sample a legal Reg M-C team pair (from usage-stats sets plus a random tail, checked by PS's `TeamValidator`). Pick a scenario: from the start, or from a crafted mid-battle state via `deserializeBattle`, with weather, Trick Room, low HP, volatiles and so on.
2. **Play in PS** with an instrumented PRNG. Each draw is logged with its context: turn, `battle.effect`, the active move, the source and target slots, and the draw kind. Each draw is mapped to a static engine event ID (ACC, CRIT, ROLL, SECONDARY, SPEED_TIE, MULTIHIT, PARA, SLEEP_LEN, RANDOM_TARGET, …).
   - Choices come from a mix of uniform-random legal moves, heuristics, and later the current policy, so testing covers the states training actually visits.
3. **Replay in JAX**, forcing the same outcome per (turn, actor, target, event).
4. **Diff at every decision point:**
   - state projection: HP, status and counters, boosts, item and ability, field and side conditions, positions, PP, Mega;
   - **legal-action mask against PS's request JSON**, which catches masking bugs that silently bias PPO;
   - per-target damage values.
5. **Shrink on mismatch.** Truncate turns, delete bench mons, and replace irrelevant moves until the case is minimal. Save it as a fixture tagged with the PS handlers that fired.
6. **Coverage-guided generation.** Hook PS `runEvent` to count which (move, ability, item, condition, event) handlers fired and which pairs fired together. Bias the generator toward combinations not yet exercised. Uniform random play almost never produces Commander + Ally Switch + Follow Me, for example.
7. **Distribution tests** for things forcing can't cover well: Protect chains, paralysis, sleep length, multihit. Run them with free RNG on both sides and compare with chi-square tests.

**Scale.** PS runs on the order of tens of battles per second per CPU core, so a many-core box produces millions of battles per day. The JAX side runs batched. The cost is CPU time, not human effort.

**What it can't give you:**
- **PS is the oracle, not the Champions game.** The Champions mod is new; `test/sim/champions.js` has 48 tests. Bugs in PS get faithfully copied. Mitigations:
  - re-pin PS regularly and re-run the suite;
  - read PS commits to `data/mods/champions/` as mechanics changelogs;
  - keep a small hand-written suite from community in-game research for disputed mechanics.
  - Policies played on Showdown face PS mechanics anyway, so PS-parity is the right first target.
- **Random coverage is not proof.** Rare interactions need coverage guidance and crafted states, and the test suite only covers the dex you've chosen.
- **The call-site mapping is a maintained artefact.** If PS refactors where it draws randomness, the mapping must be updated. Detect this with a "draw consumed with no matching engine event" alarm.

**Order of work:** build this harness *before* the long-tail mechanics (M3 before M4). Mechanics work then becomes an autonomous fix-loop driven by the harness, which is how an LLM coding agent can carry most of the M4 burden.

## 7. Throughput target and performance notes

- **Baseline:** Jaxcalibur end to end is about 165 games/s on 1 H100 including learning (my derivation from 100M games per week). PS is about 16 battles/s per process. pkmn/engine (Zig, Gen 1) is about 51k battles/s per CPU thread.
- **Target:** at least 10⁵ env-steps/s for the engine alone on 1 H100 at B ≈ 16–64k. Profile early, because doubles × 3-target × all-branches can blow up compile time and memory.
- **Practices:**
  - `lax.fori_loop` / `scan` rather than Python loops;
  - avoid `lax.switch` explosions;
  - keep int16/int8 dtypes;
  - write the whole rollout (env + policy) as one `scan`, as in PureJaxRL;
  - watch XLA compile time. Large handler sets can push it to minutes, so split hooks into separately jitted pieces during development.

---

## 8. Training stack (downstream of the engine)

- **Algorithm:** PPO + GAE in a PureJaxRL-style loop, following Jaxcalibur. Team preview is a separate decision head at t=0.
- **Opponents:** self-play plus a league / fictitious-play pool. VGC-Bench's PSRO results show plain self-play becomes exploitable as team diversity grows.
- **Warm start:** behaviour cloning from VGC-Bench's 700k+ human logs (HF `vgc-battle-logs`) or the pokechamp dataset. This needs a log→engine-state converter, which the validation harness gives you almost for free.
- **Showdown evaluation** via poke-env `DoubleBattle` / `DoublesEnv`: obs-translation layer from PS protocol to engine state. Baselines: VGC-Bench agents, PokéChamp (which supports VGC), heuristics.
- **Test-time search** (optional, later): Jaxcalibur-style pUCT over sampled spread worlds. The engine needs `set_state` / `clone` and per-rollout seeds, which pure-functional JAX provides.

---

## 9. Work breakdown and rough estimates

Estimates assume one experienced ML engineer with heavy LLM-assisted codegen.

| # | Milestone | Deliverable | Est. |
|---|---|---|---|
| M0 | Decisions + data pipeline | Format chosen; `@pkmn/data` → NumPy tables; dex subset from usage stats; team sampler | 1–2 wk |
| M1 | Core singles-style kernel in doubles layout | State pytree, damage formula (calc-validated), types, status, boosts, PP, basic moves, switching, faint/replace phases, RNG scheme | 3–4 wk |
| M2 | Doubles turn engine | Team preview, dynamic-speed queue loop, 3-target resolution, spread, redirection, retargeting, Protect family, mid-turn switch phase | 3–4 wk |
| M3 | Oracle harness | Node PS driver, forced-outcome replay, state diff, fuzz CI, calc tests | 2–3 wk (parallel with M2) |
| M4 | Meta-relevant effects | Remaining v1 moves, abilities and items (Intimidate set, weather/terrain, Paradox, Ruin, Commander, items); burn down divergences to ≥90–95% turn match | 4–8 wk (long tail) |
| M5 | RL integration | Obs encoder, joint action head, PPO loop, auto-reset, throughput tuning | 2–3 wk |
| M6 | Eval bridge | poke-env adapter, ladder/baseline evals, BC warm start | 2–3 wk |

**Total:** about 4–6 months to a trainable, validated v1. This matches the Jaxcalibur author's "at least a few months" for singles randbats. Doubles is harder, but open team sheets plus a fixed dex is narrower than randbats in other ways.

**Compute:** engine development is cheap. A training run is roughly 1–4 H100-weeks (several hundred to a few thousand dollars), scaling with team-pool diversity.

---

## 10. Key risks

1. **Fidelity long tail.** A policy may exploit engine bugs that don't exist on Showdown. Mitigate with conformance CI and Showdown evals, and by restricting teams to the validated dex.
2. **Format churn.** Champions regulations change roughly every few months and add mechanics. Keep data tables generated, not hand-written.
3. **XLA compile time and all-branch cost** as handlers accumulate.
4. **Game-theoretic difficulty.** Simultaneous moves plus team-matchup cycles mean PPO self-play may cycle; a league or PSRO is likely needed.
5. **Champions mechanics in PS** may themselves be incomplete, since the oracle is only as good as PS.

---

## 11. References

- Jaxcalibur write-up: https://jaxcalibur.github.io · Smogon thread linked in §1
- Pokémon Showdown: https://github.com/smogon/pokemon-showdown (`sim/battle.ts`, `battle-actions.ts`, `battle-queue.ts`, `data/*.ts`, `SIM-PROTOCOL.md`)
- pkmn: https://github.com/pkmn/ps (`@pkmn/sim`, `@pkmn/data`) · https://github.com/pkmn/engine (state layout and testing design, `docs/TESTING.md`)
- vgc-engine (Rust, doubles/Champions): https://github.com/colefoster/vgc-engine, especially `docs/accuracy/2026-09-accuracy-proof.md`
- poke-engine / foul-play (note foul-play is GPL-3.0): https://github.com/pmariglia/poke-engine · https://github.com/pmariglia/foul-play
- poke-env: https://github.com/hsahovic/poke-env
- VGC-Bench: https://github.com/cameronangliss/vgc-bench (arXiv 2506.10326)
- PokeAgent Challenge: https://pokeagent.github.io · Metamon: https://github.com/UT-Austin-RPL/metamon · PokéChamp: https://github.com/sethkarten/pokechamp
- VGC AI Framework: https://gitlab.com/DracoStriker/pokemon-vgc-engine
- JAX: minimon https://github.com/spktrm/minimon · pgx https://github.com/sotetsuk/pgx · gymnax https://github.com/RobertTLange/gymnax · jumanji https://github.com/instadeepai/jumanji · PureJaxRL https://github.com/luchris429/purejaxrl
- Usage stats: https://www.smogon.com/stats/2026-09/ (chaos JSON) · Pikalytics · Victory Road
