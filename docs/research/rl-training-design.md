# RL and training-system requirements for a pure-JAX Champions VGC engine

*Research date: 2026-10-21. Scope: what the **engine API** must provide so that a Jaxcalibur-style trainer (PPO + GAE self-play, ~8–30M-param transformer, 1–8 H100s, optional test-time search) can be built on it without redesign.*

*Labels: **(verified)** means I read the primary source (paper, code or data) myself. **(measured)** means I computed it from downloaded data. **(inference)** means my own reasoning or estimate. Unlabelled background claims are well-known literature.*

Companion to `../jax-vgc-engine-scope.md`. Where this note disagrees with the scope doc, this note is newer (see §0.3).

---

## 0. TL;DR

### 0.1 Engine requirements (details in §7)
1. **Pure, single-battle functions, vmapped outside** (pgx/minimon style). `init(key, teams)`, `step(state, actions[2], key)`, `observe(state, player)`, `legal(state, player)`. Explicit keys and no global state.
2. **Both players act on every step.** A player with no choice gets a pass-only mask and `has_decision=False`. The engine auto-advances through steps where *neither* player has a real choice.
3. **Phases:**
   - `PREVIEW`: 90 options, bring-4 × lead pair.
   - `MOVE`: 29 actions per slot, joint 29² = 841, with an exact joint mask.
   - `FORCE_SWITCH` (end-of-turn replacement, both sides blind) and `MIDTURN_SWITCH` (U-turn etc., one side).
   - All battle decisions share the per-slot action space; only the masks differ.
4. **Per-player observations come from the information state only.**
   - Canonical perspective: self = index 0.
   - Opponent HP uses PS's **Champions rule `floor(100·hp/maxhp) || 1`**, plus a colour bit at 20 and 50 **(verified)**.
   - Hidden fields: the opponent's Stat Points, unrevealed brought mons and exact HP. Nature is **public** in Champions OTS **(verified)**.
   - The damage calculator is a separate model-callable function on *hypothesised* spreads, never on the true state.
   - CI property test: perturbing hidden fields must leave the observation unchanged.
5. **Privileged observation** (same pytree, hidden fields filled) for an asymmetric critic. **Hidden-info labels** (opponent SP spread, brought set, opponent action) for auxiliary/belief heads.
6. **Zero-sum terminal reward** only. Shaping *features* go in `info` for optional potential-based shaping.
7. **Turn cap** (~40 turns) with a VGC-style tie-break. Truncation is flagged, and `final_obs` is provided.
8. **Auto-reset wrapper** samples new team pairs **inside jit** from a device-resident `TeamBank`, using weights supplied by the trainer.
   - Heterogeneous teams per batch come for free, because everything is data-driven.
   - SP spreads are sampled per mon from candidate menus at reset. The same menus serve as the belief prior.
9. **Episode statistics:** winner, turns, end reason, team IDs, opponent-policy ID, preview choices, Mega use, error flags. They are enough to update matchup tables in-jit.
10. **Search support:**
    - free cloning;
    - per-rollout keys;
    - `determinize(infostate, hidden_sample)`, which is also the PS→engine state builder at deployment;
    - `hash_infostate(hp_buckets)`;
    - `apply_preview` for 90×90 preview matrix solves;
    - forced-outcome overrides, plus optional outcome enumeration.
11. **Interop and debug:**
    - an action ↔ PS choice-string map;
    - PS request JSON → mask comparison;
    - a chance-outcome trace, so any engine game can be **re-simulated in real PS** to produce a genuine replay. This doubles as a differential test.
    - Error flags live in a `uint32` bitfield (no exceptions inside jit).

### 0.2 Throughput target (details in §8)
- **Measured Reg M-C game length** (800 recent PS replays): mean ≈ **8 turns** (median 7, p99 15–19). That is **≈ 12.4 env steps per game**: preview, about 8 turns, about 2.9 end-of-turn replacement phases, and about 0.5 mid-turn switches.
- **Jaxcalibur's pipeline:** about 165 games/s, about 5k env-steps/s, about 10k policy samples/s on 1 H100, including learning.
- **The same sample budget in VGC** is about 400 games/s, about **5k env-steps/s**, about **240M games/week**.
- An 8.5M-param, ~100-token model uses roughly half of an H100 at that rate. The engine must therefore stay under ~20% of wall-clock. **Engine targets:**
  - minimum **≥3×10⁴ env-steps/s per H100**, standalone, at batch 16k–64k;
  - **target 10⁵**;
  - **≥2×10⁴ at batch 1k–4k** for test-time search;
  - compile time under ~2 min;
  - under ~15 GB at batch 64k.
- Larger models (30M) or more GPUs *relax* the per-GPU requirement, because the network becomes the bottleneck.

### 0.3 Findings that change earlier assumptions
- **PokaiTrainer** (arXiv 2608.29197, Aug 2026) **(verified)**:
  - A Champions Reg M-B VGC agent built on a Rust engine (~99% PS parity, 0.08 ms/turn) and Student-of-Games-style CFR search.
  - Already rated on the live PS Bo3 ladder: 1350–1400 Elo band, peak 1492.
  - Its policy heads alone were weak, and a 256k-game PPO baseline lost to a greedy search. **But nobody has tried Jaxcalibur-scale (10⁸-game) model-free training on VGC.** That is exactly what a JAX engine enables.
- **PokeJAX** (arXiv 2603.12145, Mar 2026) **(verified)**:
  - An LLM-translated, 55k-line JAX port of Showdown claiming 5×10⁸ random-action SPS.
  - The format isn't stated, it appears to be singles, I found no public code, and its fidelity check is only win-rate equivalence.
  - Takeaway: the all-branches approach scales, but **validation remains the hard part**.
- **VGC-Bench's dataset is now Champions-only** **(verified)**:
  - 88,905 OTS logs, Reg M-A and M-B, last updated 2026-06-20. The 700k SV logs are archived separately.
  - **No Reg M-C data**, so Reg M-C BC data must be scraped.
  - Only **15% of Bo1 Reg M-C replays use OTS** (61/400), against 100% of Bo3 (measured). **Deploy and evaluate on the Bo3 format**, `gen9championsvgc2026regmcbo3`.
- **VGC-Bench's "degrades with more teams" result came from only 5.0M PPO timesteps** **(verified, Table 7)**. That is about 1000× less than Jaxcalibur. It shows sample starvation, not a proven limit of self-play.
- **The PS bot policy changed on 2026-10-09** (verified via the Smogon thread):
  - a bot **registry**, with unregistered bots bannable;
  - opt-in bot matchmaking;
  - bot games don't affect human ratings;
  - AI assistance in tournaments counts as ghosting (immediate ban).

---

## 1. Jaxcalibur in detail (the blueprint)

Source: https://jaxcalibur.github.io (verified; re-read in full on 2026-10-21, including the architecture diagram `architecture.png`). Community thread: https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/ (read via fetch).

### 1.1 What it is
- A Gen 9 Random Battles (singles) bot. It was on and off **#1 on the PS randbats ladder from 6 to 27 August** (2026, inferred from the thread IDs and dates).
  - Peak **Elo 2557, GXE 95.3**. Top of the ladder ~20% of the time. **68% win rate against other top players.**
  - The previous best bot, Foul Play, peaked at ~2341 / 88 GXE.
  - One human (MichaelderBeste2) did better over the same period: top 50% of the time, GXE 95.5.
- Design resembles AlphaGo Zero. Uses **poke-env** to connect to Showdown.
- **Code and weights not released.**
  - The author stopped searching for ladder games on 28 Aug and let the rating decay.
  - On 12 Sep they decided not to release, citing "preserving the integrity of the ladder and the broader Showdown/Smogon competitive scene".
  - Community and staff reactions: "bot ghosting is a major hot-button issue"; Smogon "needs an official policy… surrounding bots". The VGC leader offered to collaborate.
  - The author noted the model "can run on an ordinary laptop and still be very strong" without search.

### 1.2 Network and observation tokens
Non-causal transformer, pre-RMSNorm, with a **position-based mixture of experts**: Pokémon, move and event positions use different MLP weights. 8.5M parameters.

| Tokens | Count | Content |
|---|---|---|
| Battlefield | 1 | weather, hazards, etc. |
| Active Pokémon | 2 | stats, moves, damage calcs, etc. |
| Candidate moves | 5 | 4 moves + Struggle |
| Own team | 6 | |
| Opponent team | 6 | masked if not yet revealed |
| Event history | 16 | last 16 game events (moves, switch-ins) |
| Matchup tokens | 24 | damage calcs etc. per Pokémon pair; each own mon matched to 4 opposing mons |

- **Matchup tokens are added halfway through the network.** The diagram shows mid-network heads that "predict item, ability, moves", whose outputs feed the **damage calcs** inside the matchup tokens. **The calcs are computed inside the forward pass from the network's beliefs**, not by the environment from the true state. (verified from the diagram)
- **Implication:** the engine's damage calculator must be a pure, jittable function callable from model code on *hypothesised* attributes (for VGC, hypothesised Stat Point spreads). If it reads the true state, it leaks.

### 1.3 Heads
- **Actions are pointer-style.** Each own-team token emits a "switch to me" logit. Each move token emits "use" and "Tera + use".
- **Value:** p(win), read from the battlefield token.
- **Opponent attributes:** item, ability, moves, Tera. Item and ability predictions are kept across turns.
- **Opponent's next action:** "made the bot substantially stronger".

### 1.4 Training
- Pure self-play PPO with GAE. **Undiscounted 1/0 win/loss reward**, no shaping.
- **Regularisation:** an entropy bonus, plus a **"zero-avoiding" KL toward uniform**.
  - Inference: "zero-avoiding" means the *forward* KL(U‖π) = −mean log π + const. It heavily penalises near-zero probabilities, so a situational move like Encore never dies out. The entropy bonus is the reverse KL and lets probabilities approach zero.
- **No search during training.** About **1 week on 1 H100, about 100M games.** "Smaller networks trained for longer outperformed larger networks trained for fewer steps."
- Not disclosed: PPO hyperparameters, batch or env counts, whether past checkpoints were used as opponents.

### 1.5 Engine
- Pure JAX. "Always executes every possible game step … and keeps only the relevant results."
- "The bottleneck in reinforcement learning is often the environment."
- Built with heavy Claude Code use.

### 1.6 Search, and what it needs from the engine
pUCT with four changes (verified):
1. **Simultaneous moves.** Edges are sampled with π(s,a) ∝ max(Q(s,a) − V(s), 0) + c·P(s,a)/√N(s). This moves toward regret matching, so the output is a mixed strategy.
2. **Chance.** No chance nodes. **Every rollout replays from scratch with a fresh seed.** Statistics live in a dict {state_hash → node}, with **HP binned into 10 buckets**.
3. **Own missing information.** **Worlds** (the opponent's hidden sets) are sampled from the network's predictions. Our nodes are hashed on revealed information; opponent nodes are also hashed on the world ID.
4. **Opponent's missing information.** The simulated opponent's search is truncated when we do something "surprising" (a move it predicted at <65%, or switching to an unrevealed mon). After that it uses only the network prior.

- Budget: **depth 4 (≈2 turns), 20,480 rollouts, 32 worlds, a few seconds per decision on GPU.** Worth about **+100–150 Elo**. "Most of Jaxcalibur's strength comes from having a really good neural net."

| Search feature | Engine requirement (inference) |
|---|---|
| From-scratch rollouts with fresh seeds | Pure `step(state, actions, key)`, vmappable over (state, key, actions). Cloning via `jax.tree.map(broadcast)`. |
| Worlds sampled from beliefs | `determinize(infostate, hidden_sample) → State`. The same function builds state from the PS protocol at deployment. |
| Hash dict with HP buckets | Jittable `hash_infostate(state, player, hp_buckets)` that ignores hidden fields, plus a world-salted variant. |
| Opponent truncation | Correct `observe(state, opponent)` on determinised states, to evaluate the opponent's prior. |
| Mixed strategies at simultaneous nodes | `step` takes both actions at once. |
| ~80k env steps in seconds | High throughput at batch 1k–20k, not only at training batch sizes. |

### 1.7 What transfers to VGC
- **Transfers:** the all-branches JAX engine, tokenised observation, position-MoE, pointer logits, 0/1 reward, entropy + KL-to-uniform, opponent-action auxiliary head, test-time-only search, small model trained long.
- **Changes:**
  - OTS shrinks hidden information to Stat Points, the brought 4 and exact HP. The item/ability/moves heads become **SP-spread + brought-set** heads.
  - The action is a **joint two-slot action** (841 options) plus a **preview** decision (90).
  - The team distribution is a design choice. Randbats gives fresh random teams for free.
  - The page is silent on opponent pools, but VGC team-matchup cycles probably require some population method (§4).

---

## 2. VGC-Bench (arXiv 2506.10326, github.com/cameronangliss/vgc-bench)

Sources (verified):
- the paper, v3 of 13 Jan 2026, AAMAS 2026 (Angliss, Cui, Hu, Rahman, Stone; UT Austin LARG): https://arxiv.org/html/2506.10326;
- repo source files `vgc_bench/src/{env,policy,utils}.py` and `train.py`;
- the commit log;
- HF dataset cards.

### 2.1 Infrastructure
- Built on **poke-env + PettingZoo**, with a real Showdown server in the loop.
- Toggles:
  - **exclude team preview** (a random preview decision, "potentially encouraging policy exploration in any team preview situation");
  - **disable mirror matches**.
- A team scraper for VGCPastes.
- **Status (Oct 2026):** Champions support added 2026-05-11 (PR #10). SV regulations removed and Reg M-B added 2026-06-21. Last commit 2026-07-25.
  - `utils.format_map = {"ma": "gen9championsvgc2026regma", "mb": "gen9championsvgc2026regmb"}`. **Reg M-C is not yet supported.**

### 2.2 Observation encoding
- **12 Pokémon vectors** (6 per side). Each concatenates per-Pokémon features p, its side's features s and global features g, giving shape 12×(g+s+p), with optional frame stacking.
- Code: `move_obs_len` = category + target + type one-hots + 12 scalars. `pokemon_obs_len` = 4×move + effects + gender + 2×types + status + 45 scalars.
- Moves, items and abilities get **learned 32-dim embeddings**. Species is represented through its stats and types.
- A rating feature is fed in (`fake_rating=2000` at play time), so BC is rating-conditioned.
- No event-history tokens. History comes only through frame stacking.

### 2.3 Action encoding and team preview
- **107 actions per slot** (Table 6):
  - −2 default, −1 forfeit, 0 pass;
  - 1–6 switch;
  - 7–26: 4 moves × 5 targets (self slot a, self slot b, no target, foe a, foe b);
  - then ×20 each for Mega, Z-Move, Dynamax and Tera variants.
- Mostly dead space for Champions. (This matches poke-env's `DoublesEnv`.)
- **Joint action as two sequential per-slot samples.** The second slot's mask is updated from the first slot's sampled action (no double switch to the same mon). The second slot's *logits do not condition on the first slot's choice* (`MaskedActorCriticPolicy._update_mask`). The factorisation is valid, but it cannot express coordination.
- **Team preview is two consecutive joint "switch-in" actions** (2 mons each, 4 total), unified with the battle action space.

### 2.4 Network
- SB3 `ActorCriticPolicy`, d_model 256.
- A **3-layer Transformer encoder over the 12 Pokémon tokens plus an aggregation (CLS) token**, then a linear layer to logits.
- Separate actor and critic of the same architecture (no parameter sharing).
- An optional temporal Transformer when frame stacking.

### 2.5 Training methods and scale
- **BC** on human logs. Then PPO in four flavours (from Lanctot et al.'s unified framework):
  - **self-play (SP)**;
  - **fictitious play (FP)**: uniform over past checkpoints, resampled each rollout period;
  - **double oracle (DO)**: Nash over the empirical payoff matrix, by LP;
  - **BC-initialised variants** BCSP, BCFP, BCDO.
- The README also lists "policy exploitation" (exploiter training).
- **PPO hyperparameters** (Table 7):

| Hyperparameter | Value |
|---|---|
| Learning rate | 1e-5 |
| γ | 1.0 |
| λ | 0.95 |
| Clip | 0.2 |
| Entropy coefficient | 0.001 |
| Value-function coefficient | 0.5 |
| Steps per update | 24×128 |
| Minibatch | 64 |
| Epochs | 10 |
| **Total timesteps** | **5,013,504** |

- Hardware: 8 A40s.
- **Reward:** terminal ±1 only (`env.calc_reward`).

### 2.6 Results: performance falls as the team count grows
- Team sets of 1, 4, 16 and 64, nested, 5 seeds. Cross-play cells are 1000 games (±0.03).
- **BC-initialised methods dominate.** Alpha-Rank #1 is BCSP at 1, 4 and 64 teams and BCDO at 16. Non-BC SP/FP/DO rank 4–8. FP and DO don't clearly beat SP.
- **Performance test** (Table 3, on the one team every agent trained on): the 1-team agent beats the 4-, 16- and 64-team agents **0.699 / 0.74 / 0.698**.
- **Generalisation test** (Table 4, 72 held-out teams, mean similarity ≈ 0.5): the 64-team agent beats the 1-, 4- and 16-team agents **0.669 / 0.578 / 0.564**.
- **Exploitability:** best-response agents trained against each agent (on the 1-team config) find that "in almost all cases, all agents are approximately **100% exploitable**." The 1-team agent resists a randomly initialised exploiter somewhat better.
- **Humans:** the 1-team BCSP agent won 5/5 against an intermediate player and 2/5 against an advanced one, and beat an expert (Aaron Traylor) at least once. "After enough successive games, strong human players can adapt and beat the agent."
- **The authors' explanations:** model-free RL can't fit "the complexity of predicting future outcomes" across many teams. They suggest search, opponent modelling and latent-variable BC.
- **My reading (inference):**
  - 5M timesteps spread over 64 teams (4,096 ordered matchups) is about 1.2k steps per matchup.
  - Jaxcalibur used about 6×10⁹ samples. The degradation is confounded with sample starvation by roughly three orders of magnitude.
  - The engine's job is to make the 10⁹–10¹⁰-sample regime cheap, so the question gets answered properly.

### 2.7 Dataset
- **Paper:** "over 700,000 OTS-enabled VGC games" (SV).
- **Now (HF `cameronangliss/vgc-battle-logs`, updated 2026-06-21):** **Champions-only**.
  - **88,905 logs** across `gen9championsvgc2026regma`, `…regmabo3`, `…regmb`, `…regmbbo3`. Latest log 2026-06-20.
  - This yields about 177,810 trajectories and about **1,474,324 transitions** (≈ 8.3 per trajectory).
  - Mostly low-rated: the Bo3 M-A file has only 4 logs at ≥1700.
  - SV logs (Regs A–J) are archived to `vgc-battle-logs-sv`.
- **No Reg M-C.** Teams: `teams/reg_ma`, `teams/reg_mb`.
- The logs are spectator-perspective. `logs2trajs.py` replays them through the environment to reconstruct each player's view; this is only possible because OTS reveals sets.

### 2.8 Evaluation methodology
- **Cross-play matrices** with uniformly drawn teams per game, and **Alpha-Rank** (arXiv 1903.01373) for ranking.
- A **performance test**: the intersection of all agents' training teams.
- A **generalisation test**: teams disjoint from all training sets.
- **Exploitability** via an RL-trained approximate best response, after Timbers et al. 2020 (arXiv 2004.09677).
- Heuristic baselines: Random, MaxBasePower, SimpleHeuristics (extended to doubles). An 8B LLM baseline.

### 2.9 Implications for our engine
- A ~1000× sample budget is the main lever. Throughput matters more than anything else.
- Support **preview on/off** (random preview) and **mirror on/off** as reset options.
- Use a joint action head with an exact joint mask, not sequential masking without conditioning.
- Provide a **frozen-opponent / exploiter mode** and **team-set splits** (train vs held-out) as first-class evaluation features.

---

## 3. Other relevant work

### 3.1 PokaiTrainer: equilibrium search for Champions VGC (arXiv 2608.29197, Max Yu, Aug/Sep 2026) **(verified)**
The closest existing system to our target.

- **Engine.** PokaiEngine, in Rust.
  - **Compute mode:** one timeline, **0.08 ms median per turn against Showdown's 2.0 ms** (26×, single process).
  - **Search mode:** forks at chance events and returns a joint action's **full weighted outcome distribution in one pass**. HP is carried as a damage-roll distribution and split only at KO thresholds. Median 9, maximum 18,176 branches per joint action. A 32-branch cap takes 1.0 ms and covers 99.5% of the mass.
  - Parity is "~99%" (6 of 500 random opening turns mismatched on the first pass), guarded by 600+ synthetic scenarios and 500+ replayed ladder battles.
  - One seat-view `BattleEnv` over three backends: the local engine, an offline Showdown process, and the live ladder. A Showdown adapter **reconstructs the same player-view state from protocol lines**.
- **Method.** Student-of-Games-style.
  - Every decision is a **Bayesian matrix game** over public belief states, solved by linear CFR.
  - **Worlds** = which 4 the opponent brought: 15 at preview, ≤6 after leads.
  - **SP spreads** are a per-mon mixture over a few candidates, conditioned on the nature revealed by OTS.
  - Leaves use a counterfactual value network (one value per world). Budgeted PUCT/CFR subgame growth.
- **Decision types** (Table 5):

| Decision | Who decides | Solve shape |
|---|---|---|
| Team preview | both | 90×90 |
| Turn | both | shortlisted joint menus, ~16×16 interior |
| Forced switch | both, blind | ≤2×2 per world |
| Mid-turn pivot | one seat | |

- **Belief over spreads uses the engine itself.** After each turn, the completed joint action is re-run per candidate spread. Each candidate is reweighted by the probability the engine assigns to the observed outcome and move order.
  - The **hidden-stat belief summary was worth +8–12 pp.**
  - Champions SP: **0–32 per stat, 66 total** (matches the PS validator; verified in `sim/team-validator.ts` and `sim/dex-formats.ts`).
- **Network.** A 14-token transformer, 256 wide, 6 layers, 5.4M parameters.
  - Tokens: our 6, their 6, a field token, a summary token.
  - Opponent tokens carry P(brought), and the field token carries the distribution over the 15 bring subsets.
  - BC on **122,804 replays (2.56M decisions)**, then 32 self-play rounds of 1k–3k games each, searching at every decision.
  - Compute: **about 250 L40S-hours self-play and about 15 A100-hours training.**
  - Teams: **663 VGCPastes teams** for self-play, 16 hand-picked for evaluation.
- **Results.**
  - Live PS Bo3 ladder (Reg M-B): **59% of 150 sets** against a field averaging ~1320. **Band 1350–1400 Elo, GXE 72–74, peak 1492** (top 500).
  - 64% of 83 opt-in private challenges.
  - "Search carries the strength": a greedy depth-1 belief search beats every raw policy head (71%).
  - A **BC+PPO baseline (VGC-Bench recipe, 256k games)** reached only 39.5% against greedy, and lost 27–73 to their round-1 network *under search*.
  - Decoupled UCT computes no equilibrium. An informed local best response wins **83.6% against a DUCT mix but 49.6% against the CFR mix**.
  - Opponent shortlists contain the realised *joint* action on only 64% of turns, against 97% per-slot marginal coverage. **Correlations between slots matter**, which supports a joint action head.
  - An early failure mode: attack-biased shortlists omitted Protect on 48% of legal turns.
- **Release:** "intend to release the engine and trainer with the live-ladder client removed", with checkpoints withheld while the format is live.
- **Relevance to us:**
  1. A strong, compute-cheap baseline that already beats PPO at small scale.
  2. Its game counts (~10⁵ self-play games) are tiny next to Jaxcalibur's 10⁸. The two regimes have not been compared at matched compute (§3.8).
  3. Its engine API (`branch()` on hypothesised states, seat-view env, protocol-reconstructed state) is a good template for our search and deployment interfaces.
  4. It handles Illusion by forfeiting, a reminder to scope the dex.

### 3.2 PokeJAX: LLM-translated Showdown in JAX (arXiv 2603.12145, Karten, Appapogu, Jin, Mar 2026) **(verified)**
- A coding agent (Gemini 3 Flash) translated Showdown (100K+ lines TS) into **55k lines of JAX across ~30 modules**.
  - **1,370 move effects dispatched with `lax.switch`.** "Every step pays the cost of all 1,370 move computations."
  - **JIT ~45 s**, **~28 GB at batch 65k**.
- **Claimed throughput:**
  - **500M SPS** random-action at batch 65k, against 21K SPS for single-threaded PS;
  - **15.2M SPS** in PPO with ~2M-param models, against 681 SPS through poke-env/WebSockets.
- **Verification:** 2,783 tests, plus **cross-backend policy transfer** (win rate against a heuristic bot equivalent within Δ=0.02 by TOST).
- The paper doesn't state the format; it appears to be singles. I found no public code link.
- **My read (inference):**
  - Encouraging on feasibility and compile cost.
  - The throughput figures are hard to interpret without the format and step definition.
  - Win-rate equivalence against a heuristic is a **much weaker** fidelity criterion than per-turn forced-outcome state diffs. A policy can exploit mechanics bugs that leave the average win rate against a heuristic unchanged.
  - Our validation plan (scope §6a) remains necessary.

### 3.3 Metamon (UT Austin; arXiv 2504.04395, RLC 2025; github.com/UT-Austin-RPL/metamon) **(verified, abstract and README)**
- Reconstructs first-person trajectories from spectator replays. **5M+ human and 20M+ self-play trajectories.**
- Trains **15M / 50M / 200M** transformer actor-critics offline (AMAGO). Offline RL plus value classification.
- Self-play scale:
  - SyntheticRLV2: 200M params, 1M human + 4M self-play battles.
  - Kadabra series: >11M self-play.
  - Kakuna: Superkazam (142M) fine-tuned on 7M more self-play. Kakuna is the best public model.
  - An ensemble (`TaurosEnsemble`) **held #1 on the human Gen 1 OU ladder**.
  - The paper's agents reached the top 10% while anonymous.
- **Singles OU only** (Gens 1–4 and 9). No doubles.
- Relevance: (i) offline RL from reconstructed replays is a viable warm start; (ii) the 2026 shift toward *smaller* models plus more self-play echoes Jaxcalibur.

### 3.4 PokeAgent Challenge (NeurIPS 2025; arXiv 2603.15563; pokeagentchallenge.com) **(verified)**
- Battling Track on **Gen 1 OU and Gen 9 OU**, played on a **dedicated, modified Showdown server "to avoid interfering with human players"**.
  - Ranked by Glicko-1, GXE and the organisers' own rating.
  - The infrastructure "supports three additional formats". The appendix defines a Gen 9 VGC (doubles) state space.
- Released data: **4M+ reconstructed human trajectories, 18M self-play trajectories, 200K+ teams.**
- Participation: 100+ teams across both tracks, 150+ submissions.
- **Winners:**
  - Gen 1 OU: **PA-Agent**, Metamon-based iterative offline RL. Six rounds shrank human data from 100% to 10%, with tournament-based team selection.
  - Gen 9 OU: **Foul Play**, root-parallel MCTS with decoupled UCB and Rust poke-engine. Won the final **50–14**.
- Now a living benchmark with a live leaderboard. Relevance: an **AI-only Showdown venue** is the right place for bot-vs-bot evaluation. It doesn't run Champions VGC (as far as I could verify).

### 3.5 PokéChamp (arXiv 2503.04094; github.com/sethkarten/pokechamp) **(verified, abstract and README)**
- LLM-guided minimax. The LLM samples actions, models the opponent and estimates value.
- Gen 9 OU: **76%** against the best previous LLM bot and **84%** against the strongest rule-based bot. Projected 1300–1500 Elo.
- Released a dataset of **3M+ games** (500k+ high-Elo).
- The repo has a **VGC LLM player** (`llm_vgc_player.py`, example `gen9vgc2025regi`, i.e. SV). A possible but slow and expensive baseline.

### 3.6 Foul Play / poke-engine (github.com/pmariglia/foul-play, GPL-3.0; poke-engine, MIT) **(verified, README and PokeAgent appendix)**
- **Search:** root-parallel MCTS with **decoupled UCT**, a handcrafted evaluation and no rollouts. It went from expectiminimax (~5-turn depth) to MCTS (10+ turns on main lines).
- **Engine:** poke-engine **groups damage rolls by KO relevance** and emits state *instructions* rather than copies.
- **Hidden information:** sets are sampled from usage statistics and inferred from damage, speed order, weather duration and so on.
- **Results:** randbats peak 2341 (~90% GXE); Gen 9 OU peak 1879 (~80% GXE).
- poke-engine is **"singles only"**, and foul-play is GPL-3.0 (do not copy code).

### 3.7 Other 2025–2026 work
Searched arXiv for pokémon/pokemon, showdown, VGC, "double battles".
- **LLM agents:**
  - Porimon (2609.32544): long/short-term knowledge-augmented LLM agent.
  - LLMs as Pokémon battle agents (2512.17308): custom simulator.
  - A multi-agent Pokémon tournament for LLM reasoning (2508.01623).
  - None of these is doubles RL.
- **Only VGC-Bench and PokaiTrainer address VGC/doubles RL.**
- The older **VGC AI Framework** (Reis et al., gitlab.com/DracoStriker/pokemon-vgc-engine; IEEE CoG VGC AI competition) uses a simplified custom engine, focuses on team building and balance, and doesn't run Showdown mechanics.

### 3.8 Synthesis: two regimes the engine should support (inference)

| | Jaxcalibur regime | PokaiTrainer regime |
|---|---|---|
| Training | Model-free PPO self-play, ~10⁸ games | Search at every self-play decision, ~10⁵ games |
| Where strength lives | The net (search +100–150 Elo) | The search (policy alone loses to greedy) |
| Engine needs | Raw batched step throughput, fused with policy in one `scan` | Clone or set state on hypothesised states, chance enumeration, belief updates by re-simulation |

- The two have never been compared at matched compute on VGC.
- Doubles' joint action space and wide chance make search harder (PokaiTrainer needed shortlists). They also make policy learning harder (VGC-Bench).
- **Design the API so both regimes work.** The pure functional design gives both almost for free. The extras are `determinize`, forced outcomes and optional enumeration.

---

## 4. Game-theory-aware RL for simultaneous-move, imperfect-information games

VGC is two-player zero-sum with simultaneous moves, imperfect information and chance. Turn-level rock-paper-scissors (Protect vs attack, switch vs punish) means equilibria are **mixed**. Team-preview and matchup cycles add a population-level intransitivity.

| Method | Core idea | Needs from the engine | Cost vs PPO self-play | Practical at Jaxcalibur scale? |
|---|---|---|---|---|
| **PPO self-play + entropy + KL-to-uniform** (Jaxcalibur) | Plain policy gradient, regularised against collapse | Symmetric per-player obs, both-act step | 1× | **Yes, the baseline** |
| **Magnetic Mirror Descent** (Sokota et al., ICLR 2023, arXiv 2206.05825) | Policy update with a KL to a "magnet" policy plus entropy. Converges to a QRE. Competitive with CFR in tabular games and works in deep RL (Dark Hex, Phantom TTT) | Same as PPO | ~1× (a loss change) | **Yes.** Swap the uniform magnet for a lagged policy |
| **Generic PG re-evaluated** (Rudolph et al., arXiv 2502.08938) | 7,000+ runs with exact exploitability on 5 large games: "FP-, DO-, and CFR-based approaches fail to outperform generic policy gradient methods" | — | — | **Evidence for regularised PPO first** |
| **R-NaD / DeepNash** (Perolat et al., arXiv 2206.15378) | Reward transformed by a log-ratio to a regularisation policy; iterate fixed points; NeuRD + V-trace. Expert Stratego, no search | Same as PPO; trainer keeps a reg-policy snapshot | ~1–1.5× | **Yes, as an ablation** (OpenSpiel has a JAX implementation). DeepNash's scale (≈1,024 TPU nodes per PokaiTrainer's citation) shows it isn't cheap to tune |
| **NFSP** (Heinrich & Silver, arXiv 1603.01121) | Best response via DQN plus an average policy via supervised learning on a reservoir | Replay buffers; off-policy | High; slow convergence | **No.** Superseded |
| **PSRO / double oracle / FP** (Lanctot et al., arXiv 1711.00832; XDO 2103.06426; Pipeline PSRO 2006.08555; Anytime PSRO 2201.07700; Self-Play PSRO 2207.06541) | A population of policies, each new one a best response to a meta-Nash mixture | **Per-env opponent identity** (several parameter sets in one batch), frozen opponents, cross-play payoff estimation | Each iteration is a full RL run (days at this scale) | **Limited.** A handful of iterations at most. VGC-Bench's FP/DO didn't beat SP |
| **NeuPL / Simplex-NeuPL** (arXiv 2202.07415, 2205.15879) | One network conditioned on a policy index represents the whole population; the meta-solver picks the opponent mixture | **Per-env policy-index input** (no parameter gathering, one forward pass) | ~1.2–2× | **Promising.** The most JAX-friendly population method |
| **League / PFSP** (AlphaStar; OpenAI Five's ~80/20 latest/past sampling) | Main agent plus past snapshots, sampled by win rate; main and league exploiters | Env batch partitioned into opponent groups; **snapshot params stacked on device** (8.5M bf16 ≈ 17 MB each); stats keyed by opponent ID | ~1.1–1.3× (extra inference only) | **Yes.** Recommended from day one |
| **SM-MCTS** (DUCT / regret matching / Exp3) | Tree search at simultaneous nodes | Clone, step with seeds, hashing | Test-time only | **Yes at test time, with regret matching.** DUCT is exploitable: PokaiTrainer's informed BR wins 83.6% against a DUCT mix and 49.6% against a CFR mix (also Shafiei et al. 2009) |
| **ISMCTS / determinisation** | Sample hidden information, search perfect-information worlds | `determinize`, `sample_hidden` | Test-time | Yes. Strategy fusion is minor under OTS, where hidden information is small |
| **ReBeL / Student of Games / PokaiTrainer** (arXiv 2007.13544, 2112.03178, 2608.29197) | CFR over public belief states with a counterfactual value network; search in training too | Belief-state machinery, enumeration or many samples, set-state on hypotheses | ~10³× per decision; few games | **A different regime (§3.8).** A JAX engine could batch the matrix-game CFR on GPU |

**Exploitability and evaluation tools:**
- Approximate best response (ABR), Timbers et al. (arXiv 2004.09677): train a best-response policy with the same RL pipeline against a frozen agent.
- Informed local best response (Lisý & Bowling 2017), used by PokaiTrainer.
- α-Rank (arXiv 1903.01373) and Nash averaging (arXiv 1806.02643) over cross-play matrices.
- Engine needs: a frozen-opponent mode, mixed-policy batches, and seat swapping.

**Recommendation (practical order):**
1. PPO self-play with entropy + forward-KL-to-uniform (Jaxcalibur). Optionally use an MMD magnet (a lagged policy).
2. From the start, add a historical opponent pool (~20% of envs against past snapshots, PFSP-weighted) and periodic exploiter agents as an exploitability probe.
3. R-NaD as an ablation if cycling appears.
4. NeuPL-style conditional populations if team or strategy cycles persist.
5. Test-time search: Jaxcalibur-style pUCT with regret matching, or a PokaiTrainer-style per-turn CFR matrix solve with value-network leaves, plus a 90×90 preview solve.

**Not practical at this scale:** full PSRO with independent best responses per iteration, and NFSP.

**Engine features implied:** per-env opponent/policy ID passed through to stats; seat randomisation; frozen-opponent evaluation; privileged observations; clone, set-hidden and hash for search.

---

## 5. Team preview and team diversity

### 5.1 Modelling the bring-4 decision
- **Size of the space:**
  - C(6,4)·C(4,2) = **90** gameplay-distinct (bring, leads) options per side.
  - ×2 for left/right lead order = 180.
  - The back pair's order only matters for Illusion, which is out of the v1 dex.
  - PokaiTrainer solves preview as exactly a 90×90 matrix game.
- **Options:**
  - **(a) A flat 90-way head with structured logits (recommended).**
    - logit(L,B) = Σᵢ∈L ℓᵢ + Σⱼ∈B bⱼ + ⟨u_L, v_L⟩ + ⟨u_B, v_B⟩ + ⟨p_L, q_B⟩, from per-mon tokens and pair embeddings.
    - Exact softmax and exact log-prob for PPO. Trivial to mask.
    - Keep left/right canonical (sorted) in the policy. The engine accepts 90 indices and maps to a PS `team` order.
  - **(b) Autoregressive** (pick 4, then leads, or 4 sequential picks). Same expressiveness, with sequential decoding. No engine change if the engine still takes one 90-index action.
  - **(c) VGC-Bench's two joint "switch-in" steps.** Reuses the battle head but adds env steps. Not recommended.
- **Learning signal.**
  - Preview is a one-shot simultaneous normal-form game with a mixed equilibrium (lead rock-paper-scissors). The PPO signal is weak and high variance.
  - Keep entropy and KL-to-uniform here, and a separate value read at the preview state.
  - The opponent-preview auxiliary label (their 90-index) gives a supervised belief signal.
- **Test-time preview solve (inference).**
  - For each of the 90×90 pairs, `apply_preview(state, a0, a1)`, then evaluate the value network (averaged over a few sampled SP worlds). That is 8,100 cheap evaluations.
  - Solve the matrix game with regret matching. Engine requirement: `apply_preview` vmappable over 8,100 pairs.

### 5.2 Training over a team distribution
- **Teams are part of the observation by construction.** OTS shows both full sheets: species, item, ability, moves, nature, gender, level.
  - The policy is automatically **team-conditioned**.
  - No team-ID feature should be given (it invites memorisation).
- **Sources for the team bank:**
  - VGCPastes (PokaiTrainer used 663 M-B teams);
  - PS Bo3 replays, whose `|showteam|` gives full sets *except SPs*;
  - usage-stat spreads to fill SPs;
  - poke-env 0.16's Smogon-stats teambuilder.
- **Spread realism:**
  - If each bank team carries one fixed spread, opponents learn "sheet → spread" lookups.
  - Instead, store K≈3–4 candidate spreads per mon with prior weights, and sample at `reset`. This is PokaiTrainer's menu: the top corpus clusters conditioned on nature, plus 15% base-stat archetypes.
  - The same table is the belief prior for auxiliary heads and determinisation.
- **Curriculum:**
  1. start from a few teams (VGC-Bench's 1-team agents were strongest per team);
  2. grow the bank;
  3. sample matchups with **prioritised weights** (PFSP over team pairs: oversample pairs with low or high win rates);
  4. hold out a team split for the generalisation test.
- **Mirror matches** on/off as an option.
- **Bo3 series.** The Bo3 ladder and tournaments replay the same two teams. "The four an agent brings is the first thing opponents read in a best-of-three" (PokaiTrainer).
  - Optional **series mode**: K consecutive games with fixed teams. The obs carries a public series summary (previous brings, leads, spread evidence).
  - PokaiTrainer carries spread posteriors across games.

### 5.3 Implications for `reset()`
- Teams are **data, not code**. Fixed-shape arrays: `[2, 6, fields]` plus SP candidates `[2, 6, K, 6]`.
  - A batch of 64k battles can hold 64k different team pairs with no retracing.
  - This works because the all-branches engine branches only on data.
- **The team bank lives on device.** `reset` gathers by index inside jit: `jax.random.categorical(key, pair_logits)` over a flattened N×N matchup table, or two independent draws.
- **The bank can be swapped between iterations** without recompiling (same shape, padded N).
- **Validation outside jit:** every bank team is checked against the supported-dex mask and PS's `TeamValidator` (Species Clause, Item Clause, SP ≤32 per stat and ≤66 total).
- **Inside jit:** an "unsupported effect" error flag catches anything that slips through.
- **Options:** `skip_preview` (random or heuristic preview), `no_mirror`, `series_len`.

---

## 6. Imperfect information under open team sheets

### 6.1 What is hidden in `[Gen 9 Champions] VGC 2026 Reg M-C` on PS (verified in source)
- **Formats** (`config/formats.ts`):
  - Bo1: ruleset `['Flat Rules', 'VGC Timer', 'Open Team Sheets']`;
  - Bo3: `['Flat Rules', 'VGC Timer', 'Force Open Team Sheets', 'Best of = 3']`.
- **`Open Team Sheets` is opt-in**: both players must click "Accept". `Force Open Team Sheets` calls `showOpenTeamSheets()` at preview.
  - **Measured:** `|showteam|` appears in **61/400 Bo1** and **400/400 Bo3** Reg M-C replays.
- **`showOpenTeamSheets()`** (`sim/battle.ts`) reveals species, item, ability, moves, gender, level, and **`nature: this.format.mod.startsWith('champions') ? set.nature : ''`**. It sets **`evs: null, ivs: null`**.
  - In Champions, IVs must all be 31 (validator), so **the only hidden build information is the SP spread** (0–32 per stat, 66 total).

| Information | Opponent sees? | Notes |
|---|---|---|
| Species, item, ability, moves, gender, level | **Yes** at preview (OTS) | `|showteam|` |
| Nature | **Yes** (Champions only) | `showOpenTeamSheets` |
| SP spread (and so exact stats, speed, max HP) | **No** | Inferable from damage, turn order and HP% over time |
| Brought 4 / unrevealed back 2 | **No** until switched in | `|teamsize|p1|4` reveals only the count. The two left at home are never revealed |
| Leads | Revealed at battle start | |
| Exact HP | **No**: Champions percentage rule (below) | Own HP exact |
| Boosts, status, most volatiles, field, side conditions | Yes, via protocol messages | |
| Sleep length, confusion length | No (random hidden counters) | Turns elapsed are public |
| PP | Derivable from observed uses (Pressure aside); PP capped at 20 in Champions | |
| Choice lock, Protect streak, Fake Out eligibility | Derivable | |
| Opponent's current-turn action | No (simultaneous) | |

### 6.2 How PS reports HP in Champions (verified, `sim/pokemon.ts` `getHealth`)
```ts
} else if (this.battle.dex.currentMod.startsWith('champions')) {
    // Pokemon Champions mechanics
    const percentage = Math.floor(100 * this.hp / this.maxhp) || 1;
    shared = `${percentage}/100`;
    if (percentage === 20) shared += this.hp * 5 > this.maxhp ? 'y' : 'r';
    else if (percentage === 50) shared += this.hp * 2 > this.maxhp ? 'g' : 'y';
} else if (this.battle.reportPercentages || this.battle.gen >= 7) {  // ceil-based HP Percentage Mod
```
- Opponents see **floor percent (minimum 1) out of 100**. At exactly 20 and 50 a colour suffix reveals which side of the colour threshold the true value is on. Players see their own `hp/maxhp` exactly ("secret" channel).
- This differs from the generic Gen 7+ rule (ceil, with 100 shown as 99 when not full).
- **The engine's observation must reproduce this exactly**, including the colour bit. Otherwise train-time and deploy-time observations diverge.
- Combined with known base stats, known nature and a max-HP SP range of 0–32, HP% observations progressively pin down the HP investment.

### 6.3 Asymmetric critic vs belief model
- **Asymmetric (privileged) critic** (Pinto et al., arXiv 1710.06542):
  - The critic sees hidden information: the opponent's SPs, brought set and exact HP.
  - This lowers variance, but a state-only critic V(s) is biased under partial observability. Baisero & Amato (arXiv 2105.11674) recommend a history-state critic **V(h, s)**: the actor's obs plus privileged extras.
  - The privileged information must not reach the actor, so a separate critic trunk or a separate forward pass is needed. That costs about +30–100% training FLOPs.
  - A related trick is Suphx's "oracle guiding" (arXiv 2003.13590): train with privileged features and decay them away.
- **Belief model via auxiliary heads** (Jaxcalibur; PokaiTrainer's belief inputs):
  - Predict each opponent mon's SP spread (bucketed or candidate index), its brought status and its next action from the public history.
  - Cheap: shared trunk, no extra forward.
  - Improves representations ("opp next action substantially stronger"; PokaiTrainer's belief summary +8–12 pp).
  - Directly supplies the determinisation sampler for search.
- **Recommendation:**
  - Start with **belief auxiliary heads plus a non-privileged value head** on the shared trunk. Under OTS the hidden gap is small.
  - Keep a privileged V(h,s) critic as an ablation. It matters most early in the game, when the opponent's back 2 is unknown.
  - The engine must support both.

### 6.4 What the engine must expose
1. **`observe(state, player)`**, computed only from (a) public state, (b) the player's own private state and (c) per-player **knowledge bits**.
   - Knowledge bits: revealed/brought flags, item consumed/removed/changed, ability changed, Mega forme.
   - It must never read opponent `sp`, `stats`, `hp` (except through the PS HP rule), unrevealed `brought`, or the RNG key.
2. **A public event ring buffer** in the state (e.g. the last 32 events: actor, move, targets, per-target HP% change, crit, effectiveness, miss/protect/immune, status, faint, switch-in species, item or ability reveal). The obs exposes the last 16, as in Jaxcalibur.
   - Without it the observation is not a sufficient statistic of the information state.
3. **`observe_privileged(state, player)`**: the same pytree, with hidden fields filled in and "known" masks set. For critics and diagnostics.
4. **`hidden_targets(state, player)`**: the opponent's true SP (per stat and candidate index), brought mask, preview action and exact HP.
5. **`infostate(state, player)`** and **`determinize(info, hidden)`**, with the round-trip property `infostate(determinize(infostate(s), h)) == infostate(s)` for any consistent h.
6. **`damage_calc`** on views plus hypotheses, callable from the model. Plus a convenience `calc_features(info, prior)` that uses own exact stats and opponent prior spreads.

### 6.5 Leak tests (CI)
- **Hidden-perturbation invariance** (minimon's `test_hidden_info.py` pattern, verified):
  - resample the opponent's SP spread within its candidate set;
  - swap their unrevealed back mons;
  - change exact HP within the same reported percentage;
  - change the RNG key.
  - Then assert `observe(state, 0)` is bitwise identical. The same perturbations on our side must change the observation.
- **Perspective symmetry:** `observe(state, 1) == observe(swap_sides(state), 0)`.
- **Protocol round-trip:**
  1. play engine games;
  2. export them through PS re-simulation (§7.10);
  3. parse the PS protocol with the deployment translation layer;
  4. assert the reconstructed obs equals the engine obs at every decision.
  - This catches both leaks and deploy-time drift.

---

## 7. The engine API the trainer needs

### 7.1 Principles
- **Pure, fixed-shape, single-battle functions; vmap outside.** Nested vmaps work for search: battles × worlds × rollouts.
- **Struct-of-arrays pytrees**, mostly int8/int16 (scope §3.2: ~2 KB per battle).
- **Everything team- or format-specific is data:** a `Dex` table pytree and a `TeamBank`. Statics (`EngineConfig`) are hashable.
- Follow the precedents:
  - pgx `core.step`: `lax.cond` no-op on terminal states; mask all-ones at terminal.
  - JaxMARL / minimon: simultaneous actions `i32[2]`, auto-reset by `select`.
  - minimon: no-op legal only when nothing to decide; illegal → first legal.

### 7.2 Core signatures
```python
# ---- static config (hashable; static arg to jit) ----
@dataclass(frozen=True)
class EngineConfig:
    ruleset: Ruleset = CHAMPIONS_REG_MC   # mechanics constants (Champions vs SV, PP cap, para 1/8, ...)
    team_sheets: Literal["open"] = "open" # v1; obs built from reveal bits so "closed" can be added later
    max_turns: int = 40                   # truncation (measured p99 15–19, max seen 59)
    turn_cap_outcome: Literal["draw", "vgc_tiebreak"] = "vgc_tiebreak"
    auto_resolve_forced: bool = True      # skip decisions with exactly one legal joint action
    event_history: int = 16
    trace: bool = False                   # emit chance + event trace for PS re-simulation (slower)
    skip_preview: bool = False            # VGC-Bench toggle: random preview

# ---- device data ----
class Dex(PyTree): ...                    # dense tables for the supported dex only
class TeamBank(PyTree):                   # N validated teams
    species, item, ability, nature, gender, mega_stone: i16[N, 6]
    moves: i16[N, 6, 4]
    sp_cand: i8[N, 6, K, 6]               # candidate SP spreads (K ≈ 3–4)
    sp_logit: f32[N, 6, K]                # prior; also the belief prior
class TeamPair(PyTree): ...               # concrete [2, 6, ...] incl. realised SPs

# ---- core (single battle) ----
def sample_teams(key, bank: TeamBank, pair_logits: f32[N*N] | None) -> tuple[TeamPair, i32[2]]
def init(key, teams: TeamPair, cfg, dex) -> State                       # phase = PREVIEW
def step(state, actions: i32[2], key, cfg, dex,
         forced: ForcedOutcomes | None = None) -> tuple[State, StepOut]  # runs to next decision point
def observe(state, player: i32, cfg, dex) -> Obs                        # info-state only, canonical frame
def legal(state, player: i32, cfg) -> Masks
def observe_privileged(state, player, cfg, dex) -> Obs
def hidden_targets(state, player) -> HiddenTargets

# ---- search / deployment ----
def infostate(state, player) -> InfoState
def determinize(info: InfoState, hidden: HiddenSample, cfg, dex) -> State   # also PS->engine builder
def sample_hidden(key, info, belief: SpreadBelief | None) -> HiddenSample  # None => bank prior
def hash_infostate(info, hp_buckets: int = 10, world_id: i32 | None = None) -> u32[2]
def apply_preview(state, a0: i32, a1: i32, key) -> State                # vmappable over 90x90
def damage_calc(att: MonView, dfn: MonView, move: i16, field: FieldView,
                hyp: SpreadHyp) -> DmgDist                              # model-callable; no State
def enumerate_outcomes(state, actions, max_branches) -> (States[M], probs[M])  # optional, via `forced`

# ---- interop (host side, table-driven) ----
def to_ps_choice(action, state, player) -> str          # "move 1 2 mega, switch 3" / "team 1234"
def ps_request_to_masks(request_json) -> Masks          # for mask-parity tests and deployment
```

`StepOut` (per battle):
```python
class StepOut(NamedTuple):
    obs: Obs            # stacked for both players, leading axis 2
    masks: Masks        # stacked [2, ...]
    reward: f32[2]      # zero-sum terminal outcome (+1/-1/0); 0 on non-terminal steps
    terminated: bool
    truncated: bool     # turn cap
    has_decision: bool[2]
    phase: i8           # PREVIEW | MOVE | FORCE_SWITCH | MIDTURN_SWITCH | DONE
    shaping: f32[2, F]  # potential features: HP-fraction sum, mons remaining, ... (NOT in reward)
    events: EventRec[E] # public events emitted this step
    error_flags: u32
    trace: ChanceTrace | None
```

### 7.3 Phases, "both act every step", and one-sided decisions
- **One env step = one simultaneous decision point.** `step` resolves until the next point where at least one player has more than one legal joint action, or until the game ends.
- **Phases:**
  - `PREVIEW` (both);
  - `MOVE` (both);
  - `FORCE_SWITCH` (end-of-turn replacements; both sides choose blind and simultaneously; either may be pass-only);
  - `MIDTURN_SWITCH` (U-turn, Volt Switch, Parting Shot, Eject Button/Pack, Emergency Exit; one side only).
  - The action queue and cursor live in the state so a turn can resume (scope §3.1).
- **One-sided steps:** the waiting player gets a **pass-only mask** and `has_decision=False`.
  - Trainer: drop these samples from the policy and entropy losses but keep them for value and auxiliary losses.
  - GAE over all steps distorts λ-weighting only slightly. Measured mid-turn switches are ~0.5 per game.
- **Auto-resolve:** with `auto_resolve_forced=True`, decisions with exactly one legal option (e.g. one bench mon left) are taken internally.
  - PS still sends a request in that case, so keep a parity mode that doesn't auto-resolve.
- Why not sequentialise like pgx's `current_player`? Simultaneity is the game. Sequentialising leaks the first mover's choice, or forces fake hidden-information bookkeeping.
- The "both act" layout also gives aligned `[T, B, 2]` trajectories, so GAE vectorises trivially.

### 7.4 Action encoding and masks
- **Preview:** `i32 ∈ [0, 90)` → (lead pair, back pair) via a static table. The mask is all-true. The engine picks the left/right order canonically.
- **Per slot (29 actions):**
  - `0–23`: move m ∈ {0..3} × target t ∈ {foe-left, foe-right, ally} × Mega ∈ {0,1}.
    - Targets are observer-relative, so mirroring for opponent-action labels is the identity on t.
    - Spread, self and side moves use t=0 only. "Normal" moves may target the ally (Pollen Puff, Skill Swap, Instruct).
  - `24–27`: switch to brought-roster position 0–3. **Identity-stable**: only benched, non-fainted mons are legal.
  - `28`: pass.
  - This differs from the scope doc's 27. The extra 2 dead actions buy pointer-style switch logits tied to mon tokens. They also avoid PS's position reshuffling: `switch N` refers to *current* team order, which changes as mons switch, so map it in `to_ps_choice`.
- **Joint (841 = 29²):**
  - `joint_mask = outer(slot_mask[0], slot_mask[1]) & pair_ok`, where `pair_ok: bool[29,29]` is state-dependent.
  - `pair_ok` forbids: both Mega; both switching to the same mon; non-pass in an empty or Commander slot; Struggle and locked moves collapsing to one option.
  - The engine returns `slot_mask`, `pair_ok` and the materialised `joint_mask`. A packed form saves memory.
- **The recommended head is joint**, with low-rank interaction: logit[i,j] = aᵢ + bⱼ + ⟨uᵢ, vⱼ⟩ (scope §5). PokaiTrainer's 64%-joint vs 97%-marginal coverage gap is direct evidence that slot correlations matter.
- **Illegal actions:** replace with the first legal action and set an error bit (minimon's convention). Never terminate with a penalty (pgx's convention), because that silently shapes the policy if masks are buggy.

### 7.5 Observation pytree (per player, canonical frame: index 0 = self)
Mostly int16, ≤ ~4 KB per player (inference). Tokenisation and embedding happen in the model.
- `meta`: phase, request kind, requesting slot(s), turn, turns to cap.
- `field`: weather and terrain id + turns elapsed and remaining (public, since items are known under OTS), Trick Room/Gravity turns.
- `side[2]`: Tailwind, screens (+turns), this-turn Wide/Quick Guard, `mega_used`, remaining and fainted counts.
- `mon[2, 6]`:
  - **sheet** (both sides): species, item, ability, nature, moves[4], gender, Mega Stone.
  - **own-private** (side 0; zero + mask for side 1): stats[6], SP[6], exact HP, max HP, PP.
  - **public:** PS-rule HP% + colour bit, status + public counters, boosts[7], public volatile bits/counters, active slot, `brought ∈ {unknown, yes, no}`, fainted, item and ability state, current forme, observed PP use, last move, choice lock, Protect streak, turns active.
- `events[16]`: public event records (§6.4).
- `masks`: §7.4.
- Calc features are *not* baked into `observe`. The model calls `damage_calc` on hypotheses (Jaxcalibur-style mid-network), or the trainer calls `calc_features(info, prior)`.

### 7.6 Reward, termination, truncation
- **Reward:** zero-sum terminal outcome `f32[2]` (+1/−1, or 1/0 via flag). **Default win/loss only** (Jaxcalibur; VGC-Bench `calc_reward`).
- **Shaping:** the engine exposes `shaping` *features* only. The trainer may apply **potential-based** shaping F = Φ(s′) − Φ(s) with Φ(terminal) = 0, so optimal policies are preserved at γ=1, and anneal it to 0.
  - Avoid ad-hoc shaping. Naive shaping is exploited (Pleines et al., cited in VGC-Bench).
- **Simultaneous last-mon KOs** follow PS's winner rule (and Champions' coded rule where it differs).
- **Turn cap:**
  - Measured p99 is 15–19 turns, max seen 59. Early random self-play runs longer.
  - Cap at about 40 turns.
  - Outcome: VGC-style tie-break (Pokémon remaining, then total HP%, otherwise a draw). Official play uses a 20-minute game clock (Play! Pokémon VGC Handbook §4.3.1: 90 s preview, 45 s per move, 7 min "Your Time", 20 min game, verified), but **verify Champions' exact timeout rule**.
  - Mark `truncated=True` and provide `final_obs` so trainers can bootstrap instead.
  - PS's Champions `flatrules` has no Endless Battle Clause (verified), so the cap is engine-only.

### 7.7 Auto-reset, episode statistics, team sampling
```python
class AutoResetEnv:   # wrapper; all inside jit
    def reset(key, sampler: SamplerState) -> (EnvState, StepOut)
    def step(env_state, actions: i32[2], key) -> (EnvState, StepOut, EpisodeStats)
    # on done: sample_teams(key, bank, sampler.pair_logits) -> init -> observe; StepOut.final_obs = pre-reset obs
class EpisodeStats(NamedTuple):   # valid where done
    winner: i8; turns: i16; steps: i16; end_reason: i8   # KO | turn_cap | error
    team_ids: i32[2]; opp_policy_id: i32; preview_actions: i32[2]
    megas: i8[2]; remaining: i8[2]; hp_frac: f32[2]; error_flags: u32
```
- Select-based auto-reset (JaxMARL) computes a reset every step. That is fine because `init` is a gather plus stat computation.
- If profiling shows otherwise, use Craftax-style "optimistic resets" (arXiv 2402.16801), computing fewer resets than envs.
- **Matchup tables:** the trainer scatter-adds `EpisodeStats` into an `[N_teams, N_teams]` win/count table, and into `[N_policies]` for PFSP. Both are pure jit ops. The new `pair_logits` feed back into `reset`.
- **Seat randomisation:** the learner plays side 0 or 1 at random. Test that the engine's semantics are side-symmetric (speed ties, simultaneous switch-ins).

### 7.8 Opponent pools
- The engine is agnostic: it just takes `actions[2]`. Two requirements on the API:
  1. `opp_policy_id` passes through to stats.
  2. Obs is canonicalised per player, so any policy can drive either seat.
- **Trainer layout (inference):**
  - Partition the batch into G contiguous groups, each against one snapshot from stacked on-device params. Run one forward pass per group.
  - Or use NeuPL-style conditioning on a policy-index embedding: one forward pass, no parameter gather.

### 7.9 Search support

| Need | API |
|---|---|
| Clone | Free (immutable pytrees). `jax.tree.map(lambda x: jnp.broadcast_to(x, (N,)+x.shape), s)` |
| Per-rollout seeds and common random numbers | `step(..., key)`. Draws are `fold_in(key, turn, actor, event)` (scope §3.1), so two different actions under the same key share crit and roll draws. This variance reduction is useful when comparing actions in search |
| Determinised worlds | `sample_hidden` + `determinize`, on top of the same SP candidate tables as the bank |
| Transposition and statistics | `hash_infostate(info, hp_buckets=10, world_id)` |
| Preview solve | `apply_preview` vmapped over 90×90 |
| Exact expectations (PokaiTrainer-style) | `forced` per-event overrides. Optional `enumerate_outcomes` over KO-relevant roll buckets × hit/crit (poke-engine-style grouping). A stretch goal |
| Belief update by re-simulation | Re-run the observed joint action per candidate spread with `forced` outcomes matching the observed public events, and score the likelihood. Needs a `public_outcome_match(state', observed_events)` helper |

### 7.10 Logging, replay export, PS interop
1. **Compact public events** per step (`StepOut.events`), stacked by `scan`. These feed observation history and dashboards.
2. **Trace mode** (`cfg.trace=True`): records chance outcomes per (turn, actor, target, event ID) and the joint actions.
3. **Replay export.** A host-side tool converts (teams with SPs, actions, chance trace) into a PS input log:
   - `>start`, `>player`, `>p1 team 1234`, `>p1 move 1 2 mega, switch 3`.
   - It runs that through a pinned PS with the instrumented PRNG from scope §6a, forced to the same outcomes.
   - The result is a **genuine PS protocol log and HTML replay**, viewable in the standard replay viewer.
   - Any divergence between PS and engine state is a fidelity bug, so **replay export doubles as differential testing**. No separate engine-native protocol emitter is needed, so there is no second source of truth.
4. **Mask parity:** compare the engine `legal()` with PS request JSON at every decision during differential tests.
5. **Deterministic replay:** (teams, initial key, actions) fully determines a game. Use it for bug reports.

### 7.11 Error handling
- A `uint32 error_flags` bitfield in the state: illegal action substituted, unsupported effect ID, queue overflow, negative HP, NaN, invariant failure. It is surfaced in `EpisodeStats` and dashboards.
- Debug builds use `jax.experimental.checkify`.

### 7.12 Requirements checklist
- **MUST:**
  - pure single-battle `init`/`step`/`observe`/`legal`, vmappable;
  - simultaneous `actions[2]`, with pass masks and `has_decision`;
  - preview/move/force-switch/mid-turn phases;
  - exact joint mask;
  - information-state-only obs with the PS Champions HP rule;
  - canonical perspective;
  - event history;
  - zero-sum terminal reward;
  - turn cap with a truncation flag;
  - auto-reset with in-jit team sampling from a device bank, including per-mon SP sampling;
  - episode statistics with team and opponent IDs;
  - error flags;
  - action ↔ PS choice mapping;
  - trace mode for PS re-simulation;
  - leak and symmetry property tests.
- **SHOULD:**
  - `observe_privileged`, `hidden_targets`;
  - `infostate`/`determinize`/`sample_hidden`;
  - `hash_infostate`;
  - model-callable `damage_calc`;
  - `apply_preview`;
  - forced outcomes;
  - shaping features;
  - `skip_preview`/`no_mirror` options;
  - seat-symmetry tests;
  - ≥10⁵ env-steps/s.
- **MAY:**
  - `enumerate_outcomes`;
  - belief update by re-simulation;
  - series (Bo3) mode;
  - a closed-team-sheet obs mode (only 15% of Bo1 games use OTS);
  - optimistic resets.

---

## 8. Throughput requirements

### 8.1 VGC game length (measured, 2026-10-21)
- Data: the 400 most recent uploaded replays each for `gen9championsvgc2026regmc` and `gen9championsvgc2026regmcbo3`, fetched via `replay.pokemonshowdown.com/search.json` and `/<id>.log`.

| | Bo1 Reg M-C | Bo3 Reg M-C (per game) |
|---|---|---|
| Games (with ≥1 turn) | 400 (395) | 400 (398) |
| Forfeits / timer losses | 110 / 13 | 150 / 5 |
| Turns, all games: mean / median / p90 / p99 / max | 7.42 / 7 / 11 / 18 / 59 | 7.18 / 7 / 11 / 16 / 19 |
| Turns, no forfeit: mean / median / p90 / p99 | **8.08 / 7 / 11 / 19** | **7.81 / 7 / 12 / 15** |
| End-of-turn replacement phases per game | 2.91 | 2.91 |
| Mid-turn switch events per game | 0.49 | 0.57 |
| Faints per game | 5.84 | 5.76 |
| Megas per game | 1.71 | 1.78 |
| **Env steps per game** (preview + turns + replacement + mid-turn) | **12.5** (median 12, p99 24) | **12.3** (median 12, p99 22) |

- Rated games ≥1300 (n=45, Bo1): 8.9 turns, 13.3 steps.
- Caveats: uploaded replays are a biased sample (low median rating ≈ 1100–1170), and forfeits shorten games.
- Self-play has no forfeits, so use **~8 turns and ~12.5 env steps per game**.
- Cross-check: VGC-Bench's Champions data averages 8.3 transitions per player-trajectory.
- Real decisions per player per game ≈ 10.5. About 85% of the 2×12.5 per-step samples carry a real decision.

### 8.2 The Jaxcalibur baseline
- 100M games/week = **165 games/s**. Singles randbats needs ~30 decisions per player (inference), so ≈ **5k env-steps/s and ≈ 10k policy samples/s** (2 players). That is ≈ 6×10⁹ samples/week, end to end, on one H100 including learning.

### 8.3 Network cost model (inference)
- Forward FLOPs ≈ 2·N·T. Jaxcalibur's position-MoE touches fewer parameters per token, so this is an upper bound.
- PPO update with E epochs ≈ 3E × forward. With E = 2, total ≈ 7× forward per sample.
- **VGC tokens:**
  - field, summary;
  - 6 + 6 mons, 4 actives;
  - 8 candidate moves;
  - 16 events;
  - 16 matchup tokens (4 own brought × 4 opponent);
  - total **T ≈ 90–110**.

| Model | Forward / sample | Train / sample (E=2) | Ceiling on 1 H100 at 20–30% MFU (200–300 TFLOP/s) |
|---|---|---|---|
| 8.5M, T≈100 | ≈1.7 GFLOP | ≈12 GFLOP | ≈17–25k samples/s |
| 30M, T≈100 | ≈6 GFLOP | ≈42 GFLOP | ≈5–7k samples/s |

### 8.4 Targets

| Configuration | Samples/s (both players) | Env-steps/s (pipeline) | Games/s, games/week | Network share of 1 GPU |
|---|---|---|---|---|
| 1×H100, 8.5M, Jaxcalibur-equivalent samples | 10k | **5k** | 400; **240M** | ~40–60% |
| 1×H100, 8.5M, Jaxcalibur-equivalent *games* (100M/wk) | 4k | 2.1k | 165; 100M | ~20% |
| 1×H100, 30M | ~3–4k | ~1.7k | ~140; 85M | ~70–80% |
| 8×H100, 30M | ~25k | ~12k (1.5k/GPU) | ~1k; 600M | ~70–80% |
| 8×H100, 8.5M | ~80k | ~40k (5k/GPU) | ~3.2k; 1.9B | ~40–60% |

- **The binding case is a small model on one GPU** (rows 1 and 5): the engine must deliver ~5k env-steps/s per GPU while using ≤ ~20–30% of the GPU.
  - So the **standalone engine needs ≥ ~2–3×10⁴ env-steps/s per H100** at batch 16k–64k.
  - **Target 10⁵.** That keeps the engine ≤5–10% for the small model, and leaves room for search in training, smaller models, or a 2× slower engine as dex coverage grows.
- **Test-time search:** Jaxcalibur-like budgets are ~20k rollouts × ~2–3 env steps (2 turns are ~2–3 simultaneous steps).
  - That is ≈ 60k engine steps plus network calls within ~5–15 s per decision. PS allows ≤55 s per turn and 7 min total; the official game allows 45 s per move.
  - So **≥2×10⁴ env-steps/s at batch 1k–4k**.
- **Reference points:**

| System | Throughput |
|---|---|
| Showdown | ~2.0 ms per turn per Node process (PokaiTrainer) ≈ 500 turns/s/core |
| poke-env multi-instance | 681 SPS (PokeJAX paper) |
| PokaiEngine Compute mode (Rust) | 0.08 ms/turn ≈ 12.5k turns/s/core |
| PokeJAX | claimed 5×10⁸ SPS random-action at 65k batch |
| VGC-Bench | 5M timesteps in total for its RL runs |

- **The bar a JAX engine must clear (inference):** a Rust engine on ~10–40 CPU cores, with batched GPU inference, could also deliver ~10⁵ steps/s.
  - The JAX engine's advantage is fusing env and policy into one `lax.scan`, with no host↔device traffic (PureJaxRL). Its cost is all-branches waste and compile time.
  - Benchmark both early.

### 8.5 Memory and compile
- **State:** 64k battles × ~2 KB ≈ 128 MB.
- **Rollout storage dominates:** T=32 × B=16k × 2 players × ~4 KB obs ≈ 4 GB.
  - Keep the obs integer-coded. Store `slot_mask` + `pair_ok` rather than 841-bool joint masks, or recompute masks and obs from stored states during the update.
- PokeJAX (1,370 moves, full dex): 45 s JIT, 28 GB at 65k. **Our v1 dex** (~158 moves, ~65 abilities, ~60 items for 99% usage at ≥1760) should come in under that.
- Track compile time per hook in CI, with a budget of under ~2 min.

---

## 9. Deployment and evaluation

### 9.1 Playing on Showdown
- **Which format.** Use **`gen9championsvgc2026regmcbo3`** (Force OTS).
  - In Bo1, OTS needs both players to accept, which happened in only 15% of replays (measured). A policy trained only on OTS would face closed sheets 85% of the time.
  - Bo3 adds series dynamics. Either run games independently, or use the series-mode observation.
- **Client.** poke-env supports doubles (`DoubleBattle`, `DoublesEnv`). It added **Pokémon Champions data in 0.15.0 (2026-04-14)** and **"Better Nature, EV, and IV inference from OTS" (#920, 2026-05-31)**. The current version is 0.16.1 (2026-08-23) (verified via the GitHub API). VGC-Bench's `play.py` loads a policy onto the live server.
- **Translation layer (protocol → engine info state).**
  - Do **not** map poke-env's `DoubleBattle` object field by field. Write a protocol reducer that maintains exactly the engine's `InfoState` (the knowledge bits of §6.4).
  - Own side comes from the request JSON (exact stats, HP, PP, legal options). Opponent side comes from `|showteam|`, `|switch|`, `|-damage|` (floor% + colour), boosts, field messages and so on.
  - Then `observe(determinize(info, prior_sample), 0)` gives the policy input. The same `determinize` feeds search.
  - Validate with the protocol round-trip test (§6.5) and **mask parity against PS's request** on every live decision. Log mismatches and fall back to PS's legal set.
  - PokaiTrainer's architecture (a protocol adapter reconstructing "the same player-view BattleState the engine emits") is the template.
- **Unsupported content.** If the opponent's sheet contains effects outside the supported dex, play with a flagged degraded mode (treat the effect as no-op, log it). Do not forfeit.

### 9.2 Showdown bot policy and etiquette
- **Policy as of 2026-10-21** (Smogon "Initial policy on bots", thread 3789528, posted 2026-10-09 by Django; staff replies by chaos and Hecate; read via fetch):
  - Staff see bots as inevitable and want to work with developers.
  - Human-vs-human play is to be protected.
  - A **registry of self-declared bots**. **Running an unregistered bot is "a bannable offense."** Register via `/ht` → "Something else" → "I want to register my bot."
  - **Users can opt in to playing bots** (a client toggle), and **bot battles won't affect human ratings**.
  - **Using AI tools to assist during tournaments is ghosting → immediate ban.**
  - Details are still marked "to be confirmed". The thread doesn't address VGC specifically, research bots, or ladder advisor tools.
  - The `pokemonshowdown.com/rules` page has no bot clause (verified). It does prohibit "gam[ing] the system".
- **Precedents:**
  - Jaxcalibur withdrew from the ladder for integrity reasons.
  - PokaiTrainer laddered from fresh accounts (Aug 2026, before the registry), 150 sets, one at a time.
  - PokeAgent runs bots on a separate server.
- **Recommended etiquette:**
  1. **register the bot** and label the account as a bot;
  2. play only in the opt-in bot pool, or on a local or PokeAgent-style server;
  3. keep volumes low and play one game at a time;
  4. don't chase ladder #1 or suspect requirements;
  5. never use it in tournaments or for human-account assistance;
  6. withhold weights while a format is live (PokaiTrainer's and Jaxcalibur's stance), or release only after community consultation.

### 9.3 Using an AI in the actual Pokémon Champions game
- **Terms.** pokemon.com Terms of Use (last updated 2023-06-15; cover "all apps published by Pokémon", including console apps; read via fetch):
  - §5(iii) prohibits "Use, or facilitate the use of, any unauthorized third-party software (e.g. bots, mods, hacks, and scripts) to modify or automate operation within the Service".
  - Also prohibited: circumventing security (§5(x)), exploiting bugs (§5(xii)) and reverse engineering (§5(vi)).
  - I did not find Champions-specific terms. On Switch, Nintendo Account terms apply as well (not checked).
  - **Conclusion: automated play in Champions (input injection, screen-reading bots) is prohibited**, and accounts can be suspended.
- **Tournaments.** The Play! Pokémon Tournament Rules Handbook (verified PDF) §3.2.1 forbids, during matches:
  - "Receiving information from an outside source regarding their match";
  - "using a cellular device … for calculator functions";
  - "Prewritten information … both physical and digital".
  - Devices must be unmodified (VGC Handbook §3.2). **Any in-match AI advice is prohibited at sanctioned events.**
- **The "advisor" case** (human plays; AI suggests from manually entered or camera-read state):
  - It isn't "automating operation within the Service". The ToU text arguably doesn't reach a separate device that never touches the game.
  - But it is **outside assistance**. It is unambiguously banned in sanctioned play and on PS tournaments (ghosting), and contrary to PS's stated aim of protecting human ladder play.
  - Defensible uses: **post-game review**, **practice against the bot**, **team-building analysis** (self-play win-rate matrices, as in PokaiTrainer's Bradley–Terry team ratings), and teaching tools. Not live ranked assistance.
  - The project memory says the goal is "for use on Pokémon Champions". The realistic, ToS-compatible version is offline analysis and practice tools. Get a legal read before shipping anything that runs alongside live ranked play.

### 9.4 Recommended evaluation protocol
- **Tier 0: every N updates, in-engine, cheap.**
  - Opponents: random-legal; max-damage (JAX heuristic using `damage_calc` and no ally hits); a VGC heuristic (Protect when threatened, Fake Out turn 1, spread moves when both foes are hittable, speed control); a frozen BC policy.
  - **Duplicate matches:** same team pair and seed, seats swapped. Luck is tied to the seat through the counter-based RNG.
  - Held-out team pairs. 1,000+ games per cell (±3 pp).
- **Tier 1: population.**
  - A cross-play matrix over checkpoints, with **Elo/Bradley–Terry plus α-Rank / Nash averaging** to see through intransitivity.
  - Win rate against past snapshots, to catch forgetting and cycles.
- **Tier 2: exploitability proxies.**
  1. **ABR**: train a fresh PPO best response against the frozen agent for a fixed budget (e.g. 5–10% of the main run), report the BR win-rate curve (VGC-Bench §4.3).
  2. **Preview exploitability**: compute the 90×90 preview payoff (value net plus rollouts) for sampled team pairs, and measure the gap between the policy's preview distribution and the best response.
  3. Optionally an informed local best response with search (PokaiTrainer §E.5).
- **Tier 3: generalisation.** VGC-Bench's performance test (shared training teams) and generalisation test (disjoint teams with similarity statistics), across bank sizes.
- **Tier 4: sim-to-real.**
  - Play the same agent pair and teams in the engine and in a local PS through the translation layer. **The win-rate gap is a fidelity metric.**
  - Then external baselines on local PS: poke-env heuristics, VGC-Bench released checkpoints (M-A/M-B; Reg M-C support is pending), PokéChamp's VGC LLM player, and **PokaiTrainer** if released.
- **Tier 5: humans, sanctioned only.**
  - A registered bot on the PS opt-in bot pool (Bo3 Reg M-C), reporting GXE and the rating band, not the peak (PokaiTrainer's practice).
  - Consenting challenge matches with strong players.
  - No tournament or advisor use.

---

## 10. Open questions and risks
1. **Champions in-game vs PS mechanics.** PS is the oracle, and the Champions mod is new (scope §6a).
2. **Closed team sheets.** 85% of Bo1 games. v1 is OTS-only, but the observation design (reveal bits) should keep the closed-sheet extension cheap.
3. **Does 10⁸-game model-free PPO close the gap with search on VGC?** Unknown. VGC-Bench's evidence is sample-starved, and PokaiTrainer only compared against a 256k-game PPO. Budget for test-time search either way.
4. **Team-matchup intransitivity** may force population methods. Instrument cycles early via cross-play.
5. **Reg M-C BC data must be scraped.** VGC-Bench has none. Use Bo3 replays for OTS. Ratings in uploaded replays are low (median ~1100–1170 in my sample).
6. **Format churn.** Reg M-C will change. Keep `Ruleset`, `Dex` and `TeamBank` generated, not hand-written.
7. **Bot policy details are still "to be confirmed"** on PS. Re-check before any online play.

---

## References (all accessed 2026-10-21)
- Jaxcalibur: https://jaxcalibur.github.io ; thread https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/
- Smogon "Initial policy on bots": https://www.smogon.com/forums/threads/initial-policy-on-bots.3789528/
- VGC-Bench: arXiv 2506.10326 (v3); https://github.com/cameronangliss/vgc-bench ; https://huggingface.co/datasets/cameronangliss/vgc-battle-logs
- PokaiTrainer: arXiv 2608.29197 (v2, 28 Sep 2026)
- PokeJAX / automatic env generation: arXiv 2603.12145
- Metamon: arXiv 2504.04395 ; https://github.com/UT-Austin-RPL/metamon
- PokeAgent Challenge: arXiv 2603.15563 ; https://pokeagentchallenge.com
- PokéChamp: arXiv 2503.04094 ; https://github.com/sethkarten/pokechamp
- Foul Play (GPL-3.0): https://github.com/pmariglia/foul-play ; poke-engine (MIT, singles only): https://github.com/pmariglia/poke-engine
- Other Pokémon LLM work: arXiv 2609.32544, 2512.17308, 2508.01623 ; VGC AI Framework: https://gitlab.com/DracoStriker/pokemon-vgc-engine
- Pokémon Showdown source (master, 2026-10-21): `config/formats.ts`, `data/rulesets.ts`, `data/mods/champions/{rulesets,scripts}.ts`, `sim/battle.ts` (`showOpenTeamSheets`), `sim/pokemon.ts` (`getHealth`), `sim/team-validator.ts`, `sim/dex-formats.ts` (evLimit 66). https://github.com/smogon/pokemon-showdown
- PS replays API: https://replay.pokemonshowdown.com/search.json?format=gen9championsvgc2026regmc (and `…regmcbo3`)
- poke-env: https://github.com/hsahovic/poke-env (releases 0.15.0, 0.16.x)
- JAX env APIs: pgx arXiv 2303.17503 (`pgx/core.py`); JaxMARL arXiv 2311.10090 (`multi_agent_env.py`); minimon https://github.com/spktrm/minimon (`rollout.py`, `tests/test_hidden_info.py`); Craftax arXiv 2402.16801; Jumanji arXiv 2306.09884; PureJaxRL https://github.com/luchris429/purejaxrl
- Game theory and RL: MMD arXiv 2206.05825 ; policy gradients re-evaluated arXiv 2502.08938 ; R-NaD/DeepNash arXiv 2206.15378 ; NFSP arXiv 1603.01121 ; PSRO arXiv 1711.00832 ; XDO arXiv 2103.06426 ; Pipeline PSRO arXiv 2006.08555 ; Anytime PSRO arXiv 2201.07700 ; Self-Play PSRO arXiv 2207.06541 ; NeuPL arXiv 2202.07415 ; Simplex-NeuPL arXiv 2205.15879 ; ReBeL arXiv 2007.13544 ; Student of Games arXiv 2112.03178 ; Abstracting imperfect information away arXiv 2301.09159 ; ABR arXiv 2004.09677 ; α-Rank arXiv 1903.01373 ; Nash averaging arXiv 1806.02643
- Asymmetric and oracle critics: arXiv 1710.06542 ; arXiv 2105.11674 ; Suphx arXiv 2003.13590
- Play! Pokémon: VGC Tournament Handbook and Tournament Rules Handbook PDFs (https://play.pokemon.com/en-us/resources/documents/); pokemon.com Terms of Use https://www.pokemon.com/us/legal/terms-of-use
