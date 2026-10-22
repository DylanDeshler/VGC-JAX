# CLAUDE.md — vgc-jax (faithful JAX engine for Pokémon Champions VGC Reg M-C)

Read `docs/design/11-agent-operating-manual.md` before doing anything. The design package is in `docs/design/`, its ADRs (`01-decisions.md`) are binding, and new decisions go in `docs/decisions/`.

## Essentials
- **Oracle.** Pokémon Showdown is pinned at `oracle/ps` (a submodule). It is the ground truth. **Never edit it.**
- **Tests.** Expectations always come from PS, via the oracle RPC or PS-generated goldens. Never hand-write expected values.
- **Data.** Content data is generated (`tools/datagen`). Never hand-type it.
- **Citations.** Every engine function and handler cites `PS: file:lines`.
- **JAX rules.** No Python control flow on tracers. Static loop bounds plus overflow flags. Integer math only. A barrier after each hook.
- **Status.** `PROGRESS.md` holds the current milestone, gate metrics and open divergence buckets. Read it at the start of a session and update it at the end.
- **Catalogue.** `docs/catalogue.csv` tracks per-entity status. Update it in the same commit as the entity. Fold nightly `reports/catalogue_status.json` in at session start.
- **Protected contract (ADR-029).** These files change only with an ADR and human approval:
  - `harness/contract/*`, `harness/gates.yaml`, `docs/coverage/waivers.json`, `docs/design/contract/drawkey.md`.
- **Target freeze (ADR-028).** No PS re-pins before G-v0.
- **Session start.** Read `PROGRESS.md` and `reports/LATEST.md`.

## Commands
- `make oracle`: build the pinned PS (02 §6)
- `make t0` / `make t1`: pre-commit / pre-merge tiers (`scripts/merge.sh` enforces T1)
- `uv run pytest -m "not slow"`: tier T1-fast
- `node oracle/bin/oracle.js gen --n 2000 --seed <S> --stage v0 --out traces/smoke/` (a directory of shards), then `uv run python -m harness.cli replay traces/smoke/`: fuzz smoke
- `uv run python -m harness.cli report <run>`: conformance summary and gate evaluation
- `node oracle/bin/oracle.js check-callsites`: RNG call-site drift alarm (run on every PS bump)
- `uv run python bench/compile_time.py --stage v0` / `bench/throughput.py`: performance budgets

## Do not
- Do not edit `oracle/ps`, `vgcjax/data/*` (generated) or `tests/fixtures/conformance/*` (generated).
- Do not copy code from Foul Play (GPL-3.0).
- Do not use PS's sequential RNG stream position as a key. Use semantic draw keys (`docs/design/contract/drawkey.md`).
- Do not start content for the next stage before the current gate passes.
- Do not weaken the conformance contract to pass a gate. Keep the canary test green.
- Do not commit `data/raw/`, virtualenvs, third-party clones or downloaded web pages.
