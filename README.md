# vgc-jax

A faithful, pure-JAX battle engine for Pokémon Showdown's `[Gen 9 Champions] VGC 2026 Reg M-C`, validated by
differential testing against a pinned Pokémon Showdown (the oracle). Design: `docs/design/` (start at `00-README.md`).

## Setup
- Node 26.8.1 (`.nvmrc`), Python 3.12 (`.python-version`), `uv` 0.11.20.
- `make oracle` — init the PS submodule (`oracle/ps` @ d601443), `npm ci`, `node build`, oracle deps.
- `uv sync` — Python env with exact pins (ADR-024).
- `make hooks` — install the T0 pre-commit hook.

## Common commands
- `node oracle/bin/oracle.js gen --n 1000 --seed 1 --workers 12 --out traces/smoke/` — trace corpus (07 §3)
- `node oracle/tools/check_trace.js traces/smoke/` — draw/tie classification sanity check
- `node oracle/bin/oracle.js serve` — JSON-RPC oracle (02 §4.2)
- `node oracle/bin/oracle.js check-callsites` — RNG call-site drift alarm
- `node --test oracle/test/` — oracle tests; `make t0` / `make t1` — local CI tiers

Status: see `PROGRESS.md`.
