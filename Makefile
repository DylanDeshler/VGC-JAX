# Local "CI" (02 §6). T0 = pre-commit, T1 = pre-merge, T2 = nightly, T3 = weekly.
SHELL := /bin/bash
PY := uv run python
PYTEST := uv run pytest

.PHONY: oracle datagen t0 t1 t2 t3 gate bench prune hooks

oracle:
	git submodule update --init oracle/ps
	cd oracle/ps && npm ci --no-audit --no-fund && node build
	cd oracle/ps && npm ls --all > ../ps-deps.lock.txt 2>&1 || true
	cd oracle && npm install --no-audit --no-fund

# Content datagen (06 §1). Format is a parameter (ADR-028); chaos inputs and the real-sheet corpus are frozen (05 §1).
FORMAT ?= gen9championsvgc2026regmc
MOD ?= champions
CHAOS_DIR ?= data/raw/smogon-chaos/2026-09
CHAOS_BO3 ?= $(CHAOS_DIR)/$(FORMAT)bo3-1760.json
CHAOS_BO1 ?= $(CHAOS_DIR)/$(FORMAT)-0.json
REPLAYS ?= data/raw/ps-replays/2026-10/$(FORMAT)bo3_teams.jsonl
SHEETS ?= data/teams/real_sheets_mc_v1.jsonl

# The frozen real-sheet corpus is rebuilt only if its raw input exists; its sha256 is recorded next to it.
datagen:
	node tools/datagen/export_dex.js --mod $(MOD) --format $(FORMAT) --out data/export
	node tools/datagen/legal_pool.js --format $(FORMAT) --mod $(MOD) --out data/export/legal_pool.json
	node tools/datagen/census.js --mod $(MOD) --legal data/export/legal_pool.json --out data/export/handler_census.json
	if [ -f $(REPLAYS) ]; then $(PY) -I tools/datagen/teams/real_sheets.py --format $(FORMAT) --mod $(MOD) \
		--replays $(REPLAYS) --chaos $(CHAOS_BO3) --chaos $(CHAOS_BO1) --legal data/export/legal_pool.json --out $(SHEETS); fi
	$(PY) -I tools/datagen/dex_stages.py --format $(FORMAT) --legal data/export/legal_pool.json \
		--census data/export/handler_census.json --sheets $(SHEETS) --core-inert tools/datagen/core_inert.json \
		--chaos bo3_1760=$(CHAOS_BO3) --chaos bo1_0=$(CHAOS_BO1) --out-dir data/export/stages
	$(PY) tools/datagen/catalogue_stages.py --stages data/export/stages --catalogue docs/catalogue.csv

t0:
	$(PY) scripts/lint.py
	$(PYTEST) tests/unit -m "not slow" -x

t1: t0
	node oracle/bin/oracle.js check-callsites
	node --test oracle/test/
	$(PYTEST) -m "not slow" -n 4

t2:
	scripts/nightly.sh t2

t3:
	scripts/nightly.sh t3

bench:
	$(PY) bench/compile_time.py --stage core

prune:
	scripts/prune.sh

hooks:
	printf '#!/bin/sh\nexec make t0\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
