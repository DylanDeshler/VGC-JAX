# Local "CI" (02 §6). T0 = pre-commit, T1 = pre-merge, T2 = nightly, T3 = weekly.
SHELL := /bin/bash
PY := uv run python
PYTEST := uv run pytest

.PHONY: oracle t0 t1 t2 t3 gate bench prune hooks

oracle:
	git submodule update --init oracle/ps
	cd oracle/ps && npm ci --no-audit --no-fund && node build
	cd oracle/ps && npm ls --all > ../ps-deps.lock.txt 2>&1 || true
	cd oracle && npm install --no-audit --no-fund

t0:
	$(PY) scripts/lint.py
	$(PYTEST) tests/unit -m "not slow" -x

t1: t0
	node oracle/bin/oracle.js check-callsites
	$(PYTEST) -m "not slow" -n auto

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
