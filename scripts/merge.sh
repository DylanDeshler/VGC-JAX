#!/usr/bin/env bash
# Merge a branch into main only if T1 passes on it (02 §6). Usage: scripts/merge.sh <branch>
set -euo pipefail
branch=${1:?usage: scripts/merge.sh <branch>}
git checkout -q "$branch"
make t1
git checkout -q main
git merge --no-ff "$branch"
