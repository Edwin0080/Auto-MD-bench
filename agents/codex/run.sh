#!/bin/bash
# Codex CLI 0.149.1, run inside the task sandbox by automdbench.run.   usage: run.sh PROMPT_FILE
# The model provider comes from your Codex configuration; MODEL, if set, overrides the model.
set -u
mkdir -p .tmp
export TMPDIR="$PWD/.tmp"
PROMPT="$(cat "$1")"
exec codex exec --skip-git-repo-check -s danger-full-access ${MODEL:+-m "$MODEL"} "${PROMPT//__BUDGET_MIN__/$BUDGET_MIN}"
