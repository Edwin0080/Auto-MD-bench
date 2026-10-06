#!/bin/bash
# AIDE 0.2.2 with aide.patch, run inside the task sandbox by automdbench.run.
# Env: MODEL; the LLM endpoint is read by AIDE's OpenAI client (OPENAI_BASE_URL, OPENAI_API_KEY).
set -u
SANDBOX="$PWD"
DATA="$SANDBOX/.taskdata"   # AIDE copies its data directory into its workspace
mkdir -p "$DATA" "$SANDBOX/.tmp"
export TMPDIR="$SANDBOX/.tmp"
for f in "$SANDBOX"/*; do [ -f "$f" ] && [ "${f##*/}" != submission.csv ] && cp -f "$f" "$DATA/"; done
printf '\nBudget for this run: %s minutes of wall-clock time.\n' "$BUDGET_MIN" >> "$DATA/description.md"

# AIDE writes submissions in its workspace; keep the newest one at ./submission.csv
( last=""
  while true; do
    newest=$(find "$SANDBOX/.aide" -maxdepth 4 -name submission.csv -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1)
    if [ -n "$newest" ] && [ "$newest" != "$last" ] \
       && cp --preserve=timestamps -f "${newest#* }" .tmp/submission.mirror \
       && mv -f .tmp/submission.mirror submission.csv; then last="$newest"; fi
    sleep 5
  done ) &

# os._exit ends the run as soon as AIDE returns or crashes, even if an interpreter child hangs
python - data_dir="$DATA" desc_file="$DATA/description.md" exp_name=aide \
  log_dir="$SANDBOX/.aide/logs" workspace_dir="$SANDBOX/.aide/workspaces" copy_data=True generate_report=False \
  agent.steps=100000 exec.timeout=3600 agent.code.model="$MODEL" agent.feedback.model="$MODEL" <<'PY'
import os, sys, traceback
sys.argv = ["aide"] + sys.argv[1:]
rc = 0
try:
    from aide.run import run
    run()
except SystemExit as e:
    rc = e.code if isinstance(e.code, int) else int(e.code is not None)
except BaseException:
    traceback.print_exc()
    rc = 1
sys.stdout.flush(); sys.stderr.flush()
os._exit(rc)
PY
rc=$?
kill $(jobs -p) 2>/dev/null
exit $rc
