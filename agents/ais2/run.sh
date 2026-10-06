#!/bin/bash
# AI Scientist v2 (commit 96bd516) with ais2.patch, experiment stages only; run inside the task
# sandbox by automdbench.run.
# Env: AIS2 (path to the AI-Scientist-v2 checkout), MODEL; the LLM endpoint is read by its OpenAI
# client (OPENAI_BASE_URL, OPENAI_API_KEY).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SANDBOX="$PWD"
DATA="$SANDBOX/.taskdata"
mkdir -p "$DATA" "$SANDBOX/.tmp"
export TMPDIR="$SANDBOX/.tmp" PYTHONPATH="$AIS2${PYTHONPATH:+:$PYTHONPATH}"
for f in "$SANDBOX"/*; do [ -f "$f" ] && [ "${f##*/}" != submission.csv ] && cp -f "$f" "$DATA/"; done
sed -e "s#__DATA_DIR__#$DATA#" -e "s#__BUDGET_MIN__#$BUDGET_MIN#" -e "s#__MODEL__#$MODEL#g" \
  "$HERE/bfts_config.yaml" > bfts_config.yaml

# The idea file carries the full task description in Abstract (shown in every stage) and Experiments.
python - "$TASK" "$BUDGET_MIN" > .idea.json <<'PY'
import json, sys
desc = open("description.md", encoding="utf-8").read()
text = desc + f"\n\nBudget for this run: {sys.argv[2]} minutes of wall-clock time."
print(json.dumps([{
    "Name": sys.argv[1],
    "Title": desc.splitlines()[0].lstrip("# ").strip(),
    "Short Hypothesis": "Build the best predictive model for this task; performance is measured by the primary metric stated in the task description on a hidden test set.",
    "Related Work": "This is a benchmark task on a clinical dataset; use only the data files provided in the task directory, which is the current working directory of your program.",
    "Abstract": text,
    "Experiments": text,
    "Risk Factors and Limitations": "The stated rules apply: work only inside the current working directory, no network access, do not attempt to locate hidden labels. The final ./submission.csv on disk at the end of the run is what gets scored.",
}], indent=2))
PY

# nodes write submissions under experiments/; keep the newest one at ./submission.csv
( last=""
  while true; do
    newest=$(find "$SANDBOX/experiments" -name submission.csv -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1)
    if [ -n "$newest" ] && [ "$newest" != "$last" ] \
       && cp --preserve=timestamps -f "${newest#* }" .tmp/submission.mirror \
       && mv -f .tmp/submission.mirror submission.csv; then last="$newest"; fi
    sleep 5
  done ) &

python "$AIS2/launch_scientist_bfts.py" --load_ideas .idea.json --idea_idx 0 \
  --skip_writeup --skip_review --model_agg_plots "$MODEL"
rc=$?
kill $(jobs -p) 2>/dev/null
exit $rc
