#!/bin/bash
# Build the 10 public task packages from DATA_DIR into OUT_DIR.
#   usage: bash scripts/build_public_tasks.sh [DATA_DIR] [OUT_DIR]
set -eu
D="${1:-data/raw}"; OUT="${2:-packages}"
for t in support metabric rotterdam uci_diabetes physionet2012 sepsis nhanes; do
  python "tasks/$t/build.py" "$D/$t" --out "$OUT"
done
