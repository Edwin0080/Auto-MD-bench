#!/usr/bin/env python
"""Submission format checker for the sepsis task (no labels involved).

Usage:  python validate.py [submission.csv]

Checks column layout, exact (patient, ICULOS) coverage against
test_features.parquet, probability range, and binary labels. Appends one JSON
line per call to ./validate_log.jsonl. Passing means your file is GRADABLE —
it says nothing about the scores.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REQUIRED_COLS = ["patient", "ICULOS", "PredictedProbability", "PredictedLabel"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path):
    try:
        sub = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        return [f"cannot read CSV: {e}"]
    if list(sub.columns) != REQUIRED_COLS:
        return [f"columns mismatch: expected exactly {REQUIRED_COLS}"]
    issues = []
    if sub[["patient", "ICULOS"]].duplicated().any():
        issues.append("duplicate (patient, ICULOS) rows")
    exp = pd.read_parquet(HERE / "test_features.parquet",
                          columns=["patient", "ICULOS"])
    a = sub.sort_values(["patient", "ICULOS"])
    b = exp.sort_values(["patient", "ICULOS"])
    if len(a) != len(b) or not (
        (a["patient"].to_numpy() == b["patient"].to_numpy()).all()
        and (a["ICULOS"].to_numpy() == b["ICULOS"].to_numpy()).all()
    ):
        issues.append(f"(patient, ICULOS) pairs must exactly match "
                      f"test_features.parquet ({len(a)} vs {len(b)} rows)")
    probs = sub["PredictedProbability"].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        issues.append("non-finite probabilities")
    elif probs.min() < 0 or probs.max() > 1:
        issues.append(f"probabilities outside [0,1]: min={probs.min():.4g} "
                      f"max={probs.max():.4g}")
    if not sub["PredictedLabel"].isin([0, 1]).all():
        issues.append("PredictedLabel must be 0 or 1")
    return issues


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "submission.csv"
    if not target.exists():
        print(f"FAIL: {target} not found")
        sys.exit(1)
    issues = check(target)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "file": str(target.name),
        "sha256": sha256(target),
        "valid": not issues,
        "issues": issues,
    }
    with open(HERE / "validate_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    if issues:
        print("FAIL:")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
    print("OK: submission format is valid and gradable")


if __name__ == "__main__":
    main()
