#!/usr/bin/env python
"""Submission format checker (no labels involved).

Usage:  python validate.py [submission.csv]

Checks column layout, patient coverage, value range and monotonicity, then
appends one JSON line per call to ./validate_log.jsonl. Passing this check
means your file is GRADABLE — it says nothing about how good the scores are.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOL = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path):
    issues = []
    try:
        sub = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        return [f"cannot read CSV: {e}"]
    grid_cols = pd.read_csv(HERE / "time_grid.csv")["column"].tolist()
    expected_cols = ["patient_id"] + grid_cols
    if list(sub.columns) != expected_cols:
        return [
            f"columns mismatch: expected patient_id + the {len(grid_cols)} columns "
            f"named in time_grid.csv, got {len(sub.columns)} columns"
        ]
    expected_ids = set(pd.read_csv(HERE / "test_features.csv")["patient_id"].tolist())
    ids = sub["patient_id"].tolist()
    if len(ids) != len(set(ids)):
        issues.append("duplicate patient_id rows")
    if set(ids) != expected_ids:
        issues.append(
            f"patient_id set mismatch: {len(expected_ids - set(ids))} missing, "
            f"{len(set(ids) - expected_ids)} unexpected"
        )
    vals = sub[expected_cols[1:]].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        issues.append("non-finite values (NaN/inf)")
    else:
        if vals.min() < -TOL or vals.max() > 1 + TOL:
            issues.append(f"values outside [0,1]: min={vals.min():.4g} max={vals.max():.4g}")
        n_viol = int((np.diff(vals, axis=1) > TOL).sum())
        if n_viol:
            issues.append(f"not non-increasing: {n_viol} increasing steps")
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
