#!/usr/bin/env python
"""Submission format checker (no labels involved). Built per-task from a
template: __ID_COL__ / __NEED_LABEL__ are substituted at capsule build time.

Usage:  python validate.py [submission.csv]
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ID_COL = "__ID_COL__"
NEED_LABEL = __NEED_LABEL__


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
    cols = [ID_COL, "PredictedProbability"] + (["PredictedLabel"] if NEED_LABEL else [])
    if list(sub.columns) != cols:
        return [f"columns mismatch: expected exactly {cols}"]
    issues = []
    if sub[ID_COL].duplicated().any():
        issues.append(f"duplicate {ID_COL} rows")
    tf = HERE / "test_features.csv"
    if not tf.exists():
        tf = HERE / "test_features.parquet"
        exp = pd.read_parquet(tf, columns=[ID_COL])[ID_COL]
    else:
        exp = pd.read_csv(tf, usecols=[ID_COL])[ID_COL]
    a = np.sort(sub[ID_COL].to_numpy())
    b = np.sort(exp.unique())
    if len(a) != len(b) or not (a == b).all():
        issues.append(f"{ID_COL} set mismatch ({len(a)} vs {len(b)} expected)")
    probs = sub["PredictedProbability"].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        issues.append("non-finite probabilities")
    elif probs.min() < 0 or probs.max() > 1:
        issues.append(f"probabilities outside [0,1]: min={probs.min():.4g} "
                      f"max={probs.max():.4g}")
    if NEED_LABEL and not sub["PredictedLabel"].isin([0, 1]).all():
        issues.append("PredictedLabel must be 0 or 1")
    return issues


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "submission.csv"
    if not target.exists():
        print(f"FAIL: {target} not found")
        sys.exit(1)
    issues = check(target)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": str(target.name),
             "sha256": sha256(target), "valid": not issues, "issues": issues}
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
