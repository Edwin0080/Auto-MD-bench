"""Score a submission against the hidden test labels.

    python -m automdbench.grade TASK SUBMISSION [--packages packages] [--out grade.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automdbench import metrics, tasks
from automdbench.sepsis_utility import score_cohort


def _check_probs(sub, need_label):
    issues = []
    p = sub["PredictedProbability"].to_numpy(dtype=float)
    if not np.isfinite(p).all():
        issues.append("non-finite probabilities")
    elif p.min() < 0 or p.max() > 1:
        issues.append(f"probabilities outside [0,1]: min={p.min():.4g} max={p.max():.4g}")
    if need_label and not sub["PredictedLabel"].isin([0, 1]).all():
        issues.append("PredictedLabel must be 0 or 1")
    return issues


def check_survival(sub, ids, n_grid):
    cols = ["patient_id"] + metrics.grid_cols(n_grid)
    if list(sub.columns) != cols:
        return [f"columns must be patient_id, t_000..t_{n_grid - 1:03d}"]
    issues = []
    got, want = set(sub["patient_id"]), set(ids)
    if len(got) != len(sub):
        issues.append("duplicate patient_id rows")
    if got != want:
        issues.append(f"patient_id mismatch: {len(want - got)} missing, {len(got - want)} unexpected")
    v = sub[cols[1:]].to_numpy(dtype=float)
    if not np.isfinite(v).all():
        return issues + ["non-finite values"]
    if v.min() < -1e-6 or v.max() > 1 + 1e-6:
        issues.append(f"values outside [0,1]: min={v.min():.4g} max={v.max():.4g}")
    if (np.diff(v, axis=1) > 1e-6).any():
        issues.append("survival curves must be non-increasing")
    return issues


def check_binary(sub, ids, id_col, need_label):
    cols = [id_col, "PredictedProbability"] + (["PredictedLabel"] if need_label else [])
    if list(sub.columns) != cols:
        return [f"columns must be {cols}"]
    issues = ["duplicate ids"] if sub[id_col].duplicated().any() else []
    a, b = np.sort(sub[id_col].to_numpy()), np.sort(np.asarray(ids))
    if len(a) != len(b) or not (a == b).all():
        issues.append(f"{id_col} mismatch ({len(a)} submitted, {len(b)} expected)")
    return issues + _check_probs(sub, need_label)


def check_sepsis(sub, labels):
    cols = ["patient", "ICULOS", "PredictedProbability", "PredictedLabel"]
    if list(sub.columns) != cols:
        return [f"columns must be {cols}"]
    key = ["patient", "ICULOS"]
    issues = ["duplicate (patient, ICULOS) rows"] if sub[key].duplicated().any() else []
    a, b = sub.sort_values(key)[key], labels.sort_values(key)[key]
    if len(a) != len(b) or not (a.to_numpy() == b.to_numpy()).all():
        issues.append(f"(patient, ICULOS) rows must match test_features ({len(a)} submitted, {len(b)} expected)")
    return issues + _check_probs(sub, True)


def grade(task, submission, packages="packages"):
    t = tasks.get(task)
    pkg = Path(packages) / task
    report = {"task": task, "valid": False, "issues": [], "scores": None, "normalized": 0.0}
    try:
        sub = pd.read_csv(submission)
        if t.kind == "sepsis":
            labels = pd.read_parquet(pkg / "private" / "test_labels.parquet")
            issues = check_sepsis(sub, labels)
        else:
            labels = pd.read_csv(pkg / "private" / "test_labels.csv")
            if t.kind == "survival":
                grid = pd.read_csv(pkg / "public" / "time_grid.csv")["time"].to_numpy()
                issues = check_survival(sub, labels["patient_id"], len(grid))
            else:
                issues = check_binary(sub, labels[t.id_col], t.id_col, t.need_label)
    except Exception as e:  # unreadable file, wrong dtypes
        issues = [f"cannot check submission: {e}"]
    report["issues"] = issues
    if issues:
        return report
    if t.kind == "survival":
        scores = metrics.survival_scores(sub, labels, grid)
    elif t.kind == "binary":
        scores = metrics.binary_scores(sub, labels, t.id_col, t.need_label)
    else:
        scores = score_cohort(labels, sub)
        m = labels.merge(sub, on=["patient", "ICULOS"])
        scores["ece"] = metrics.ece_binary(m["SepsisLabel"], m["PredictedProbability"])
    report.update(valid=True, scores=scores, normalized=tasks.normalize(t.metric, scores[t.metric]))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("submission")
    ap.add_argument("--packages", default="packages")
    ap.add_argument("--out")
    a = ap.parse_args()
    report = grade(a.task, a.submission, a.packages)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
