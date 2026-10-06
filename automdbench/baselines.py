"""Reference models with fixed settings ("simple baseline models" in the paper), five fits each.

    python -m automdbench.baselines TASK [--seeds 0 1 2 3 4] [--packages packages] [--out runs/baselines]

survival        CoxPH neural network, MLP [32, 32], 64 epochs (pycox)
classification  HistGradientBoosting, 300 iterations
physionet2012   the same on per-variable mean/min/max/last; label threshold tuned on 15% of training data
sepsis          the same on forward-filled hourly rows; alarm threshold tuned on 15% of training patients
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from automdbench import metrics, tasks
from automdbench.grade import grade
from automdbench.sepsis_utility import score_cohort


def hgb(seed, **kw):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=False,
                                          random_state=seed, **kw)


def survival(pub, seed):
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH
    from sklearn.preprocessing import StandardScaler

    train, test = pd.read_csv(pub / "train.csv"), pd.read_csv(pub / "test_features.csv")
    grid = pd.read_csv(pub / "time_grid.csv")["time"].to_numpy()
    cols = [c for c in train.columns if c not in ("patient_id", "duration", "event")]
    x, xt = train[cols], test[cols]
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(x[c])]
    if cat:
        x = pd.get_dummies(x, columns=cat, dummy_na=True)
        xt = pd.get_dummies(xt, columns=cat, dummy_na=True).reindex(columns=x.columns, fill_value=0)
    x, xt = x.astype("float64"), xt.astype("float64")
    med = x.median()
    x, xt = x.fillna(med), xt.fillna(med)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(x)
    net = tt.practical.MLPVanilla(x.shape[1], [32, 32], 1, batch_norm=True, dropout=0.1, output_bias=False)
    model = CoxPH(net, tt.optim.Adam(1e-3))
    model.fit(scaler.transform(x).astype("float32"),
              (train["duration"].to_numpy("float32"), train["event"].to_numpy("float32")),
              batch_size=256, epochs=64, verbose=False)
    model.compute_baseline_hazards()
    surv = model.predict_surv_df(scaler.transform(xt).astype("float32"))
    idx = np.searchsorted(surv.index.to_numpy(float), grid, side="right") - 1
    s = np.ones((len(grid), surv.shape[1]))
    s[idx >= 0] = surv.to_numpy()[idx[idx >= 0]]
    sub = pd.DataFrame(np.clip(np.minimum.accumulate(s, axis=0), 0, 1).T, columns=metrics.grid_cols(len(grid)))
    sub.insert(0, "patient_id", test["patient_id"].to_numpy())
    return sub


def binary(pub, seed, id_col):
    train, test = pd.read_csv(pub / "train.csv"), pd.read_csv(pub / "test_features.csv")
    cols = [c for c in test.columns if c != id_col and pd.api.types.is_numeric_dtype(train[c])]
    clf = hgb(seed).fit(train[cols].to_numpy("float32"), train["label"].to_numpy())
    sub = test[[id_col]].copy()
    sub["PredictedProbability"] = np.round(clf.predict_proba(test[cols].to_numpy("float32"))[:, 1], 6)
    return sub


def _tuned(clf, tune_x, tune, test_x, test, key, score, quantiles):
    """Fit on the rest, choose the label threshold that maximizes `score` on the tuning split."""
    p = clf.predict_proba(tune_x)[:, 1]
    best, thr = -np.inf, 0.5
    for t in np.unique(np.quantile(p, quantiles)):
        s = score(tune.assign(PredictedProbability=p, PredictedLabel=(p >= t).astype(int)))
        if s > best:
            best, thr = s, float(t)
    pt = clf.predict_proba(test_x)[:, 1]
    sub = test[key].copy()
    sub["PredictedProbability"] = np.round(pt, 6)
    sub["PredictedLabel"] = (pt >= thr).astype(int)
    return sub


def physionet2012(pub, seed):
    def wide(s):
        a = (s.sort_values(["record_id", "minutes"]).groupby(["record_id", "parameter"])["value"]
             .agg(["mean", "min", "max", "last"]).reset_index().pivot(index="record_id", columns="parameter"))
        a.columns = [f"{p}_{stat}" for stat, p in a.columns]
        return a.reset_index()

    train = wide(pd.read_parquet(pub / "train_series.parquet")).merge(pd.read_csv(pub / "train_labels.csv"),
                                                                      on="record_id")
    cols = [c for c in train.columns if c not in ("record_id", "label")]
    test = wide(pd.read_parquet(pub / "test_features.parquet")).reindex(columns=["record_id"] + cols)
    fit, tune = train_test_split(train, test_size=0.15, random_state=seed, stratify=train["label"])
    clf = hgb(seed).fit(fit[cols].to_numpy("float32"), fit["label"].to_numpy())

    def score1(d):
        return metrics.binary_scores(d[["record_id", "PredictedProbability", "PredictedLabel"]],
                                     d[["record_id", "label"]], "record_id", True)["score1_min_se_ppv"]

    return _tuned(clf, tune[cols].to_numpy("float32"), tune[["record_id", "label"]],
                  test[cols].to_numpy("float32"), test, ["record_id"], score1, np.linspace(0.5, 0.99, 40))


def sepsis(pub, seed):
    def prep(df):
        df = df.sort_values(["patient", "ICULOS"]).reset_index(drop=True)
        cols = [c for c in df.columns if c not in ("patient", "hospital", "SepsisLabel")]
        df[cols] = df.groupby("patient", sort=False)[cols].ffill()
        df["hosp_B"] = (df["hospital"] == "B").astype("float32")
        return df

    train, test = prep(pd.read_parquet(pub / "train.parquet")), prep(pd.read_parquet(pub / "test_features.parquet"))
    cols = [c for c in train.columns if c not in ("patient", "hospital", "SepsisLabel") and train[c].notna().any()]
    pat = train.groupby("patient")["SepsisLabel"].max().reset_index()
    fit_p, tune_p = train_test_split(pat, test_size=0.15, random_state=seed, stratify=pat["SepsisLabel"])
    fit, tune = train[train["patient"].isin(set(fit_p["patient"]))], train[train["patient"].isin(set(tune_p["patient"]))]
    clf = hgb(seed, max_leaf_nodes=31).fit(fit[cols].to_numpy("float32"), fit["SepsisLabel"].to_numpy())
    return _tuned(clf, tune[cols].to_numpy("float32"), tune[["patient", "ICULOS", "SepsisLabel"]],
                  test[cols].to_numpy("float32"), test, ["patient", "ICULOS"],
                  lambda d: score_cohort(d[["patient", "ICULOS", "SepsisLabel"]], d)["utility_norm"],
                  np.linspace(0.80, 0.999, 40))


def fit(task, seed, packages="packages"):
    t, pub = tasks.get(task), Path(packages) / task / "public"
    if t.kind == "survival":
        return survival(pub, seed)
    if t.kind == "sepsis":
        return sepsis(pub, seed)
    return physionet2012(pub, seed) if task == "physionet2012" else binary(pub, seed, t.id_col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--packages", default="packages")
    ap.add_argument("--out", default="runs/baselines")
    a = ap.parse_args()
    out = Path(a.out) / a.task
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    for seed in a.seeds:
        path = out / f"seed{seed}.csv"
        fit(a.task, seed, a.packages).to_csv(path, index=False)
        r = grade(a.task, path, a.packages)
        scores.append(r["scores"])
        print(f"seed {seed}: " + json.dumps(r["scores"]))
    print("mean: " + json.dumps({k: float(np.mean([s[k] for s in scores])) for k in scores[0] if scores[0][k] is not None}))


if __name__ == "__main__":
    main()
