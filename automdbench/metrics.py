"""Scores computed from a submission file and the hidden test labels."""
import numpy as np
import pandas as pd
import scipy.integrate
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

if not hasattr(scipy.integrate, "simps"):  # pycox 0.3.0 still calls simps
    scipy.integrate.simps = scipy.integrate.simpson


def grid_cols(n):
    return [f"t_{i:03d}" for i in range(n)]


def make_time_grid(durations, events, n=100):
    """Evaluation times: quantiles of the training event times, plus 0."""
    t = np.asarray(durations, dtype=float)[np.asarray(events, dtype=int) == 1]
    grid = np.unique(np.quantile(t, np.linspace(0.0, 1.0, n)))
    return np.concatenate([[0.0], grid]) if grid[0] > 0.0 else grid


def km_curve(durations, events, times):
    """Kaplan-Meier survival evaluated at `times`."""
    order = np.argsort(np.asarray(durations, dtype=float))
    d = np.asarray(durations, dtype=float)[order]
    e = np.asarray(events, dtype=int)[order]
    u = np.unique(d[e == 1])
    at_risk = len(d) - np.searchsorted(d, u, side="left")
    deaths = np.array([np.sum((d == t) & (e == 1)) for t in u])
    s = np.cumprod(1.0 - deaths / at_risk)
    idx = np.searchsorted(u, np.asarray(times, dtype=float), side="right")
    out = np.ones(len(times))
    out[idx > 0] = s[idx[idx > 0] - 1]
    return out


def _km_at(d, e, t):
    s = 1.0
    for u in np.unique(d[(e == 1) & (d <= t)]):
        s *= 1 - ((d == u) & (e == 1)).sum() / (d >= u).sum()
    return s


def ece_binary(y, p, n_bins=10):
    """Expected calibration error, 10 equal-width bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean())
                     for k in range(n_bins) if (b == k).any()))


def ece_survival(sub, labels, grid):
    """Mean ECE at the 25th, 50th and 75th percentiles of the test event times. At each time the
    patients are split into 10 bins by predicted S(t); each bin's mean prediction is compared with
    its Kaplan-Meier estimate."""
    m = labels[["patient_id", "duration", "event"]].merge(sub, on="patient_id")
    cols = grid_cols(len(grid))
    ev = m.loc[m["event"] == 1, "duration"].to_numpy(float)
    if len(ev) < 20:
        return None
    out = []
    for q in (0.25, 0.5, 0.75):
        j = int(np.argmin(np.abs(np.asarray(grid, dtype=float) - np.quantile(ev, q))))
        pred = m[cols[j]].to_numpy(float)
        d, e = m["duration"].to_numpy(float), m["event"].to_numpy(int)
        rank = pd.Series(pred).rank(method="first").to_numpy()
        b = np.minimum((rank - 1) * 10 // len(pred), 9).astype(int)
        out.append(sum((b == k).mean() * abs(pred[b == k].mean() - _km_at(d[b == k], e[b == k], grid[j]))
                       for k in range(10) if (b == k).sum() >= 5))
    return float(np.mean(out))


def survival_scores(sub, labels, grid):
    """Time-dependent concordance (Antolini), integrated Brier score, integrated NBLL, ECE."""
    from pycox.evaluation import EvalSurv

    grid = np.asarray(grid, dtype=float)
    s = sub.set_index("patient_id").loc[labels["patient_id"].to_numpy()][grid_cols(len(grid))]
    ev = EvalSurv(pd.DataFrame(np.clip(s.to_numpy(float), 0, 1).T, index=grid),
                  labels["duration"].to_numpy(float), labels["event"].to_numpy(int), censor_surv="km")
    return {"concordance_td": float(ev.concordance_td("adj_antolini")),
            "integrated_brier_score": float(ev.integrated_brier_score(grid)),
            "integrated_nbll": float(ev.integrated_nbll(grid)),
            "ece": ece_survival(sub, labels, grid)}


def binary_scores(sub, labels, id_col, need_label=False):
    """AUROC, AUPRC, Brier, ECE; with predicted labels also min(sensitivity, PPV)."""
    m = labels.merge(sub, on=id_col, validate="one_to_one")
    y, p = m["label"].to_numpy(int), m["PredictedProbability"].to_numpy(float)
    out = {"auroc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else 0.5,
           "auprc": float(average_precision_score(y, p)),
           "brier": float(brier_score_loss(y, p)),
           "ece": ece_binary(y, p)}
    if need_label:
        pred = m["PredictedLabel"].to_numpy(int)
        tp, fn, fp = ((y == 1) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum(), ((y == 0) & (pred == 1)).sum()
        se = tp / (tp + fn) if tp + fn else 0.0
        ppv = tp / (tp + fp) if tp + fp else 0.0
        out["score1_min_se_ppv"] = float(min(se, ppv))
    return out
