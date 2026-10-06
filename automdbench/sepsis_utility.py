"""Official PhysioNet/CinC Challenge 2019 scoring functions, vendored VERBATIM.

Source: https://github.com/physionetchallenges/evaluation-2019
        evaluate_sepsis_score.py (BSD 2-Clause, PhysioNet Challenge organizers)
Only the three pure scoring functions are kept (the psv-directory I/O wrapper
is not needed); nothing inside the function bodies is modified. Any change to
scoring semantics must happen in sepsis_scoring.py, never here.
"""

import warnings

import numpy as np


def compute_auc(labels, predictions, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not 0 <= prediction <= 1:
                warnings.warn('Predictions do not satisfy 0 <= prediction <= 1.')

    # Find prediction thresholds.
    thresholds = np.unique(predictions)[::-1]
    if thresholds[0] != 1:
        thresholds = np.insert(thresholds, 0, 1)
    if thresholds[-1] == 0:
        thresholds = thresholds[:-1]

    n = len(labels)
    m = len(thresholds)

    # Populate contingency table across prediction thresholds.
    tp = np.zeros(m)
    fp = np.zeros(m)
    fn = np.zeros(m)
    tn = np.zeros(m)

    # Find indices that sort the predicted probabilities from largest to
    # smallest.
    idx = np.argsort(predictions)[::-1]

    i = 0
    for j in range(m):
        # Initialize contingency table for j-th prediction threshold.
        if j == 0:
            tp[j] = 0
            fp[j] = 0
            fn[j] = np.sum(labels)
            tn[j] = n - fn[j]
        else:
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

        # Update contingency table for i-th largest predicted probability.
        while i < n and predictions[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Summarize contingency table.
    tpr = np.zeros(m)
    tnr = np.zeros(m)
    ppv = np.zeros(m)
    npv = np.zeros(m)

    for j in range(m):
        if tp[j] + fn[j]:
            tpr[j] = tp[j] / (tp[j] + fn[j])
        else:
            tpr[j] = 1
        if fp[j] + tn[j]:
            tnr[j] = tn[j] / (fp[j] + tn[j])
        else:
            tnr[j] = 1
        if tp[j] + fp[j]:
            ppv[j] = tp[j] / (tp[j] + fp[j])
        else:
            ppv[j] = 1
        if fn[j] + tn[j]:
            npv[j] = tn[j] / (fn[j] + tn[j])
        else:
            npv[j] = 1

    # Compute AUROC as the area under a piecewise linear function with TPR /
    # sensitivity (x-axis) and TNR / specificity (y-axis) and AUPRC as the area
    # under a piecewise constant with TPR / recall (x-axis) and PPV / precision
    # (y-axis).
    auroc = 0
    auprc = 0
    for j in range(m-1):
        auroc += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
        auprc += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    return auroc, auprc


def compute_accuracy_f_measure(labels, predictions, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

    # Populate contingency table.
    n = len(labels)
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(n):
        if labels[i] and predictions[i]:
            tp += 1
        elif not labels[i] and predictions[i]:
            fp += 1
        elif labels[i] and not predictions[i]:
            fn += 1
        elif not labels[i] and not predictions[i]:
            tn += 1

    # Summarize contingency table.
    if tp + fp + fn + tn:
        accuracy = float(tp + tn) / float(tp + fp + fn + tn)
    else:
        accuracy = 1.0

    if 2 * tp + fp + fn:
        f_measure = float(2 * tp) / float(2 * tp + fp + fn)
    else:
        f_measure = 1.0

    return accuracy, f_measure


def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)


# ======================================================================
"""Cohort-level sepsis scoring on DataFrames, mirroring the official
evaluate_sepsis_score() cohort loop (lines that compute best/inaction
predictions and the normalization) exactly, with the official per-patient
functions imported verbatim from sepsis_official.

Normalization identity (from the official code): all-zero predictions score
exactly 0; predicting exactly the optimal window for every septic patient
scores exactly 1. test_scoring_identities() checks both.
"""

import numpy as np
import pandas as pd

# (official functions are defined above in this file)

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1, -2, -0.05, 0


def score_cohort(labels_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """labels_df: patient, ICULOS, SepsisLabel. pred_df: patient, ICULOS,
    PredictedProbability, PredictedLabel. Must cover exactly the same
    (patient, ICULOS) pairs; rows are aligned here by sorting both the same way.
    """
    lab = labels_df.sort_values(["patient", "ICULOS"])
    pred = pred_df.sort_values(["patient", "ICULOS"])
    if len(lab) != len(pred):
        raise ValueError("labels and predictions have different row counts")
    if not (lab["patient"].to_numpy() == pred["patient"].to_numpy()).all():
        raise ValueError("patient alignment mismatch")
    if not (lab["ICULOS"].to_numpy() == pred["ICULOS"].to_numpy()).all():
        raise ValueError("ICULOS alignment mismatch")

    all_labels = lab["SepsisLabel"].to_numpy(dtype=int)
    all_probs = pred["PredictedProbability"].to_numpy(dtype=float)

    observed_u, best_u, inaction_u = [], [], []
    for _, g in pd.concat(
        [lab.reset_index(drop=True),
         pred.reset_index(drop=True)[["PredictedProbability", "PredictedLabel"]]],
        axis=1,
    ).groupby("patient", sort=False):
        labels = g["SepsisLabel"].to_numpy(dtype=int)
        predictions = g["PredictedLabel"].to_numpy(dtype=int)
        num_rows = len(labels)

        best_predictions = np.zeros(num_rows)
        if np.any(labels):
            t_sepsis = np.argmax(labels) - DT_OPTIMAL
            best_predictions[max(0, t_sepsis + DT_EARLY):
                             min(t_sepsis + DT_LATE + 1, num_rows)] = 1
        inaction_predictions = np.zeros(num_rows)

        args = (DT_EARLY, DT_OPTIMAL, DT_LATE, MAX_U_TP, MIN_U_FN, U_FP, U_TN)
        observed_u.append(compute_prediction_utility(labels, predictions, *args,
                                                     check_errors=False))
        best_u.append(compute_prediction_utility(labels, best_predictions, *args,
                                                 check_errors=False))
        inaction_u.append(compute_prediction_utility(labels, inaction_predictions,
                                                     *args, check_errors=False))

    obs, best, inaction = map(np.sum, (observed_u, best_u, inaction_u))
    utility_norm = float((obs - inaction) / (best - inaction))
    auroc, auprc = compute_auc(all_labels, all_probs, check_errors=False)
    return {
        "utility_norm": utility_norm,
        "auroc": float(auroc),
        "auprc": float(auprc),
    }


def test_scoring_identities(labels_df: pd.DataFrame) -> None:
    """Sanity checks required before trusting any real score."""
    zero = labels_df[["patient", "ICULOS"]].copy()
    zero["PredictedProbability"] = 0.0
    zero["PredictedLabel"] = 0
    s0 = score_cohort(labels_df, zero)
    assert abs(s0["utility_norm"] - 0.0) < 1e-12, s0

    optimal = []
    for pid, g in labels_df.sort_values(["patient", "ICULOS"]).groupby("patient",
                                                                       sort=False):
        labels = g["SepsisLabel"].to_numpy(dtype=int)
        p = np.zeros(len(labels))
        if np.any(labels):
            t_sepsis = np.argmax(labels) - DT_OPTIMAL
            p[max(0, t_sepsis + DT_EARLY): min(t_sepsis + DT_LATE + 1, len(labels))] = 1
        d = g[["patient", "ICULOS"]].copy()
        d["PredictedProbability"] = p
        d["PredictedLabel"] = p.astype(int)
        optimal.append(d)
    s1 = score_cohort(labels_df, pd.concat(optimal))
    assert abs(s1["utility_norm"] - 1.0) < 1e-12, s1
    print("scoring identities ok: inaction=0, optimal=1")
