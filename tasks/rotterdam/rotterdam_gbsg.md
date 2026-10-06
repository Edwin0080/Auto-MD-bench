# Task: Breast-cancer survival — EXTERNAL VALIDATION on a different cohort

You are building a **recurrence-free survival** model (time to recurrence or
death). **Important: your training data are the Rotterdam tumor-bank cohort
(a Dutch registry), but your model will be evaluated on the GBSG cohort — a
German randomized-trial population.** Case mix, treatment patterns, and
follow-up differ between the two cohorts.

One row per patient, 8 harmonized covariates: `age`, `meno`, `sizecat`
(1 = ≤20mm, 2 = 20–50mm, 3 = >50mm), `grade`, `nodes`, `pgr`, `er`,
`hormon` — plus right-censored `duration` (days) / `event`.

## Files

| file | contents |
|---|---|
| `train.csv` | 2,982 Rotterdam patients WITH labels |
| `test_features.csv` | 686 GBSG patients, features only |
| `time_grid.csv` | evaluation times (quantile-spaced): submission column ↔ time |
| `sample_submission.csv` | valid null example: train-set Kaplan-Meier for every patient |
| `validate.py` | format checker — `python validate.py submission.csv` |

## What to submit

`./submission.csv` with columns `patient_id, t_000, t_001, ...` (every column
in `time_grid.csv`, in order): `t_k` = predicted probability of surviving
event-free beyond the mapped time ([0,1], non-increasing per row), one row
per test patient.

## Scoring (hidden GBSG labels, from your file alone)

- **`concordance_td`** (Antolini) — primary, higher better

## Environment

One machine: 8 CPU threads (4 physical cores), 16 GB of memory, no GPU. The
memory limit is enforced: a process that exceeds it is killed. `python` (3.11)
with numpy, pandas, pyarrow, scipy, scikit-learn, lightgbm, xgboost, torch
(CPU), torchtuples, pycox, lifelines, scikit-survival and matplotlib (the
example datasets bundled with lifelines, scikit-survival and pycox have been
removed). No network access; additional packages cannot be installed. Use only
the files provided here.

## Rules

- Work entirely inside this directory; do not read or write files outside it.
- Do not attempt to locate the hidden test labels.
- The run has a fixed wall-clock budget of 12 hours (720 minutes). The
  `submission.csv` on disk when the budget ends is what gets graded; an
  invalid or missing submission is a failed run.
