# Task: Survival prediction on the METABRIC breast-cancer cohort

You are building a survival model for breast-cancer patients (METABRIC
cohort, 1,523 training patients). Each row is one patient with 9 numeric
covariates (`x0`..`x8`: four gene-expression markers and five clinical
indicators; the exact dictionary is not provided) and a **right-censored**
survival outcome:

- `duration`: follow-up time (months); `event`: 1 = death observed, 0 =
  censored (alive at last contact — true survival time only known to exceed
  `duration`).

## Files

| file | contents |
|---|---|
| `train.csv` | 1,523 patients WITH labels — all the labeled data you get |
| `test_features.csv` | 381 held-out patients, features only |
| `time_grid.csv` | evaluation times (quantile-spaced): submission column ↔ time |
| `sample_submission.csv` | valid null example: train-set Kaplan-Meier for every patient |
| `validate.py` | format checker — run `python validate.py submission.csv` |

## What to submit

`./submission.csv` with columns `patient_id, t_000, t_001, ...` (every column
in `time_grid.csv`, in order): column `t_k` = your predicted probability that
the patient survives beyond the mapped time (values in [0,1], non-increasing
along each row), one row per test patient.

## Scoring (computed on hidden test labels from your file alone)

- **`concordance_td`** (Antolini) — primary, higher is better

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
