# Task: Recurrence-free survival on the Rotterdam breast-cancer cohort

You are building a survival model for **recurrence-free survival** (time to
recurrence or death, whichever first) in breast-cancer patients from the
Rotterdam tumor-bank cohort. One row per patient, 8 clinical covariates:

`age` (years), `meno` (menopausal status 0/1), `sizecat` (tumor size: 1 =
≤20mm, 2 = 20–50mm, 3 = >50mm), `grade` (differentiation grade), `nodes`
(positive lymph nodes), `pgr` (progesterone receptor, fmol/l), `er`
(estrogen receptor, fmol/l), `hormon` (hormonal therapy 0/1) — plus the
**right-censored** outcome `duration` (days) / `event` (1 = recurrence or
death observed; 0 = censored).

## Files

| file | contents |
|---|---|
| `train.csv` | 2,385 patients WITH labels |
| `test_features.csv` | 597 held-out patients, features only |
| `time_grid.csv` | evaluation times (quantile-spaced): submission column ↔ time |
| `sample_submission.csv` | valid null example: train-set Kaplan-Meier for every patient |
| `validate.py` | format checker — `python validate.py submission.csv` |

## What to submit

`./submission.csv` with columns `patient_id, t_000, t_001, ...` (every column
in `time_grid.csv`, in order): `t_k` = predicted probability of surviving
event-free beyond the mapped time ([0,1], non-increasing per row), one row
per test patient.

## Scoring (hidden test labels, from your file alone)

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
