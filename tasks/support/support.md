# Task: Survival prediction on the SUPPORT cohort

You are building a machine-learning model for **patient survival prediction**
on SUPPORT, a study of seriously ill hospitalized adults. Each row is one
patient with 14 clinical covariates (`x0`..`x13`: demographics, comorbidity
indicators, vital signs, laboratory values; the exact data dictionary is not
provided) and a
**right-censored** survival outcome:

- `duration`: follow-up time in days
- `event`: 1 = death observed at `duration`; 0 = censored (patient was still
  alive when last seen — their true survival time is only known to exceed
  `duration`)

## Files in this directory

| file | contents |
|---|---|
| `train.csv` | 7098 patients WITH labels — all the labeled data you get |
| `test_features.csv` | 1775 held-out patients, features only (you will never see their labels) |
| `time_grid.csv` | the evaluation times (quantile-spaced, dense where events concentrate): submission column name ↔ time in days |
| `sample_submission.csv` | a valid example: the train-set Kaplan-Meier curve for every patient (a patient-independent null model) |
| `validate.py` | format checker — run `python validate.py submission.csv` |

## What to submit

Write `./submission.csv` with exactly these columns:

```
patient_id, t_000, t_001, ..., (every column listed in time_grid.csv, in order)
```

one row per patient in `test_features.csv`, where column `t_k` is your
predicted probability that the patient survives beyond the time that
`time_grid.csv` maps `t_k` to (values in [0,1], non-increasing along each row).

## How you are scored

Computed on the hidden test labels from your submission file alone:

- **`concordance_td`** (time-dependent concordance, Antolini) — primary, higher is better

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
