# Task: 30-day readmission prediction (Diabetes, 130 US hospitals)

Predict whether a hospital encounter of a diabetic patient is followed by a
**readmission within 30 days**. One row per ENCOUNTER; columns:

- `encounter_id` — the unit you predict for
- `patient_nbr` — patient identifier, **train only**. Patients can have
  several encounters; the hidden test split is at PATIENT level. The test file
  has NO patient identifier and its rows are shuffled: every test row stands
  on its own.
- ~40 features (demographics, admission details, diagnoses, lab counts,
  medications). Categorical columns arrive ordinal-encoded; missing = NaN.
- `label` (train only): 1 = readmitted within 30 days (~11% of encounters).

## Files

| file | contents |
|---|---|
| `train.csv` | ~81k encounters WITH labels |
| `test_features.csv` | ~20k encounters, features only |
| `sample_submission.csv` | valid null: the training prevalence for every encounter |
| `validate.py` | format checker — `python validate.py submission.csv` |

## What to submit

`./submission.csv` with exactly: `encounter_id, PredictedProbability`
(probability of readmission within 30 days, in [0,1]), one row per test
encounter.

## Scoring (hidden labels, from your file alone)

- **`brier`** — primary, lower better

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
