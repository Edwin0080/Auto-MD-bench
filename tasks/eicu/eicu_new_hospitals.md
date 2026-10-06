# Task: ICU in-hospital mortality — CROSS-HOSPITAL deployment

You are building a model that predicts **in-hospital mortality** for adult ICU
patients from the **first 24 hours** of the ICU stay.

**Important: your training data come from one group of US hospitals, but your
model will be evaluated on ICU stays from a DIFFERENT, held-out group of
hospitals.** None of the test hospitals appears in your training data. Case
mix, charting habits, lab-ordering practice and documentation completeness
differ between hospitals.

One row per ICU stay; every stay has at least 24 hours of ICU time. Roughly
1 in 11 patients does not survive to discharge.

## Features

| group | columns |
|---|---|
| static | `age` (years), `gender` (1 = male) |
| vital signs (first 24h, `_mean`/`_min`/`_max` each) | `heart_rate`, `sbp`, `dbp`, `map`, `resp_rate`, `temp_c`, `spo2` |
| laboratory (first 24h, `_mean`/`_min`/`_max` each) | `glucose`, `creatinine`, `bun`, `sodium`, `potassium`, `chloride`, `bicarbonate`, `hemoglobin`, `hematocrit`, `wbc`, `platelets`, `lactate`, `bilirubin`, `albumin` |
| day-1 clinical assessment | `gcs` (Glasgow Coma Scale total, 3–15), `vent` (mechanical ventilation, 0/1), `urine_out` (24h urine output, mL), `pao2`, `fio2` (%) |
| admission context | `pre_icu_los_days` (time in hospital before ICU), `elective_surgery` (0/1) |
| chronic conditions | `aids`, `hepatic_failure`, `lymphoma`, `metastatic_cancer`, `leukemia`, `immunosuppression`, `cirrhosis`, `diabetes` (each 0/1) |

Missing values are frequent, and which fields get recorded, and how often,
differs between hospitals.

## Files

| file | contents |
|---|---|
| `train.csv` | training-hospital stays WITH `label` (1 = died in hospital) |
| `test_features.csv` | stays from the held-out hospitals, features only |
| `sample_submission.csv` | constant prevalence — a valid, weak baseline |
| `validate.py` | format checker — `python validate.py submission.csv` |

## What to submit

`./submission.csv` with exactly two columns:

```
stay_id, PredictedProbability
```

one row per stay in `test_features.csv`, probability in [0,1].

## Scoring (hidden labels, from your file alone)

- **`auroc`** — primary, higher better

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
