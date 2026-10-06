"""Write a task package: public/ is what the agent sees, private/ holds the hidden test labels."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from automdbench import metrics, tasks

HERE = Path(__file__).resolve().parent


def opaque_ids(ids, prefix, seed=1042):
    """Random opaque test ids, so test rows cannot be joined back to the public source data."""
    ids = list(dict.fromkeys(ids))
    width = max(4, len(str(len(ids))))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {orig: f"{prefix}{k:0{width}d}" for orig, k in zip(ids, perm)}


def write(out, task, description, public, private):
    """public / private: {file name: DataFrame}; .csv or .parquet by suffix."""
    t = tasks.get(task)
    root = Path(out) / task
    if root.exists():
        shutil.rmtree(root)
    for sub, files in (("public", public), ("private", private)):
        (root / sub).mkdir(parents=True)
        for name, df in files.items():
            if name.endswith(".parquet"):
                df.to_parquet(root / sub / name, index=False)
            else:
                df.to_csv(root / sub / name, index=False, lineterminator="\n")
    shutil.copyfile(description, root / "public" / "description.md")
    validate = (HERE / "templates" / f"validate_{t.kind}.py").read_text(encoding="utf-8")
    if t.kind == "binary":
        validate = validate.replace("__ID_COL__", t.id_col).replace("__NEED_LABEL__", str(t.need_label))
    (root / "public" / "validate.py").write_text(validate, encoding="utf-8", newline="\n")
    if t.kind == "sepsis":
        shutil.copyfile(HERE / "sepsis_utility.py", root / "public" / "utility_score.py")
    print(f"{task}: {', '.join(f'{k} {len(v):,}' for k, v in {**public, **private}.items())}")


def survival(out, task, description, train, test, grid=None):
    """Survival package: S(t) on a fixed grid of times; the sample submission is the training KM curve."""
    if grid is None:
        grid = metrics.make_time_grid(train["duration"], train["event"])
    km = metrics.km_curve(train["duration"], train["event"], grid)
    sample = pd.DataFrame(np.tile(np.round(km, 6), (len(test), 1)), columns=metrics.grid_cols(len(grid)))
    sample.insert(0, "patient_id", test["patient_id"].to_numpy())
    write(out, task, description,
          {"train.csv": train,
           "test_features.csv": test.drop(columns=["duration", "event"]),
           "time_grid.csv": pd.DataFrame({"column": metrics.grid_cols(len(grid)), "time": grid}),
           "sample_submission.csv": sample},
          {"test_labels.csv": test[["patient_id", "duration", "event"]]})


def binary(out, task, description, train, test, id_col):
    """Tabular binary package; the sample submission predicts the training prevalence."""
    sample = test[[id_col]].copy()
    sample["PredictedProbability"] = round(float(train["label"].mean()), 6)
    write(out, task, description,
          {"train.csv": train, "test_features.csv": test.drop(columns=["label"]), "sample_submission.csv": sample},
          {"test_labels.csv": test[[id_col, "label"]]})
