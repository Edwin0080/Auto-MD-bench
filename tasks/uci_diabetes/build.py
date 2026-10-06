"""UCI Diabetes 130-US hospitals: 30-day readmission, one row per encounter, split by patient.
Test rows carry no patient number and are shuffled, since a model cannot see a patient's other
encounters at prediction time.

    python tasks/uci_diabetes/build.py RAW_DIR [--out packages]   RAW_DIR holds diabetic_data.csv
Also writes uci_diabetes_brier and uci_diabetes_ece (calibration control; same data, other scoring text).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from automdbench import package

HERE = Path(__file__).resolve().parent


def main(raw, out):
    df = pd.read_csv(Path(raw) / "diabetic_data.csv").replace("?", np.nan)
    label = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted", "weight", "payer_code", "medical_specialty"])  # >40% missing
    for c in df.columns:
        if c not in ("encounter_id", "patient_nbr") and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category").cat.codes.replace(-1, np.nan).astype("float32")
    df["label"] = label

    pat = df.groupby("patient_nbr")["label"].max().reset_index()
    train_pat, test_pat = train_test_split(pat, test_size=0.20, random_state=42, stratify=pat["label"])
    train = df[df["patient_nbr"].isin(set(train_pat["patient_nbr"]))]
    test = df[df["patient_nbr"].isin(set(test_pat["patient_nbr"]))]
    ids = package.opaque_ids(test["encounter_id"].tolist(), "e")
    test = test.assign(encounter_id=test["encounter_id"].map(ids))
    features = (test.drop(columns=["label", "patient_nbr"])
                .sample(frac=1.0, random_state=4242).reset_index(drop=True))
    sample = features[["encounter_id"]].copy()
    sample["PredictedProbability"] = round(float(train["label"].mean()), 6)
    for task in ("uci_diabetes", "uci_diabetes_brier", "uci_diabetes_ece"):
        package.write(out, task, HERE / f"{task}.md",
                      {"train.csv": train, "test_features.csv": features, "sample_submission.csv": sample},
                      {"test_labels.csv": test[["encounter_id", "label"]]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", default="packages")
    main(**vars(ap.parse_args()))
