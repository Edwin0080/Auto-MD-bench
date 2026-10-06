"""PhysioNet/CinC Challenge 2012: in-hospital death from the first 48 h of an ICU stay.
Train on set A; test on sets B and C together (8,000 stays, shuffled opaque ids).

    python tasks/physionet2012/build.py RAW_DIR [--out packages]
RAW_DIR holds set-a/, set-b/, set-c/ (one .txt per stay) and Outcomes-{a,b,c}.txt.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from automdbench import package

HERE = Path(__file__).resolve().parent
STATICS = ["Age", "Gender", "Height", "ICUType", "Weight"]


def series(setdir):
    rows = []
    for f in sorted(setdir.glob("*.txt")):
        d = pd.read_csv(f)
        rows.append(pd.DataFrame({
            "record_id": int(f.stem),
            "minutes": d["Time"].str.slice(0, 2).astype(int) * 60 + d["Time"].str.slice(3, 5).astype(int),
            "parameter": d["Parameter"],
            "value": pd.to_numeric(d["Value"], errors="coerce")}).query("parameter != 'RecordID'"))
    out = pd.concat(rows, ignore_index=True)
    out.loc[out["parameter"].isin(STATICS) & (out["value"] < 0), "value"] = np.nan  # -1 = missing
    return out


def outcomes(path):
    return pd.read_csv(path).rename(columns={"RecordID": "record_id", "In-hospital_death": "label"})[
        ["record_id", "label"]]


def main(raw, out):
    raw = Path(raw)
    train, train_labels = series(raw / "set-a"), outcomes(raw / "Outcomes-a.txt")
    test = pd.concat([series(raw / f"set-{c}") for c in "bc"], ignore_index=True)
    labels = pd.concat([outcomes(raw / f"Outcomes-{c}.txt") for c in "bc"], ignore_index=True)
    ids = package.opaque_ids(sorted(labels["record_id"].tolist()), "t")
    test["record_id"] = test["record_id"].map(ids)
    labels["record_id"] = labels["record_id"].map(ids)
    sample = pd.DataFrame({"record_id": sorted(labels["record_id"])})
    sample["PredictedProbability"] = round(float(train_labels["label"].mean()), 6)
    sample["PredictedLabel"] = 0
    package.write(out, "physionet2012", HERE / "physionet2012.md",
                  {"train_series.parquet": train, "train_labels.csv": train_labels,
                   "test_features.parquet": test, "sample_submission.csv": sample},
                  {"test_labels.csv": labels})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", default="packages")
    main(**vars(ap.parse_args()))
