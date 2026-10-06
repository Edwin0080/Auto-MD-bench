"""PhysioNet/CinC Challenge 2019: hourly early prediction of sepsis onset in two hospital systems.

    sepsis               patients from both hospitals, 80/20 split stratified by hospital and sepsis
    sepsis_new_hospital  train on hospital A only, test on the hospital-B patients of the sepsis test set
    sepsis_500_labels    as above plus 500 labeled hospital-B patients in training

    python tasks/sepsis/build.py RAW_DIR [--out packages]
RAW_DIR holds training_setA/ and training_setB/ (one .psv per patient).
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from automdbench import package

HERE = Path(__file__).resolve().parent


def load(raw):
    frames = []
    for hospital, sub in (("A", "training_setA"), ("B", "training_setB")):
        for f in sorted((Path(raw) / sub).glob("p*.psv")):
            d = pd.read_csv(f, sep="|")
            d.insert(0, "patient", f.stem)
            d.insert(1, "hospital", hospital)
            frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data["ICULOS"] = data["ICULOS"].astype(int)
    data["SepsisLabel"] = data["SepsisLabel"].astype(int)
    floats = [c for c in data.columns if c not in ("patient", "hospital", "ICULOS", "SepsisLabel")]
    data[floats] = data[floats].astype("float32")
    return data.sort_values(["patient", "ICULOS"]).reset_index(drop=True)


def write(out, task, data, train_ids, test_ids):
    test = data[data["patient"].isin(test_ids)]
    ids = package.opaque_ids(sorted(set(test["patient"])), "s")
    test = (test.assign(patient=test["patient"].map(ids))
            .sort_values(["patient", "ICULOS"], kind="stable").reset_index(drop=True))
    sample = test[["patient", "ICULOS"]].copy()
    sample["PredictedProbability"] = 0.0
    sample["PredictedLabel"] = 0
    package.write(out, task, HERE / f"{task}.md",
                  {"train.parquet": data[data["patient"].isin(train_ids)],
                   "test_features.parquet": test.drop(columns=["SepsisLabel"]),
                   "sample_submission.csv": sample},
                  {"test_labels.parquet": test[["patient", "ICULOS", "SepsisLabel", "hospital"]]})


def main(raw, out):
    data = load(raw)
    pat = (data.groupby(["patient", "hospital"])["SepsisLabel"].max().reset_index()
           .rename(columns={"SepsisLabel": "septic"}))
    pat["stratum"] = pat["hospital"] + pat["septic"].astype(str)
    train_pat, test_pat = train_test_split(pat, test_size=0.20, random_state=42, shuffle=True,
                                           stratify=pat["stratum"])
    write(out, "sepsis", data, set(train_pat["patient"]), set(test_pat["patient"]))

    test_b = set(test_pat.loc[test_pat["hospital"] == "B", "patient"])
    hosp_a = set(pat.loc[pat["hospital"] == "A", "patient"])
    write(out, "sepsis_new_hospital", data, hosp_a, test_b)

    pool_b = pat[(pat["hospital"] == "B") & ~pat["patient"].isin(test_b)]
    labeled_b, _ = train_test_split(pool_b, train_size=500, random_state=42, stratify=pool_b["septic"])
    write(out, "sepsis_500_labels", data, hosp_a | set(labeled_b["patient"]), test_b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", default="packages")
    main(**vars(ap.parse_args()))
