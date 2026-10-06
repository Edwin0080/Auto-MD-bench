"""NHANES 2011-2016 linked to NCHS mortality through 2019: all-cause death after the exam.
Survey design variables (weight, PSU, stratum) are included with the features.

    python tasks/nhanes/build.py RAW_DIR [--out packages]
RAW_DIR holds the XPT files of cycles G, H, I and the public-use linked mortality files
(scripts/download_public_data.sh fetches them).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from automdbench import package

HERE = Path(__file__).resolve().parent
CYCLES = {"G": 2011, "H": 2013, "I": 2015}
SPEC = {
    "DEMO": {"RIDAGEYR": "age", "RIAGENDR": "sex", "RIDRETH3": "race_eth", "DMDEDUC2": "education",
             "INDFMPIR": "income_poverty_ratio", "WTMEC2YR": "survey_weight", "SDMVPSU": "survey_psu",
             "SDMVSTRA": "survey_stratum"},
    "BMX": {"BMXBMI": "bmi", "BMXWAIST": "waist_cm"},
    "BPX": {"BPXSY1": "sbp1", "BPXSY2": "sbp2", "BPXDI1": "dbp1", "BPXDI2": "dbp2"},
    "TCHOL": {"LBXTC": "total_cholesterol"},
    "HDL": {"LBDHDD": "hdl_cholesterol"},
    "GHB": {"LBXGH": "hba1c"},
    "CBC": {"LBXWBCSI": "wbc", "LBXHGB": "hemoglobin", "LBXPLTSI": "platelets"},
    "BIOPRO": {"LBXSCR": "creatinine", "LBXSAL": "albumin", "LBXSGL": "glucose", "LBXSAPSI": "alk_phos",
               "LBXSASSI": "ast"},
    "SMQ": {"SMQ020": "ever_smoked_100"},
}


def mortality(path):
    """Fixed-width public-use file: SEQN 1-6, ELIGSTAT 15, MORTSTAT 16, follow-up months from exam."""
    rows = []
    for line in open(path):
        line = line.rstrip("\r\n")
        if len(line) < 20:
            continue
        tail = line[42:].split()
        rows.append({"SEQN": int(line[0:6]), "eligible": line[14:15].strip(), "mortstat": line[15:16].strip(),
                     "permth_exm": tail[1] if len(tail) > 1 else ""})
    df = pd.DataFrame(rows)
    df = df[(df["eligible"] == "1") & df["mortstat"].isin(["0", "1"])]
    df["event"] = df["mortstat"].astype(int)
    df["duration"] = pd.to_numeric(df["permth_exm"], errors="coerce")
    return df[["SEQN", "duration", "event"]].dropna()


def cycle(raw, letter):
    out = None
    for name, cols in SPEC.items():
        d = pd.read_sas(raw / f"{name}_{letter}.xpt", format="xport")
        d = d[["SEQN"] + [c for c in cols if c in d.columns]].rename(columns=cols)
        out = d if out is None else out.merge(d, on="SEQN", how="left")
    return out


def main(raw, out):
    raw = Path(raw)
    frames = []
    for letter, year in CYCLES.items():
        df = cycle(raw, letter).merge(mortality(raw / f"NHANES_{year}_{year + 1}_MORT_2019_PUBLIC.dat"),
                                      on="SEQN", how="inner")
        df["cycle"] = year
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["age"] >= 20) & (df["duration"] > 0) & (df["survey_weight"] > 0)].reset_index(drop=True)
    df["sbp"] = df[["sbp1", "sbp2"]].mean(axis=1)
    df["dbp"] = df[["dbp1", "dbp2"]].replace(0, np.nan).mean(axis=1)
    df = df.drop(columns=["sbp1", "sbp2", "dbp1", "dbp2"])
    df["sex"] = (df["sex"] == 1).astype(int)  # 1 = male
    df["ever_smoked_100"] = df["ever_smoked_100"].map({1.0: 1, 2.0: 0})
    df["patient_id"] = df["SEQN"].astype("int64")
    df = df.drop(columns=["SEQN"])

    train, test = train_test_split(df, test_size=0.20, random_state=42, shuffle=True, stratify=df["event"])
    train = train.sort_values("patient_id").reset_index(drop=True)
    ids = package.opaque_ids(test["patient_id"].tolist(), "nh")
    test = test.assign(patient_id=test["patient_id"].map(ids)).sort_values("patient_id").reset_index(drop=True)
    package.survival(out, "nhanes", HERE / "nhanes.md", train, test)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", default="packages")
    main(**vars(ap.parse_args()))
