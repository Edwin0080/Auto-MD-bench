"""SUPPORT: 8,873 seriously ill hospitalized adults, time to death (pycox distribution).

    python tasks/support/build.py RAW_DIR [--out packages]      RAW_DIR holds support.feather
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from automdbench import package

HERE = Path(__file__).resolve().parent


def main(raw, out):
    df = pd.read_feather(Path(raw) / "support.feather").reset_index(drop=True)
    df.insert(0, "patient_id", df.index.astype("int64"))
    df = df[["patient_id"] + [f"x{i}" for i in range(14)] + ["duration", "event"]]
    train, test = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)
    train = train.sort_values("patient_id").reset_index(drop=True)
    test = test.sort_values("patient_id").reset_index(drop=True)
    ids = package.opaque_ids(test["patient_id"].tolist(), "t")
    test = test.assign(patient_id=test["patient_id"].map(ids)).sort_values("patient_id").reset_index(drop=True)
    package.survival(out, "support", HERE / "support.md", train, test)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", default="packages")
    main(**vars(ap.parse_args()))
