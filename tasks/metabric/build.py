"""METABRIC: 1,904 breast-cancer patients, time to death (pycox distribution).

    python tasks/metabric/build.py RAW_DIR [--out packages]     RAW_DIR holds metabric.feather
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from automdbench import package

HERE = Path(__file__).resolve().parent


def main(raw, out):
    df = pd.read_feather(Path(raw) / "metabric.feather").reset_index(drop=True)
    df.insert(0, "patient_id", df.index.astype("int64"))
    train, test = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)
    train = train.sort_values("patient_id").reset_index(drop=True)
    ids = package.opaque_ids(test["patient_id"].tolist(), "m")
    test = test.assign(patient_id=test["patient_id"].map(ids)).sort_values("patient_id").reset_index(drop=True)
    package.survival(out, "metabric", HERE / "metabric.md", train, test)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", default="packages")
    main(**vars(ap.parse_args()))
