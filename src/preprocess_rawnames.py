#!/usr/bin/env python3
"""
Usage:
    python src/preprocess_rawnames.py .
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

SEED = 42
CORR_THRESHOLD = 0.75
NEAR_CONSTANT_SHARE = 0.99
VAL_FRACTION = 0.25

META = {"label", "device", "scenario", "source", "chunk", "flow_key"}
SRC_DIR = "data/processed/v3"
OUT_DIR = "data/processed/v3_rawnames"


def split_train_val(df):
    baseline = df["source"] == "benign_capture"
    parts_tr, parts_va = [], []

    sub = df[~baseline]
    if len(sub):
        strat = sub["scenario"].astype(str) + "|" + sub["label"].astype(str)
        vc = strat.value_counts()
        strat = strat.where(strat.map(vc) >= 2, "rare")
        tr, va = train_test_split(sub, test_size=VAL_FRACTION,
                                  random_state=SEED, stratify=strat)
        parts_tr.append(tr); parts_va.append(va)

    sub = df[baseline]
    if len(sub):
        if "chunk" in sub.columns:
            w = np.sort(sub["chunk"].unique())
            cut = w[int(len(w) * (1 - VAL_FRACTION))]
            parts_tr.append(sub[sub["chunk"] < cut])
            parts_va.append(sub[sub["chunk"] >= cut])
        else:
            tr, va = train_test_split(sub, test_size=VAL_FRACTION,
                                      random_state=SEED)
            parts_tr.append(tr); parts_va.append(va)

    return (pd.concat(parts_tr).reset_index(drop=True),
            pd.concat(parts_va).reset_index(drop=True))


def serialise_raw(df, feats):
    def fmt(v):
        return str(int(v)) if float(v).is_integer() else f"{v:.2f}"
    out = []
    for _, row in df[feats].iterrows():
        out.append(" | ".join(f"{c} is {fmt(row[c])}" for c in feats))
    return pd.Series(out, index=df.index)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    src = root / SRC_DIR
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    spec = json.loads((src / "selected_features.json").read_text())
    selected = spec["selected"]
    print(f"reusing {len(selected)} features from the original selection")

    full_train = pd.read_csv(src / "train.csv")
    test = pd.read_csv(src / "test.csv")
    train, val = split_train_val(full_train)

    for name, d in [("train", train), ("val", val), ("test", test)]:
        missing = [c for c in selected if c not in d.columns]
        if missing:
            sys.exit(f"{name} is missing columns: {missing}")
        frame = pd.DataFrame({
            "text": serialise_raw(d, selected),
            "label": d["label"].values,
            "device": d["device"].values,
            "scenario": d["scenario"].values,
        })
        frame.to_csv(out / f"{name}_text.csv", index=False)
        print(f"  {name}_text.csv  {len(frame)} rows")

    shutil.copy(src / "selected_features.json", out / "selected_features.json")

    ex = pd.read_csv(out / "train_text.csv")
    a = ex.loc[ex.label == 1, "text"].iloc[0]
    b = ex.loc[ex.label == 0, "text"].iloc[0]
    print(f"\n  attack example:\n    {a}")
    print(f"\n  benign example:\n    {b}")

    lens = ex["text"].str.split().str.len()
    print(f"\n  whitespace tokens: mean={lens.mean():.0f}  max={lens.max()}")
    print("  (compare against 80 for the readable serialisation)")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()