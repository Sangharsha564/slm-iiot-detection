#!/usr/bin/env python3
"""
Preprocessing and feature selection for the Slowloris dataset.

Pipeline (all measurement on TRAIN only; validation and test receive the
resulting column list, nothing is measured on them):

  0. Split the existing train.csv into train / validation
       - attack: stratified within scenario, so both edge1 ports appear on
         each side
       - benign from attack captures: stratified the same way
       - benign from the segmented baseline: split by TIME WINDOW, so no
         62 s window contributes to both sides
  1. Drop near-constant features (>= 99% one value in train)
  2. Correlation clustering (Spearman, rho >= 0.75), keep the highest-MI
     member of each cluster
  3. Mutual information ranking (all survivors kept; ranking determines
     serialisation order, strongest evidence first)
  4. Serialise to pipe-separated text for the encoder

No scaling and no log transform: Stage 1 consumes text, so feature
magnitude does not affect input geometry, and raw values stay readable in
Stage 2 explanations.

No class balancing: ratio is ~1:3.9, handled by class weights in the loss.

Usage:
    python preprocess.py ./dataset_out_v3
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

SEED = 42
K_FEATURES = None      # None = keep every cluster survivor
CORR_THRESHOLD = 0.75          # cluster distance 1 - rho
NEAR_CONSTANT_SHARE = 0.99
VAL_FRACTION = 0.25

META = {"label", "device", "scenario", "source", "chunk", "flow_key"}

# Units for readable serialisation. ms -> s for time, everything else raw.
MS_COLUMNS_SUFFIX = "_ms"

MECHANISM_PATTERNS = [
    ("server response", ["dst2src_psh", "dst2src_ack", "dst2src_packets",
                         "dst2src_bytes", "dst2src_rst"]),
    ("client volume",   ["src2dst_packets", "src2dst_bytes", "src2dst_psh",
                         "src2dst_ack"]),
    ("packet size",     ["_ps"]),
    ("inter-arrival",   ["_piat_"]),
    ("duration",        ["duration"]),
    ("total volume",    ["bidirectional_packets", "bidirectional_bytes",
                         "bidirectional_ack", "bidirectional_psh"]),
]


def mechanism_of(col):
    for name, pats in MECHANISM_PATTERNS:
        if any(p in col for p in pats):
            return name
    return "other"


def pretty_name(col):
    """Readable label for serialisation."""
    s = col
    s = s.replace("bidirectional_", "total ")
    s = s.replace("src2dst_", "client ")
    s = s.replace("dst2src_", "server ")
    s = s.replace("_piat_ms", " inter-arrival")
    s = s.replace("_ms", "")
    s = s.replace("_ps", " packet size")
    s = s.replace("_packets", " packets")
    s = s.replace("_bytes", " bytes")
    s = s.replace("stddev", "stddev ")
    s = s.replace("_", " ")
    return " ".join(s.split())


# ------------------------------------------------------------------ split ---

def split_train_val(df):
    """Split into train / validation without letting a benign window span both."""
    baseline = df["source"] == "benign_capture"
    from_attack = ~baseline

    parts_tr, parts_va = [], []

    # attack-capture rows (both attack and benign): stratify by scenario+label
    sub = df[from_attack]
    if len(sub):
        strat = sub["scenario"].astype(str) + "|" + sub["label"].astype(str)
        vc = strat.value_counts()
        strat = strat.where(strat.map(vc) >= 2, "rare")
        tr, va = train_test_split(sub, test_size=VAL_FRACTION,
                                  random_state=SEED, stratify=strat)
        parts_tr.append(tr)
        parts_va.append(va)

    # baseline benign: split by time window when available
    sub = df[baseline]
    if len(sub):
        if "chunk" in sub.columns:
            windows = np.sort(sub["chunk"].unique())
            cut = windows[int(len(windows) * (1 - VAL_FRACTION))]
            parts_tr.append(sub[sub["chunk"] < cut])
            parts_va.append(sub[sub["chunk"] >= cut])
        else:
            print("  WARNING: no 'chunk' column - baseline benign split "
                  "randomly instead of by time window. Flows from the same "
                  "62 s window may appear in both train and validation. "
                  "Add 'chunk' to keep_cols in the builder and rebuild "
                  "for a window-disjoint split.")
            tr, va = train_test_split(sub, test_size=VAL_FRACTION,
                                      random_state=SEED)
            parts_tr.append(tr)
            parts_va.append(va)

    return (pd.concat(parts_tr).reset_index(drop=True),
            pd.concat(parts_va).reset_index(drop=True))


# -------------------------------------------------------------- selection ---

def drop_near_constant(train, feats):
    kept, dropped = [], []
    for c in feats:
        share = train[c].value_counts(normalize=True, dropna=False).iloc[0]
        if share >= NEAR_CONSTANT_SHARE:
            dropped.append((c, round(float(share), 4)))
        else:
            kept.append(c)
    return kept, dropped


def mi_scores(train, feats):
    x = train[feats].fillna(0).values
    y = train["label"].values
    scores = mutual_info_classif(x, y, random_state=SEED)
    return pd.Series(scores, index=feats).sort_values(ascending=False)


def cluster_features(train, feats, mi):
    corr = train[feats].corr(method="spearman").abs().fillna(0)
    dist = squareform(np.clip(1 - corr.values, 0, None), checks=False)
    labels = fcluster(linkage(dist, method="average"),
                      t=1 - CORR_THRESHOLD, criterion="distance")

    table = pd.DataFrame({"feature": feats, "cluster": labels})
    table["mi"] = table["feature"].map(mi)
    table["mechanism"] = table["feature"].map(mechanism_of)

    reps = (table.sort_values("mi", ascending=False)
            .groupby("cluster", as_index=False).first())
    return table, reps


# ------------------------------------------------------------ serialising ---

def serialise(df, feats):
    """Pipe-separated text. Milliseconds rendered as seconds for readability."""
    def fmt(col, v):
        if col.endswith(MS_COLUMNS_SUFFIX):
            return f"{v / 1000:.1f}s"
        if float(v).is_integer():
            return str(int(v))
        return f"{v:.2f}"

    names = [pretty_name(c) for c in feats]
    out = []
    for _, row in df[feats].iterrows():
        out.append(" | ".join(f"{n} is {fmt(c, row[c])}"
                              for n, c in zip(names, feats)))
    return pd.Series(out, index=df.index)


# ------------------------------------------------------------------- main ---

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "./dataset_out_v3")
    full_train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")

    log = []
    def w(s=""):
        print(s)
        log.append(str(s))

    w("=" * 74)
    w("PREPROCESSING AND FEATURE SELECTION")
    w("=" * 74)
    w(f"seed={SEED}  target features={K_FEATURES}  "
      f"corr threshold={CORR_THRESHOLD}")
    w()

    # ---- step 0: split --------------------------------------------------
    train, val = split_train_val(full_train)
    w("0. TRAIN / VALIDATION SPLIT")
    for name, d in [("train", train), ("val", val), ("test", test)]:
        n1, n0 = int((d.label == 1).sum()), int((d.label == 0).sum())
        w(f"   {name:5s}  attack={n1:6d}  benign={n0:6d}  "
          f"ratio=1:{n0 / max(n1, 1):.2f}")
    if "chunk" in train.columns:
        tr_win = set(train.loc[train.source == "benign_capture", "chunk"])
        va_win = set(val.loc[val.source == "benign_capture", "chunk"])
        shared = tr_win & va_win
        w(f"   benign windows shared between train and val: "
          f"{len(shared)}  ({'PASS' if not shared else 'FAIL'})")
    else:
        w("   benign window split: NOT AVAILABLE ('chunk' column absent)")
        w("   baseline benign was split randomly - see warning above")
    w()

    feats = [c for c in train.columns
             if c not in META and pd.api.types.is_numeric_dtype(train[c])]
    w(f"starting features: {len(feats)}")
    w()

    # ---- step 1: near-constant ------------------------------------------
    feats, dropped = drop_near_constant(train, feats)
    w("1. NEAR-CONSTANT REMOVAL")
    for c, share in dropped:
        w(f"   dropped {c:32s} {share:.1%} one value")
    w(f"   remaining: {len(feats)}")
    w()

    # ---- step 2: correlation clustering ---------------------------------
    mi = mi_scores(train, feats)
    table, reps = cluster_features(train, feats, mi)
    w(f"2. CORRELATION CLUSTERING  (Spearman, rho >= {CORR_THRESHOLD})")
    for cid, grp in table.groupby("cluster"):
        grp = grp.sort_values("mi", ascending=False)
        rep = grp.iloc[0]["feature"]
        if len(grp) == 1:
            w(f"   cluster {cid:2d}  singleton: {rep}")
        else:
            others = [f for f in grp["feature"] if f != rep]
            w(f"   cluster {cid:2d}  keep {rep}  "
              f"(mi={grp.iloc[0]['mi']:.4f})")
            w(f"              drop {', '.join(others)}")
    survivors = reps["feature"].tolist()
    w(f"   remaining: {len(survivors)}")
    w()

    # ---- step 3: MI ranking ---------------------------------------------
    ranked = mi[survivors].sort_values(ascending=False)
    selected = (ranked.index.tolist() if K_FEATURES is None
                else ranked.head(K_FEATURES).index.tolist())
    w(f"3. MUTUAL INFORMATION RANKING  "
      f"({'all survivors kept' if K_FEATURES is None else f'top {K_FEATURES}'})")
    for i, (f, sc) in enumerate(ranked.items(), 1):
        mark = "*" if f in selected else " "
        w(f"  {mark} {i:2d}. {f:32s} mi={sc:.4f}  [{mechanism_of(f)}]")
    w(f"   final feature count: {len(selected)}")
    w()

    w("   SELECTED FEATURES BY MECHANISM")
    for mech in sorted({mechanism_of(f) for f in selected}):
        members = [f for f in selected if mechanism_of(f) == mech]
        w(f"     {mech:16s} {', '.join(members)}")
    w()

    # ---- step 4: serialise ----------------------------------------------
    w("4. SERIALISATION")
    outputs = {}
    for name, d in [("train", train), ("val", val), ("test", test)]:
        cols = {
            "text": serialise(d, selected),
            "label": d["label"].values,
            "device": d["device"].values,
            "scenario": d["scenario"].values,
        }
        if "flow_key" in d.columns:
            cols["flow_key"] = d["flow_key"].values
        out = pd.DataFrame(cols)
        out.to_csv(root / f"{name}_text.csv", index=False)
        outputs[name] = out
        w(f"   {name}_text.csv  {len(out)} rows")
    w()
    w("   example attack row:")
    ex = outputs["train"].loc[outputs["train"].label == 1, "text"]
    w(f"     {ex.iloc[0] if len(ex) else '(none)'}")
    w("   example benign row:")
    ex = outputs["train"].loc[outputs["train"].label == 0, "text"]
    w(f"     {ex.iloc[0] if len(ex) else '(none)'}")
    w()

    lens = outputs["train"]["text"].str.split().str.len()
    w(f"   whitespace tokens: mean={lens.mean():.0f}  max={lens.max()}")
    w("   (check against the encoder tokenizer before training)")
    w()

    # ---- artefacts -------------------------------------------------------
    (root / "selected_features.json").write_text(json.dumps({
        "seed": SEED,
        "corr_threshold": CORR_THRESHOLD,
        "k": K_FEATURES if K_FEATURES is not None else len(selected),
        "selection_rule": "all cluster survivors, ordered by mutual information",
        "selected": selected,
        "pretty_names": {f: pretty_name(f) for f in selected},
        "mechanisms": {f: mechanism_of(f) for f in selected},
        "mi_scores": {f: float(mi[f]) for f in selected},
    }, indent=2))

    table.to_csv(root / "cluster_assignment.csv", index=False)
    (root / "selection_report.txt").write_text("\n".join(log))

    # class weight for the loss
    n1 = int((train.label == 1).sum())
    n0 = int((train.label == 0).sum())
    w(f"class weight for attack class: {n0 / n1:.3f}")
    w(f"\nWritten to {root.resolve()}/")


if __name__ == "__main__":
    main()
