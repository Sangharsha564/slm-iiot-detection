#!/usr/bin/env python3
"""
Combine train, validation and test into one file for device benchmarking.

The purpose is a longer, more stable latency measurement than the 4,643-flow
test set alone provides. Accuracy must still be reported per split: training
flows were seen during fitting and score far higher than held-out flows, so a
combined accuracy figure would describe neither.

A `split` column is therefore retained so the benchmark can report per-split
metrics alongside the combined timing. The train-versus-test gap measured on
the deployment device is itself informative.

Usage:
    python src/make_combined.py
    python src/make_combined.py --processed data/processed/v3_rawnames \\
                               --out combined_rawnames.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed/v3",
                    help="directory holding train/val/test _text.csv")
    ap.add_argument("--out", default=None,
                    help="output filename; defaults to combined_text.csv "
                         "inside the processed directory")
    ap.add_argument("--shuffle", action="store_true",
                    help="interleave splits. Off by default: leaving them "
                         "contiguous lets a partial run still cover a whole "
                         "split if it is interrupted.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = Path(args.processed)
    out = Path(args.out) if args.out else d / "combined_text.csv"

    frames = []
    for split in ["train", "val", "test"]:
        f = d / f"{split}_text.csv"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        df = pd.read_csv(f)
        df.insert(0, "split", split)
        frames.append(df)
        print(f"  {split:5s}  {len(df):6d} flows  "
              f"({int(df.label.sum())} attack, "
              f"{int((df.label == 0).sum())} benign)")

    combined = pd.concat(frames, ignore_index=True)
    if args.shuffle:
        combined = combined.sample(frac=1, random_state=args.seed) \
                           .reset_index(drop=True)
        print("\n  shuffled")

    combined.to_csv(out, index=False)

    print(f"\n  total  {len(combined):6d} flows  "
          f"({int(combined.label.sum())} attack, "
          f"{int((combined.label == 0).sum())} benign)")
    print(f"  columns: {', '.join(combined.columns)}")
    print(f"\nwritten to {out}")

    print("\nnote: report accuracy per split, not combined. Training flows "
          "were seen during fitting.")


if __name__ == "__main__":
    main()