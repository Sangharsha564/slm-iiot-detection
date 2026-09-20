#!/usr/bin/env python3
"""
Usage:
    python src/make_demo_flows.py
    python src/make_demo_flows.py --n-filler 80        # shorter run
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--test", default="data/processed/v3/test_text.csv",
                    help="must be the readable serialisation with flow_key: "
                         "the v3_rawnames variant uses different feature "
                         "names and the detector was not trained on it")
    ap.add_argument("--sample", default="results/08_stage2/explainer_sample.csv")
    ap.add_argument("--n-filler", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="demo_flows.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    test = pd.read_csv(root / args.test)
    sel = pd.read_csv(root / args.sample)

    for c in ("text", "flow_key"):
        if c not in test.columns:
            raise SystemExit(
                f"{args.test} has no '{c}' column. Columns: "
                f"{list(test.columns)}. This must be the v3 readable file.")

    keys = set(sel.flow_key.astype(str))
    test["flow_key"] = test.flow_key.astype(str)
    chosen = test[test.flow_key.isin(keys)]
    if len(chosen) != len(keys):
        missing = keys - set(chosen.flow_key)
        raise SystemExit(
            f"{len(missing)} of the 20 selected flows are not in {args.test}: "
            f"{list(missing)[:3]}. Wrong test file?")

    rest = test[~test.flow_key.isin(keys)]
    filler = rest.sample(n=min(args.n_filler, len(rest)),
                         random_state=args.seed)

    demo = (pd.concat([chosen, filler], ignore_index=True)
              .sample(frac=1, random_state=args.seed)
              .reset_index(drop=True))
    demo.to_csv(root / args.out, index=False)

    n_att = int((demo.label == 1).sum())
    print(f"  selected alerts   {len(chosen)}")
    print(f"  filler flows      {len(filler)}  (random from the rest of test)")
    print(f"  total             {len(demo)}   attack-labelled {n_att} "
          f"({100*n_att/len(demo):.0f}%)")
    print(f"\n  written {root/args.out}")
    print(f"\n  More than 20 will alert -- the filler contains attack flows "
          f"too, and\n  the detector scores them on their merits. Expect "
          f"roughly {n_att} alerts and\n  budget about 10 s each on the "
          f"laptop. Use --n-filler to shorten it.")
    print(f"\n  Copy it over:")
    print(f"    scp {args.out} sangharsha@192.168.1.42:~/stage1/")


if __name__ == "__main__":
    main()
