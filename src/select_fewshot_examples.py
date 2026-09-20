#!/usr/bin/env python3
"""
Usage:
    python src/select_fewshot_examples.py --run experiments/stage1_full_13_readable192/bert-mini_s42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_configs
from stage2_generate import build_evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--exp-config", default="experiment_readable192.yaml")
    ap.add_argument("--xai-dir", default="results/07_xai")
    ap.add_argument("--dir", default="results/08_stage2")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--dominant", type=float, default=0.85,
                    help="example A: minimum share of the top factor")
    ap.add_argument("--spread", type=float, default=0.60,
                    help="example B: maximum share of the top factor")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    xai = root / args.xai_dir
    d = root / args.dir

    _, exp_cfg = load_configs(root, args.exp_config)
    data_dir = root / exp_cfg["data"]["processed_dir"]

    ig = pd.read_csv(xai / "ig_flows.csv")
    ig = ig[ig.outcome == "detected"].reset_index(drop=True)
    baseline = json.loads((xai / "ig_baseline.json").read_text())["baseline"]
    labels = [c[5:] for c in ig.columns if c.startswith("attr_")]

    test = pd.read_csv(data_dir / exp_cfg["data"]["test_file"])
    ig["text"] = ig.flow_key.map(test.set_index("flow_key").text)

    used = set(pd.read_csv(d / "explainer_sample.csv").flow_key.astype(str))
    pool = ig[~ig.flow_key.astype(str).isin(used)].reset_index(drop=True)
    print(f"pool      {len(ig)} detected, {len(used)} excluded as evaluation "
          f"alerts, {len(pool)} candidates")

    A = pool[[f"attr_{l}" for l in labels]].to_numpy()
    order = np.argsort(-np.abs(A), axis=1)

    rows = []
    for i in range(len(pool)):
        top = order[i, :args.top_k]
        shares = [float(pool.iloc[i][f"share_{labels[j]}"]) for j in top]
        signs = [float(A[i, j]) for j in top]
        rows.append({"i": i, "top_share": shares[0],
                     "n_lowers": sum(1 for s in signs if s < 0),
                     "shares": shares})
    info = pd.DataFrame(rows)

    print(f"          top-share >= {args.dominant}: "
          f"{int((info.top_share >= args.dominant).sum())} flows   "
          f"top-share < {args.spread}: "
          f"{int((info.top_share < args.spread).sum())} flows   "
          f"with a LOWERS factor: {int((info.n_lowers >= 1).sum())} flows")

    cands = {
        "A  single dominant factor":
            info[info.top_share >= args.dominant],
        "B  combination with a LOWERS factor":
            info[(info.top_share < args.spread) & (info.n_lowers >= 1)],
    }

    picks = []
    for name, c in cands.items():
        print(f"\n{'='*72}\nEXAMPLE {name}")
        print(f"{'='*72}")
        if c.empty:
            print(f"  NO CANDIDATE. Relax --dominant / --spread and re-run.")
            continue
        target = c.top_share.median()
        pick = c.iloc[(c.top_share - target).abs().argsort().iloc[0]]
        i = int(pick.i)
        picks.append(i)
        r = pool.iloc[i]
        print(f"  {len(c)} candidates; chose the one nearest the median "
              f"top-share ({target:.2f})")
        print(f"  flow_key {r.flow_key}   scenario {r.scenario}   "
              f"confidence {r.prob:.4f}   "
              f"{int(pick.n_lowers)} mitigating factor(s) in the top three")
        print(f"\n{build_evidence(r, labels, baseline, args.top_k)}")

    if len(picks) == 2 and picks[0] == picks[1]:
        print("\n  WARNING: both categories chose the same flow.")

    print(f"\n{'='*72}")


if __name__ == "__main__":
    main()