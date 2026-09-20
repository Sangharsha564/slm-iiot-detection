#!/usr/bin/env python3
"""
Usage:
    python src/select_explainer_sample.py --run experiments/stage1_full_13_readable192/bert-mini_s42
    python src/select_explainer_sample.py --run <dir> -n 20 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_configs, write_json


def zclip(a, lo=-3.0, hi=3.0):
    """z-score then clip, so one extreme flow cannot dominate the geometry."""
    a = np.asarray(a, dtype=float)
    s = a.std()
    if s == 0:
        return np.zeros_like(a)
    return np.clip((a - a.mean()) / s, lo, hi)


def farthest_point(X, k, seeds):
    chosen = list(seeds)
    d = np.full(len(X), np.inf)
    for c in chosen:
        d = np.minimum(d, np.linalg.norm(X - X[c], axis=1))
    d[chosen] = -1.0
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        if d[nxt] <= 0:                      
            remaining = [i for i in range(len(X)) if i not in set(chosen)]
            if not remaining:
                break
            nxt = remaining[0]
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
        d[chosen] = -1.0
    return chosen


def coverage(df, labels, idx):
    sub = df.iloc[idx]
    A = sub[[f"attr_{l}" for l in labels]].to_numpy()
    o = np.argsort(-np.abs(A), axis=1)[:, :3]
    pat, val, dom = set(), set(), set()
    for r in range(len(sub)):
        names = tuple(labels[j] for j in o[r])
        pat.add(names)
        vals = tuple(f"{sub.iloc[r][f'val_{labels[j]}']:g}" for j in o[r])
        val.add(names + vals)
        dom.add(f"{sub.iloc[r][f'val_{labels[o[r][0]]}']:g}")
    return {
        "scenarios": int(sub.scenario.nunique()),
        "rank_patterns": len(pat),
        "full_prompt_patterns": len(val),
        "dominant_values": len(dom),
        "prob_spread": float(sub.prob.max() - sub.prob.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--exp-config", default="experiment_readable192.yaml")
    ap.add_argument("--xai-dir", default="results/07_xai")
    ap.add_argument("--out", default="results/08_stage2")
    ap.add_argument("-n", "--n-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-trials", type=int, default=200,
                    help="random draws to compare the selection against")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    xai = root / args.xai_dir
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    _, exp_cfg = load_configs(root, args.exp_config)
    data_dir = root / exp_cfg["data"]["processed_dir"]

    ig = pd.read_csv(xai / "ig_flows.csv")
    ig = ig[ig.outcome == "detected"].reset_index(drop=True)
    labels = [c[5:] for c in ig.columns if c.startswith("attr_")]

    test = pd.read_csv(data_dir / exp_cfg["data"]["test_file"])
    ig["text"] = ig.flow_key.map(test.set_index("flow_key").text)
    if ig.text.isna().any():
        raise SystemExit("some flows have no matching text in the test set")

    print(f"pool      {len(ig)} detected flows, {ig.scenario.nunique()} scenarios")

    
    A = ig[[f"attr_{l}" for l in labels]].to_numpy()
    order = np.argsort(-np.abs(A), axis=1)
    top3 = order[:, :3]

    onehot = np.zeros((len(ig), 3 * len(labels)))
    for r in range(len(ig)):
        for rank, j in enumerate(top3[r]):
            onehot[r, rank * len(labels) + j] = 1.0

    shares = np.stack([[ig.iloc[r][f"share_{labels[j]}"] for j in top3[r]]
                       for r in range(len(ig))])
    vals = np.stack([[ig.iloc[r][f"val_{labels[j]}"] for j in top3[r]]
                     for r in range(len(ig))])
    vals_z = np.stack([zclip(vals[:, c]) for c in range(3)], axis=1)
    prob_z = zclip(ig.prob.to_numpy()).reshape(-1, 1)

    X = np.hstack([onehot, shares, vals_z, prob_z])
    print(f"space     {X.shape[1]} dimensions "
          f"({3*len(labels)} rank one-hot + 3 shares + 3 values + 1 confidence)")

    seeds = []
    for sc, g in ig.groupby("scenario"):
        idx = g.index.to_numpy()
        centre = X[idx].mean(0)
        seeds.append(int(idx[np.argmin(np.linalg.norm(X[idx] - centre, axis=1))]))
    print(f"seeds     {len(seeds)} (one per scenario, nearest that scenario's centre)")

    chosen = farthest_point(X, args.n_samples, seeds)
    sel = ig.iloc[chosen].copy().reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    cov_sel = coverage(ig, labels, chosen)
    rand = [coverage(ig, labels,
                     list(rng.choice(len(ig), args.n_samples, replace=False)))
            for _ in range(args.random_trials)]
    rand_mean = {k: float(np.mean([r[k] for r in rand])) for k in cov_sel}
    rand_max = {k: float(np.max([r[k] for r in rand])) for k in cov_sel}

    print(f"\n  coverage of the {args.n_samples} selected flows")
    print(f"    {'measure':24s} {'selected':>9s} {'random avg':>11s} {'random best':>12s}")
    print("    " + "-" * 60)
    for k in cov_sel:
        v = cov_sel[k]
        fmt = f"{v:9.3f}" if isinstance(v, float) else f"{v:9d}"
        print(f"    {k:24s} {fmt} {rand_mean[k]:11.2f} {rand_max[k]:12.2f}")

    better = sum(1 for k in cov_sel if cov_sel[k] > rand_mean[k])
    print(f"\n    the selection beats the random average on {better} of "
          f"{len(cov_sel)} measures")
    if better <= 1:
        print(f"    -> the pool is too homogeneous for selection to help. "
              f"Report this: a random sample would have been just as good, "
              f"which is a fact about the data, not a flaw in the method.")

    rows = []
    for r in range(len(sel)):
        a = sel.iloc[r][[f"attr_{l}" for l in labels]].to_numpy(dtype=float)
        o = np.argsort(-np.abs(a))[:3]
        rows.append({
            "n": r + 1,
            "flow_key": str(sel.iloc[r].flow_key)[:12],
            "scenario": sel.iloc[r].scenario,
            "prob": sel.iloc[r].prob,
            "f1": labels[o[0]], "v1": sel.iloc[r][f"val_{labels[o[0]]}"],
            "s1": sel.iloc[r][f"share_{labels[o[0]]}"],
            "f2": labels[o[1]], "v2": sel.iloc[r][f"val_{labels[o[1]]}"],
            "f3": labels[o[2]], "v3": sel.iloc[r][f"val_{labels[o[2]]}"],
        })
    summary = pd.DataFrame(rows)

    print(f"\n  the {len(sel)} selected flows\n")
    print(f"  {'#':>2s} {'scenario':24s} {'conf':>7s} {'top feature':26s} "
          f"{'val':>8s} {'share':>6s}  {'2nd':22s} {'3rd':22s}")
    print("  " + "-" * 124)
    for _, r in summary.iterrows():
        print(f"  {r.n:2d} {r.scenario:24s} {r.prob:7.4f} {r.f1:26s} "
              f"{r.v1:8g} {100*r.s1:5.1f}% "
              f" {r.f2[:20]:22s} {r.f3[:20]:22s}")

    print(f"\n  scenario counts: " +
          "  ".join(f"{k}={v}" for k, v in sel.scenario.value_counts().items()))

    sel.to_csv(out_dir / "explainer_sample.csv", index=False)
    summary.to_csv(out_dir / "explainer_sample_summary.csv", index=False)
    write_json(out_dir / "explainer_sample.json", {
        "n_samples": args.n_samples, "seed": args.seed,
        "pool": "detected flows only", "pool_size": len(ig),
        "method": "farthest-point sampling over prompt-relevant features, "
                  "seeded with one flow per scenario",
        "coverage_selected": cov_sel,
        "coverage_random_mean": rand_mean,
        "coverage_random_best": rand_max,
        "random_trials": args.random_trials,
        "flow_keys": [str(k) for k in sel.flow_key],
        "scenario_counts": {k: int(v) for k, v in
                            sel.scenario.value_counts().items()},
    })
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
