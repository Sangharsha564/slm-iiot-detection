#!/usr/bin/env python3

"""
Usage:
    python src/xai_analyse.py
    python src/xai_analyse.py --dir results/07_xai
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bar(frac, width=28):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "#" * n + "." * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/07_xai")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    d = Path(args.root).resolve() / args.dir
    df = pd.read_csv(d / "ig_flows.csv")
    base = json.loads((d / "ig_baseline.json").read_text())
    summary = json.loads((d / "summary.json").read_text())

    labels = [c[5:] for c in df.columns if c.startswith("attr_")]
    A = df[[f"attr_{l}" for l in labels]].to_numpy()
    absA = np.abs(A)
    top = summary["top_feature_overall"]

    pd.set_option("display.width", 200)
    print(f"flows {len(df)}   steps {summary['ig_steps']}   "
          f"completeness {summary['completeness_error']['median']:.2e}\n")

    print("=" * 78)
    print("1  HOW CONCENTRATED IS THE DECISION")
    print("=" * 78)
    share_sorted = np.sort(absA / absA.sum(1, keepdims=True), axis=1)[:, ::-1]
    for k in (1, 2, 3, 5):
        v = share_sorted[:, :k].sum(1)
        print(f"  top-{k} feature(s) carry {100*v.mean():5.1f}% of |attribution|   "
              f"(median {100*np.median(v):5.1f}%)")
    p = absA / absA.sum(1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
    print(f"\n  effective number of contributing features: "
          f"{np.exp(ent).mean():.2f} of {len(labels)}")
    print(f"    (exp of attribution entropy: {len(labels):.0f} = all equal, "
          f"1.0 = single feature)")
    print("\n" + "=" * 78)
    print("2  DETECTED vs MISSED")
    print("=" * 78)
    det = df[df.outcome == "detected"]
    mis = df[df.outcome == "missed"]
    print(f"  detected {len(det)}   missed {len(mis)}")
    print(f"  mean log-odds vs baseline:  detected {det.logodds_delta.mean():+8.3f}"
          f"   missed {mis.logodds_delta.mean():+8.3f}")
    print(f"\n  {'feature':30s} {'attr det':>9s} {'attr mis':>9s} {'diff':>9s}"
          f" {'val det':>10s} {'val mis':>10s}")
    print("  " + "-" * 82)
    rows = []
    for l in labels:
        rows.append((l, det[f"attr_{l}"].mean(), mis[f"attr_{l}"].mean(),
                     det[f"val_{l}"].median(), mis[f"val_{l}"].median()))
    for l, ad, am, vd, vm in sorted(rows, key=lambda r: -abs(r[1] - r[2])):
        print(f"  {l:30s} {ad:9.3f} {am:9.3f} {ad-am:9.3f} {vd:10.2f} {vm:10.2f}")
    print("\n" + "=" * 78)
    print(f"3  {top.upper()}: ATTRIBUTION vs VALUE")
    print("=" * 78)
    b = base["baseline"].get(top, "?")
    print(f"  benign baseline value: {b}")
    sub = df[[f"val_{top}", f"attr_{top}", "prob", "outcome"]].copy()
    sub.columns = ["val", "attr", "prob", "outcome"]
    try:
        sub["bin"] = pd.qcut(sub.val, 10, duplicates="drop")
    except ValueError:
        sub["bin"] = pd.cut(sub.val, 10)
    g = sub.groupby("bin", observed=True).agg(
        n=("val", "size"), val_lo=("val", "min"), val_hi=("val", "max"),
        attr=("attr", "mean"), prob=("prob", "mean"),
        missed=("outcome", lambda s: (s == "missed").mean() * 100))
    print(f"\n  {'value range':>20s} {'n':>5s} {'mean attr':>10s} "
          f"{'mean prob':>10s} {'missed%':>8s}")
    print("  " + "-" * 60)
    for _, r in g.iterrows():
        print(f"  {r.val_lo:9.1f} - {r.val_hi:8.1f} {int(r.n):5d} "
              f"{r.attr:10.3f} {r.prob:10.4f} {r.missed:7.1f}%")
    c = np.corrcoef(sub.val, sub.attr)[0, 1]
    print(f"\n  corr(value, attribution) = {c:+.3f}")
    print("    near 0 with a wide value range -> the model is not reading the")
    print("    magnitude, it is reacting to the token pattern itself")
    print("\n" + "=" * 78)
    print("4  DIRECTION: attribution sign vs value relative to benign baseline")
    print("=" * 78)
    print(f"  {'feature':30s} {'base':>9s} {'atk med':>9s} {'dir':>5s} "
          f"{'pos%':>6s} {'mean attr':>10s}")
    print("  " + "-" * 74)
    for l in labels:
        try:
            bv = float(str(base["baseline"][l]).rstrip("s"))
        except (KeyError, ValueError):
            continue
        av = df[f"val_{l}"].median()
        pos = (df[f"attr_{l}"] > 0).mean() * 100
        direction = "up" if av > bv else ("down" if av < bv else "same")
        print(f"  {l:30s} {bv:9.2f} {av:9.2f} {direction:>5s} {pos:5.1f}% "
              f"{df[f'attr_{l}'].mean():10.3f}")

    print("\n" + "=" * 78)
    print("5  FLOWS WHERE SOMETHING ELSE WON")
    print("=" * 78)
    winner = pd.Series([labels[i] for i in A.argmax(1)], index=df.index)
    exc = df[winner != top].copy()
    exc["winner"] = winner[winner != top]
    print(f"  {len(exc)} of {len(df)} flows ({100*len(exc)/len(df):.1f}%)")
    if len(exc):
        print(f"\n  by winning feature:")
        for w, n in exc.winner.value_counts().items():
            print(f"    {w:32s} {n:4d}")
        print(f"\n  by scenario:")
        for s, n in exc.scenario.value_counts().items():
            print(f"    {s:32s} {n:4d}  "
                  f"({100*n/(df.scenario == s).sum():.1f}% of that scenario)")
        print(f"\n  outcome: " + "  ".join(
            f"{k}={v}" for k, v in exc.outcome.value_counts().items()))
        print(f"  these are the flows a one-feature explanation would get wrong")

    print("\n" + "=" * 78)
    print("6  PER SCENARIO")
    print("=" * 78)
    print(f"  {'scenario':28s} {'n':>5s} {'miss%':>7s} {'top feature':30s} {'share':>7s}")
    print("  " + "-" * 82)
    for s, g in df.groupby("scenario"):
        G = np.abs(g[[f"attr_{l}" for l in labels]].to_numpy())
        sh = G.sum(0) / G.sum()
        j = int(sh.argmax())
        print(f"  {s:28s} {len(g):5d} "
              f"{100*(g.outcome == 'missed').mean():6.1f}% {labels[j]:30s} "
              f"{100*sh[j]:6.1f}%")
    print("\n" + "=" * 78)
    print("7  UNASSIGNED ATTRIBUTION TAIL")
    print("=" * 78)
    u = df.unassigned_frac
    print(f"  median {100*u.median():.2f}%   q75 {100*u.quantile(.75):.2f}%   "
          f"q95 {100*u.quantile(.95):.2f}%   max {100*u.max():.2f}%")
    for t in (0.15, 0.20, 0.30):
        print(f"  above {100*t:.0f}%: {(u > t).sum():4d} flows "
              f"({100*(u > t).mean():.1f}%)")
    if "n_tokens_assigned" in df.columns:
        cc = np.corrcoef(df.n_tokens_flow, u)[0, 1]
        print(f"\n  corr(n_tokens_flow, unassigned_frac) = {cc:+.3f}")
        print("    strongly positive -> long flows losing tail segments to the")
        print("    128-token limit. near zero -> not a truncation problem.")
        print(f"  n_tokens_flow: min {df.n_tokens_flow.min()}  "
              f"median {int(df.n_tokens_flow.median())}  "
              f"max {df.n_tokens_flow.max()}  (limit {summary['max_length']})")
        at_limit = (df.n_tokens_flow >= summary["max_length"]).sum()
        print(f"  flows at the token limit: {at_limit}")
    print(f"\n  worst 8 flows")
    cols = [c for c in ["flow_key", "scenario", "outcome", "prob",
                        "unassigned_frac", "n_tokens_flow", "n_tokens_assigned"]
            if c in df.columns]
    print(df.nlargest(8, "unassigned_frac")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
