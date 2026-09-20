#!/usr/bin/env python3
"""
Usage:
    python src/xai_ig.py --run experiments/stage1_full_13_readable192/bert-mini_s42

    # quick check first
    python src/xai_ig.py --run <dir> --limit 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_configs, write_json
from xai_methods import (build_baseline, feature_labels, integrated_gradients,
                         rebuild, split_segments)


def as_float(value_str: str) -> float:
    """'15.1s' -> 15.1 ; '234' -> 234.0. For analysis only, never re-serialised."""
    return float(value_str.rstrip("s"))


@torch.no_grad()
def predict_prob(model, tokenizer, texts, max_length, batch_size=64):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding="max_length",
                        truncation=True, max_length=max_length,
                        return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        out.append(torch.softmax(model(**enc).logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def stats(values) -> dict:
    s = pd.Series(values).dropna()
    if s.empty:
        return {}
    return {"median": float(s.median()), "mean": float(s.mean()),
            "q75": float(s.quantile(.75)), "max": float(s.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an experiments/... run dir")
    ap.add_argument("--root", default=".")
    ap.add_argument("--exp-config", default="experiment_readable192.yaml")
    ap.add_argument("--n-features", type=int, default=13,
                    help="expected feature count; the run aborts if the "
                         "serialisation does not match")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap flows, sampled at random across scenarios")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ig-steps", type=int, default=500,
                    help="path points. 500 chosen from the convergence sweep "
                         "in xai_steps_check.py: median completeness error "
                         "1e-4 (0.00%% relative), and under 0.03%% even on "
                         "near-boundary missed flows. 50 leaves a ~10%% gap "
                         "on detected flows and up to 70%% on missed ones.")
    ap.add_argument("--chunk", type=int, default=25,
                    help="path points per backward pass. Raise for speed if "
                         "memory allows, lower if it does not. Does not "
                         "change results.")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--include-benign", action="store_true",
                    help="also attribute benign flows (default: attacks only)")
    ap.add_argument("--out", default="results/07_xai")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    run = Path(args.run).resolve()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    _, exp_cfg = load_configs(root, args.exp_config)
    data_dir = root / exp_cfg["data"]["processed_dir"]
    threshold = json.loads((run / "metrics.json").read_text())["tuned_threshold"]

    ckpt = run / "checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(ckpt, model_max_length=args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.eval()
    print(f"model     {run.name}  threshold={threshold:.4f}  "
          f"max_length={args.max_length}  ig_steps={args.ig_steps}")

    train = pd.read_csv(data_dir / exp_cfg["data"]["train_file"])
    benign = train.loc[train.label == 0, "text"].tolist()
    labels = feature_labels(benign[0])

    if len(labels) != args.n_features:
        raise SystemExit(
            f"expected {args.n_features} features, the training serialisation "
            f"has {len(labels)}: {labels}\n"
            f"the data or the serialiser changed -- fix that before "
            f"attributing, or pass --n-features {len(labels)} deliberately")

    baseline = build_baseline(benign, labels)
    baseline_text = rebuild([(lab, baseline[lab]) for lab in labels])
    print(f"baseline  median benign flow from {len(benign):,} training flows")
    print(f"          {baseline_text[:96]}...")

    write_json(out_dir / "ig_baseline.json",
               {"baseline": baseline, "baseline_text": baseline_text,
                "n_benign_train_flows": len(benign)})

    test = pd.read_csv(data_dir / exp_cfg["data"]["test_file"])
    flows = test if args.include_benign else test.loc[test.label == 1]
    flows = flows.reset_index(drop=True)
    flows["prob"] = predict_prob(model, tokenizer, flows.text.tolist(),
                                 args.max_length)
    flows["pred"] = (flows.prob >= threshold).astype(int)
    flows["outcome"] = np.where(
        flows.label == 1,
        np.where(flows.pred == 1, "detected", "missed"),
        np.where(flows.pred == 1, "false_alarm", "correct_benign"))

    if args.limit and args.limit < len(flows):
        flows = flows.sample(args.limit, random_state=args.seed).reset_index(drop=True)

    print(f"flows     {len(flows)}  " +
          "  ".join(f"{k}={v}" for k, v in flows.outcome.value_counts().items()))
    expected = set(labels)
    for src, texts in (("train", benign[:200]), ("test", flows.text.tolist())):
        for t in texts:
            got = feature_labels(t)
            if len(got) != len(labels) or set(got) != expected:
                raise SystemExit(
                    f"{src} flow has {len(got)} features, expected {len(labels)}\n"
                    f"  missing: {sorted(expected - set(got))}\n"
                    f"  extra:   {sorted(set(got) - expected)}\n"
                    f"  text:    {t[:160]}")
    print(f"features  {len(labels)} present and consistent across "
          f"train sample and all {len(flows)} test flows")

    rows = []
    t0 = time.perf_counter()
    for i, r in flows.iterrows():
        attr, diag = integrated_gradients(
            model, tokenizer, r.text, baseline_text,
            steps=args.ig_steps, max_length=args.max_length,
            chunk=min(args.chunk, args.ig_steps))
        values = dict(split_segments(r.text))
        total = sum(abs(attr[l]) for l in labels) or 1.0

        row = {"flow_key": r.get("flow_key", i), "scenario": r.scenario,
               "device": r.get("device", ""), "label": int(r.label),
               "pred": int(r.pred), "outcome": r.outcome, "prob": float(r.prob),
               "completeness_error": diag["completeness_error"],
               "unassigned_frac": diag["unassigned_frac"],
               "unassigned_attr": diag["unassigned_attr"],
               "n_tokens_assigned": diag["n_tokens_assigned"],
               "logodds_delta": diag["logodds_delta"],
               "fx": diag["fx"], "f0": diag["f0"],
               "n_tokens_flow": diag["n_tokens_flow"],
               "n_tokens_baseline": diag["n_tokens_baseline"]}
        for lab in labels:
            row[f"attr_{lab}"] = attr[lab]
            row[f"share_{lab}"] = abs(attr[lab]) / total
            row[f"val_{lab}"] = as_float(values[lab])
        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(flows)}")
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ig_flows.csv", index=False)

    A = df[[f"attr_{l}" for l in labels]].to_numpy()
    ranks = (-A).argsort(1).argsort(1) + 1            

    by_feature = pd.DataFrame({
        "feature": labels,
        "mean_attr": A.mean(0),
        "median_attr": np.median(A, 0),
        "std_attr": A.std(0),
        "mean_abs_attr": np.abs(A).mean(0),
        "mean_share": df[[f"share_{l}" for l in labels]].to_numpy().mean(0),
        "pct_positive": (A > 0).mean(0) * 100,
        "pct_rank1": (ranks == 1).mean(0) * 100,
        "pct_top3": (ranks <= 3).mean(0) * 100,
        "mean_rank": ranks.mean(0),
    }).sort_values("mean_abs_attr", ascending=False)
    by_feature.to_csv(out_dir / "ig_by_feature.csv", index=False)
    def breakdown(key):
        out = []
        for grp, g in df.groupby(key):
            G = g[[f"attr_{l}" for l in labels]].to_numpy()
            S = g[[f"share_{l}" for l in labels]].to_numpy()
            for j, lab in enumerate(labels):
                out.append({key: grp, "n": len(g), "feature": lab,
                            "mean_attr": G[:, j].mean(),
                            "mean_share": S[:, j].mean(),
                            "pct_positive": (G[:, j] > 0).mean() * 100})
        return pd.DataFrame(out)

    breakdown("scenario").to_csv(out_dir / "ig_by_scenario.csv", index=False)
    breakdown("outcome").to_csv(out_dir / "ig_by_outcome.csv", index=False)

    logodds_scale = float(np.abs(df.logodds_delta).mean()) or 1.0
    tok_mismatch = int((df.n_tokens_flow != df.n_tokens_baseline).sum())
    summary = {
        "run": str(run), "threshold": threshold,
        "ig_steps": args.ig_steps, "max_length": args.max_length,
        "n_flows": len(df),
        "outcome_counts": df.outcome.value_counts().to_dict(),
        "completeness_error": {
            **stats(df.completeness_error),
            "median_relative_to_logodds":
                float(df.completeness_error.median() / logodds_scale)},
        "unassigned_frac": stats(df.unassigned_frac),
        "token_alignment": {
            "n_flows_with_length_mismatch": tok_mismatch,
            "pct_with_length_mismatch": 100 * tok_mismatch / len(df),
            "mean_n_tokens_flow": float(df.n_tokens_flow.mean()),
            "mean_n_tokens_baseline": float(df.n_tokens_baseline.mean()),
        },
        "runtime_s_total": elapsed,
        "runtime_ms_per_flow": 1000 * elapsed / len(df),
        "top_feature_overall": by_feature.iloc[0].feature,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"\n  all 13 features, ranked by mean |attribution|\n")
    print(f"  {'feature':32s} {'mean':>9s} {'median':>9s} {'share%':>7s} "
          f"{'pos%':>6s} {'rank1%':>7s} {'top3%':>6s} {'rank':>5s}")
    print("  " + "-" * 88)
    for _, r in by_feature.iterrows():
        print(f"  {r.feature:32s} {r.mean_attr:9.4f} {r.median_attr:9.4f} "
              f"{100*r.mean_share:7.1f} {r.pct_positive:6.1f} "
              f"{r.pct_rank1:7.1f} {r.pct_top3:6.1f} {r.mean_rank:5.1f}")

    ce, uf, ta = (summary["completeness_error"], summary["unassigned_frac"],
                  summary["token_alignment"])
    print(f"\n  checks")
    print(f"    completeness error   median {ce['median']:.5f}  "
          f"max {ce['max']:.5f}  "
          f"({100 * ce['median_relative_to_logodds']:.3f}% of mean |log-odds|)")
    print(f"      -> is IG itself correct? should be ~0")
    print(f"    unassigned fraction  median {100*uf['median']:.2f}%  "
          f"q75 {100*uf['q75']:.2f}%  max {100*uf['max']:.2f}%")
    print(f"      -> share of TOTAL |attribution| landing outside the 13")
    print(f"         feature spans: [CLS], [SEP], padding, '|' separators")
    print(f"    token alignment      {ta['n_flows_with_length_mismatch']}/{len(df)} "
          f"flows differ in length from the baseline "
          f"({ta['mean_n_tokens_flow']:.1f} vs {ta['mean_n_tokens_baseline']:.1f} tokens)")
    print(f"  runtime: {elapsed:.1f}s total, "
          f"{summary['runtime_ms_per_flow']:.1f} ms/flow")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()