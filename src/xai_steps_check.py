#!/usr/bin/env python3
"""
Usage:
    python src/xai_steps_check.py --run experiments/stage1_full_13_readable192/bert-mini_s42

    # wider sweep, more flows
    python src/xai_steps_check.py --run <dir> --n-flows 12 --steps 50 100 200 500 1000 2000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_configs
from xai_methods import (build_baseline, feature_labels, integrated_gradients,
                         rebuild)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--exp-config", default="experiment_readable192.yaml")
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[50, 100, 200, 500, 1000])
    ap.add_argument("--n-flows", type=int, default=8,
                    help="flows to test, sampled to span detected and missed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=25,
                    help="path points per backward pass; lower if memory is tight")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    run = Path(args.run).resolve()
    _, exp_cfg = load_configs(root, args.exp_config)
    data_dir = root / exp_cfg["data"]["processed_dir"]
    threshold = json.loads((run / "metrics.json").read_text())["tuned_threshold"]

    ckpt = run / "checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(ckpt, model_max_length=args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.eval()

    train = pd.read_csv(data_dir / exp_cfg["data"]["train_file"])
    benign = train.loc[train.label == 0, "text"].tolist()
    labels = feature_labels(benign[0])
    baseline = build_baseline(benign, labels)
    baseline_text = rebuild([(lab, baseline[lab]) for lab in labels])

    test = pd.read_csv(data_dir / exp_cfg["data"]["test_file"])
    attacks = test.loc[test.label == 1].reset_index(drop=True)
    attacks["prob"] = predict_prob(model, tokenizer, attacks.text.tolist(),
                                   args.max_length)
    attacks["outcome"] = np.where(attacks.prob >= threshold, "detected", "missed")

    per = max(1, args.n_flows // 2)
    picks = []
    for o, g in attacks.groupby("outcome"):
        picks.append(g.sample(min(per, len(g)), random_state=args.seed))
    flows = pd.concat(picks).reset_index(drop=True)

    print(f"model     {run.name}   threshold={threshold:.4f}   "
          f"max_length={args.max_length}")
    print(f"flows     {len(flows)}  " +
          "  ".join(f"{k}={v}" for k, v in flows.outcome.value_counts().items()))
    print(f"steps     {args.steps}\n")

    rows = []
    for i, r in flows.iterrows():
        print(f"  flow {i}  [{r.outcome}]  p={r.prob:.4f}")
        print(f"    {'steps':>6s} {'err':>10s} {'gap(signed)':>12s} {'rel':>8s}  top1")
        for s in args.steps:
            scores, d = integrated_gradients(model, tokenizer, r.text,
                                             baseline_text, steps=s,
                                             max_length=args.max_length,
                                             chunk=min(args.chunk, s))
            total = d["assigned_attr"] + d["unassigned_attr"]
            gap = total - d["logodds_delta"]
            rel = abs(gap) / (abs(d["logodds_delta"]) or 1.0)
            top1 = max(scores, key=lambda k: scores[k])
            print(f"    {s:6d} {abs(gap):10.5f} {gap:12.5f} {100*rel:7.2f}%  {top1}")
            rows.append({"flow": i, "outcome": r.outcome, "prob": float(r.prob),
                         "steps": s, "err": abs(gap), "gap": gap, "rel": rel,
                         "logodds_delta": d["logodds_delta"], "top1": top1})
        print()

    df = pd.DataFrame(rows)

    print("  median error by step count")
    print(f"    {'steps':>6s} {'err':>10s} {'rel':>8s}   verdict-relevant: is it falling?")
    prev = None
    for s, g in df.groupby("steps"):
        e = g.err.median()
        arrow = "" if prev is None else ("  down" if e < prev * 0.9 else "  FLAT")
        print(f"    {s:6d} {e:10.5f} {100*g.rel.median():7.2f}%{arrow}")
        prev = e

    hi = df[df.steps == max(args.steps)]
    if len(hi) > 2:
        c = np.corrcoef(np.abs(hi.logodds_delta), hi.err)[0, 1]
        print(f"\n  at {max(args.steps)} steps: corr(|log-odds delta|, err) = {c:+.3f}")
        print(f"    strongly positive -> approximation error that scales with the")
        print(f"    integral's size (benign). near zero with a large constant error")
        print(f"    -> a fixed offset, which would point at a bug, not step count.")

    stable = df.groupby("flow").top1.nunique()
    print(f"\n  top-1 feature stable across all step counts: "
          f"{int((stable == 1).sum())}/{len(stable)} flows")
    print("    if this is all flows, the conclusion does not depend on step count")
    print("    and the completeness gap is a reporting issue, not an answer issue.")


if __name__ == "__main__":
    main()
