#!/usr/bin/env python3
"""

Usage:
    python src/xai_probe.py --run experiments/stage1_full_13_readable192/bert-mini_s42
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
from xai_methods import feature_labels, rebuild, split_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--exp-config", default="experiment_readable192.yaml")
    ap.add_argument("--feature", default="client max packet size")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--n-probe", type=int, default=3,
                    help="flows of each outcome to sweep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    run = Path(args.run).resolve()
    _, exp_cfg = load_configs(root, args.exp_config)
    data_dir = root / exp_cfg["data"]["processed_dir"]
    threshold = json.loads((run / "metrics.json").read_text())["tuned_threshold"]
    F = args.feature

    ckpt = run / "checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(ckpt, model_max_length=args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.eval()
    device = next(model.parameters()).device

    @torch.no_grad()
    def prob(texts, bs=128):
        out = []
        for i in range(0, len(texts), bs):
            enc = tokenizer(texts[i:i + bs], padding="max_length", truncation=True,
                            max_length=args.max_length, return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def value_of(text, lab=None):
        return dict(split_segments(text))[lab or F]

    def substitute(text, new_value):
        segs = [(l, (str(new_value) if l == F else v))
                for l, v in split_segments(text)]
        return rebuild(segs)

    train = pd.read_csv(data_dir / exp_cfg["data"]["train_file"])
    test = pd.read_csv(data_dir / exp_cfg["data"]["test_file"])
    labels = feature_labels(train.text.iloc[0])
    if F not in labels:
        raise SystemExit(f"{F!r} not in {labels}")

    train["v"] = [float(value_of(t).rstrip("s")) for t in train.text]
    test["v"] = [float(value_of(t).rstrip("s")) for t in test.text]
    test["prob"] = prob(test.text.tolist())
    test["outcome"] = np.where(
        test.label == 1,
        np.where(test.prob >= threshold, "detected", "missed"),
        np.where(test.prob >= threshold, "false_alarm", "correct_benign"))

    print(f"feature   {F}")
    print(f"model     {run.name}   threshold {threshold:.4f}")

    print("\n" + "=" * 76)
    print("1  WHAT DID TRAINING SHOW THE MODEL AT EACH VALUE?")
    print("=" * 76)
    print("   if the missed band was mostly benign in training, the model")
    print("   learned correctly and the dataset is the problem\n")
    g = (train.groupby("v")
              .agg(n=("label", "size"), n_attack=("label", "sum"))
              .reset_index())
    g["pct_attack"] = 100 * g.n_attack / g.n
    print(f"  {'value':>8s} {'train n':>9s} {'attack':>8s} {'% attack':>9s}   "
          f"{'test n':>7s} {'missed':>7s}")
    print("  " + "-" * 60)
    tv = test[test.label == 1].groupby("v").agg(
        n=("v", "size"), miss=("outcome", lambda s: (s == "missed").sum()))
    for _, r in g.sort_values("v").iterrows():
        t = tv.loc[r.v] if r.v in tv.index else None
        tn = f"{int(t.n):7d}" if t is not None else "      -"
        tm = f"{int(t.miss):7d}" if t is not None else "      -"
        flag = "   <-- all missed" if t is not None and t.miss == t.n and t.n else ""
        print(f"  {r.v:8.1f} {int(r.n):9d} {int(r.n_attack):8d} "
              f"{r.pct_attack:8.1f}% {tn} {tm}{flag}")
    print(f"\n  distinct values: train {train.v.nunique()}   test {test.v.nunique()}")

    print("\n" + "=" * 76)
    print("2  HOW DOES THE TOKENISER SPLIT EACH VALUE?")
    print("=" * 76)
    print("   a value that splits into more pieces than its neighbours is a")
    print("   different pattern to the model, regardless of its size\n")
    vals = sorted(set(g.v) | set(test.v))
    print(f"  {'value':>8s} {'pieces':>7s}  tokens")
    print("  " + "-" * 52)
    for v in vals:
        s = str(int(v)) if float(v).is_integer() else str(v)
        toks = tokenizer.tokenize(s)
        print(f"  {s:>8s} {len(toks):7d}  {' '.join(toks)}")

    print("\n" + "=" * 76)
    print("3  COUNTERFACTUAL: change ONLY this feature, hold the other 12 fixed")
    print("=" * 76)

    attacks = test[test.label == 1]
    grid = sorted(set(int(v) for v in vals) |
                  set(range(int(min(vals)), int(max(vals)) + 1,
                            max(1, int((max(vals) - min(vals)) / 40)))))

    for want in ("missed", "detected"):
        pool = attacks[attacks.outcome == want]
        if pool.empty:
            continue
        for _, r in pool.sample(min(args.n_probe, len(pool)),
                                random_state=args.seed).iterrows():
            orig = value_of(r.text)
            texts = [substitute(r.text, v) for v in grid]
            ps = prob(texts)
            print(f"\n  flow {str(r.flow_key)[:16]}  [{want}]  "
                  f"scenario {r.scenario}")
            print(f"  actual value {orig}, actual probability {r.prob:.4f}")
            print(f"\n    {'value':>7s} {'prob':>9s} {'':32s} {'pieces':>7s}")
            for v, p in zip(grid, ps):
                n = int(round(30 * p))
                here = " <-- actual" if str(v) == str(orig) else ""
                cross = "" if p >= threshold else "  (below threshold)"
                print(f"    {v:7d} {p:9.4f} |{'#' * n:30s}|"
                      f" {len(tokenizer.tokenize(str(v))):5d}{here}{cross}")



if __name__ == "__main__":
    main()
