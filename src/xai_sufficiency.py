#!/usr/bin/env python3
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
from xai_methods import feature_labels, rebuild, split_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--exp-config", default="experiment_readable192.yaml")
    ap.add_argument("--xai-dir", default="results/07_xai")
    ap.add_argument("--out", default="results/07_xai")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rank", choices=["abs", "signed"], default="abs",
                    help="abs = rank by |attribution| (standard). signed = "
                         "rank by attribution, so features arguing AGAINST "
                         "the attack label are never selected first.")
    ap.add_argument("--target", type=float, default=0.95,
                    help="floor = smallest N keeping this fraction detected")
    ap.add_argument("--all-attacks", action="store_true",
                    help="include missed flows (default: detected only)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    run = Path(args.run).resolve()
    xai = root / args.xai_dir
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    _, exp_cfg = load_configs(root, args.exp_config)
    data_dir = root / exp_cfg["data"]["processed_dir"]
    threshold = json.loads((run / "metrics.json").read_text())["tuned_threshold"]

    ckpt = run / "checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(ckpt, model_max_length=args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.eval()
    device = next(model.parameters()).device

    @torch.no_grad()
    def prob(texts):
        out = []
        for i in range(0, len(texts), args.batch_size):
            enc = tokenizer(texts[i:i + args.batch_size], padding="max_length",
                            truncation=True, max_length=args.max_length,
                            return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    ig = pd.read_csv(xai / "ig_flows.csv")
    baseline = json.loads((xai / "ig_baseline.json").read_text())["baseline"]
    labels = [c[5:] for c in ig.columns if c.startswith("attr_")]

    test = pd.read_csv(data_dir / exp_cfg["data"]["test_file"])
    if test.flow_key.duplicated().any():
        raise SystemExit("flow_key is not unique in the test set; cannot join "
                         "attributions back to their flow text")
    text_of = test.set_index("flow_key").text

    if not args.all_attacks:
        ig = ig[ig.outcome == "detected"]
    if args.limit and args.limit < len(ig):
        ig = ig.sample(args.limit, random_state=args.seed)
    ig = ig.reset_index(drop=True)
    ig["text"] = ig.flow_key.map(text_of)
    if ig.text.isna().any():
        raise SystemExit(f"{int(ig.text.isna().sum())} flows in ig_flows.csv "
                         f"have no matching text in the test set")

    print(f"model     {run.name}   threshold {threshold:.4f}")
    print(f"flows     {len(ig)}  ({'all attacks' if args.all_attacks else 'detected only'})")
    print(f"ranking   by {'|attribution|' if args.rank == 'abs' else 'signed attribution'}")
    print(f"baseline  median benign flow\n")

    A = ig[[f"attr_{l}" for l in labels]].to_numpy()
    key = np.abs(A) if args.rank == "abs" else A
    order = np.argsort(-key, axis=1)                      

    def variant(i, keep_idx):
        segs = dict(split_segments(ig.text.iloc[i]))
        keep = set(keep_idx)
        return rebuild([(lab, segs[lab] if j in keep else baseline[lab])
                        for j, lab in enumerate(labels)])

    p_full = prob(ig.text.tolist())

    rows, per_flow = [], []
    t0 = time.perf_counter()
    for N in range(0, len(labels) + 1):
        keep_texts, drop_texts = [], []
        for i in range(len(ig)):
            top = order[i, :N]
            keep_texts.append(variant(i, top))                       
            drop_texts.append(variant(i, order[i, N:]))              
        p_keep = prob(keep_texts)
        p_drop = prob(drop_texts)

        rows.append({
            "N": N,
            "kept_detected_pct": 100 * float((p_keep >= threshold).mean()),
            "kept_mean_prob": float(p_keep.mean()),
            "sufficiency_drop": float((p_full - p_keep).mean()),
            "removed_detected_pct": 100 * float((p_drop >= threshold).mean()),
            "removed_mean_prob": float(p_drop.mean()),
            "comprehensiveness_drop": float((p_full - p_drop).mean()),
        })
        for i in range(len(ig)):
            per_flow.append({"flow_key": ig.flow_key.iloc[i], "N": N,
                             "scenario": ig.scenario.iloc[i],
                             "p_full": float(p_full[i]),
                             "p_top_n_only": float(p_keep[i]),
                             "p_top_n_removed": float(p_drop[i])})
        print(f"  N={N:2d}  kept {rows[-1]['kept_detected_pct']:5.1f}% detected   "
              f"removed {rows[-1]['removed_detected_pct']:5.1f}% detected")
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sufficiency.csv", index=False)
    pd.DataFrame(per_flow).to_csv(out_dir / "sufficiency_flows.csv", index=False)

    
    full_row = df[df.N == len(labels)].iloc[0]
    zero_row = df[df.N == 0].iloc[0]
    rebuild_err = abs(full_row.sufficiency_drop)
    print(f"\n  CORRECTNESS")
    print(f"    N=13 (nothing substituted) must reproduce the original")
    print(f"      mean |probability difference| = {rebuild_err:.2e}  "
          f"{'OK' if rebuild_err < 1e-4 else 'FAILED -- rebuild is lossy'}")
    print(f"    N=0 (everything substituted) is the baseline flow itself")
    print(f"      mean probability = {zero_row.kept_mean_prob:.4f}  "
          f"{'OK' if zero_row.kept_mean_prob < threshold else 'UNEXPECTED'}")
    if rebuild_err >= 1e-4:
        print(f"\n    STOP: the table below cannot be trusted. The string "
              f"rebuild is changing the text even when no feature is "
              f"substituted, so every row conflates the effect of "
              f"substitution with a formatting artefact.")

    print(f"\n  {'N':>3s} {'kept: detected':>15s} {'mean p':>8s} {'suff drop':>10s}"
          f"   {'removed: detected':>18s} {'mean p':>8s} {'compr drop':>11s}")
    print("  " + "-" * 82)
    for _, r in df.iterrows():
        print(f"  {int(r.N):3d} {r.kept_detected_pct:14.1f}% {r.kept_mean_prob:8.4f} "
              f"{r.sufficiency_drop:10.4f}   {r.removed_detected_pct:17.1f}% "
              f"{r.removed_mean_prob:8.4f} {r.comprehensiveness_drop:11.4f}")

    ok = df[(df.N > 0) & (df.kept_detected_pct >= 100 * args.target)]
    print(f"\n  FLOOR")
    if len(ok):
        floor = int(ok.N.min())
        r = df[df.N == floor].iloc[0]
        print(f"    N = {floor} is the smallest number of features that keeps "
              f"{args.target:.0%} of flows detected")
        print(f"    at N={floor}: {r.kept_detected_pct:.1f}% still detected, "
              f"mean probability {r.kept_mean_prob:.4f} "
              f"(original {float(p_full.mean()):.4f})")
    else:
        floor = None
        print(f"    no N reaches {args.target:.0%}. The decision cannot be "
              f"reproduced from any subset, which would mean the attribution "
              f"ranking does not identify the evidence.")

    gains = df.set_index("N").kept_detected_pct.diff()
    print(f"\n    marginal gain from each extra feature")
    for N in range(1, min(7, len(labels) + 1)):
        pct = df[df.N == N].kept_detected_pct.iloc[0]
        g = gains.get(N, float("nan"))
        delta = "" if np.isnan(g) else f"  ({g:+.1f} points)"
        print(f"      N={N}: {pct:5.1f}% detected{delta}")
    print(f"\n    the floor is where this stops improving. anything beyond it "
          f"is a\n    choice about explanation quality, not about faithfulness.")

    write_json(out_dir / "sufficiency_summary.json", {
        "run": str(run), "threshold": threshold,
        "n_flows": len(ig), "scope": "all_attacks" if args.all_attacks else "detected",
        "ranking": args.rank, "target": args.target, "floor_N": floor,
        "rebuild_error": rebuild_err,
        "baseline_only_prob": float(zero_row.kept_mean_prob),
        "original_mean_prob": float(p_full.mean()),
        "runtime_s": elapsed,
        "table": df.to_dict("records"),
    })
    print(f"\n  runtime {elapsed:.1f}s")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
