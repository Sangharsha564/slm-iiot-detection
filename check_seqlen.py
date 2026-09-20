#!/usr/bin/env python3
"""
Usage:
    python check_seqlen.py --run experiments/stage1_full_13_readable192/bert-mini_s42
    python check_seqlen.py --run <dir> --lengths 192 160 128 --n 400
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=None,
                    help="defaults to the processed_dir recorded in run_config")
    ap.add_argument("--lengths", nargs="+", type=int, default=[192, 128])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    run = Path(args.run).resolve()
    ck = run / "checkpoint"
    cfg = json.loads((run / "run_config.json").read_text())
    trained_len = cfg["hyperparameters"]["max_length"]

    data = Path(args.data) if args.data else \
        run.parents[1] / cfg.get("processed_dir", "data/processed/v3") / "test_text.csv"
    if not data.exists():
        raise SystemExit(f"test data not found at {data}")

    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(str(ck))
    model = AutoModelForSequenceClassification.from_pretrained(str(ck)).eval()

    texts = pd.read_csv(data)["text"].head(args.n + args.warmup).tolist()

    print(f"run:          {run.name}")
    print(f"model:        {cfg['model']}  ({cfg['n_params']:,} parameters)")
    print(f"trained at:   max_length = {trained_len}")
    print(f"data:         {data}")
    print(f"threads:      {args.threads}\n")

    # actual token usage, before any padding
    raw = [len(tok.encode(t, truncation=False)) for t in texts]
    print(f"actual tokens: mean {np.mean(raw):.1f}, max {max(raw)}")
    print(f"padding at 192: {192 - max(raw)} positions unused\n")

    results, logits = [], {}
    for L in args.lengths:
        truncated = sum(1 for r in raw if r > L)
        enc = [tok(t, truncation=True, padding="max_length", max_length=L,
                   return_tensors="pt") for t in texts]
        with torch.no_grad():
            for e in enc[:args.warmup]:
                model(**e)
            times, outs = [], []
            for e in enc[args.warmup:]:
                t0 = time.perf_counter()
                o = model(**e).logits
                times.append((time.perf_counter() - t0) * 1000)
                outs.append(o)
        t = np.array(times)
        logits[L] = torch.cat(outs)
        results.append({
            "max_length": L,
            "truncated_flows": truncated,
            "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean()),
            "flows_per_sec": float(1000 / t.mean()),
        })
        print(f"  max_length={L:4d}  p50={np.percentile(t, 50):6.2f}ms  "
              f"p95={np.percentile(t, 95):6.2f}ms  "
              f"{1000 / t.mean():7.1f} flows/s"
              + (f"   TRUNCATES {truncated} flows" if truncated else ""))

    df = pd.DataFrame(results)
    base = df.iloc[0]
    df["speedup"] = (base.mean_ms / df.mean_ms).round(3)
    df["saving_pct"] = ((1 - df.mean_ms / base.mean_ms) * 100).round(1)
    print("\nprediction equivalence against "
          f"max_length={args.lengths[0]}:")
    ref = logits[args.lengths[0]]
    for L in args.lengths[1:]:
        diff = (ref - logits[L]).abs().max().item()
        same = int((ref.argmax(1) == logits[L].argmax(1)).sum())
        pref = torch.softmax(ref, -1)[:, 1]
        pl = torch.softmax(logits[L], -1)[:, 1]
        pdiff = (pref - pl).abs().max().item()
        print(f"  {L:4d}: max logit diff {diff:.2e}  "
              f"max prob diff {pdiff:.2e}  "
              f"identical predictions {same}/{len(ref)}")
        if same == len(ref) and diff < 1e-4:
            print(f"        equivalent - {L} is safe to deploy")
        elif same == len(ref):
            print(f"        predictions match but logits differ; check for "
                  f"truncation")
        else:
            print(f"        NOT equivalent - {len(ref) - same} predictions "
                  f"change")

    print("\n" + df.to_string(index=False))
    out = run / "seqlen_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
