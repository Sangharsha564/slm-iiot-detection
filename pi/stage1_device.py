#!/usr/bin/env python3
"""
Stage 1 on the deployment device: whole dataset in, metrics and latency out.

Run this on the Raspberry Pi AND on the development machine. The two runs
together answer three separate questions, which are easy to conflate:

  1  DOES IT STILL WORK?   Detection metrics, reported PER SPLIT.
        The model was fitted on train, so its performance there is not
        evidence of generalisation. A COMBINED row is printed as well
        because it is useful for the cross-device comparison and for a
        whole-dataset throughput figure -- but it is dominated by training
        flows and is not a detection result. Quote the test row.

  2  IS THE DEPLOYED MODEL THE EVALUATED MODEL?
        The development runs used float16 on MPS; the Pi runs float32 on ARM.
        Same weights, different arithmetic. This script writes a per-flow
        probability file so the two devices can be compared directly: how
        many predictions agree, and how far the probabilities drift. This is
        the one measurement that legitimately uses every split -- it compares
        two implementations of one model, not generalisation.

  3  HOW FAST, AND IN HOW MUCH MEMORY?
        Two different numbers, both reported:
          batched throughput  flows/second with a full batch. What you get
                              when a backlog of flows is processed at once.
          single-flow latency batch size 1, one flow at a time. What an
                              online detector actually experiences, and
                              always worse per flow than the batched figure.

Latency is measured after a warm-up. The first forward pass through a freshly
loaded model includes lazy allocation and kernel selection, and on a cold
device it can be several times the steady-state cost; including it would
inflate the reported latency without describing anything real.

MAX_LENGTH IS PINNED TO 128, NOT READ FROM THE TOKENIZER

    tokenizer_config.json carries model_max_length 192, inherited from the
    training config. Reading it would pad every sequence to 192 instead of
    128: identical predictions, because the longest record is ~123 tokens and
    padding is masked out of attention, but ~1.5x the positions to compute
    over, with attention quadratic in length. The published attribution run
    (results/07_xai/summary.json) used 128, so 128 is what keeps latency
    figures comparable with it. Override with --max-length if you want the
    contrast.

METRICS MATCH metrics.json

    accuracy, balanced_accuracy, precision, recall, f1, mcc, tp/fp/tn/fn,
    false_positive_rate, false_negative_rate, pr_auc, roc_auc -- the same
    keys the training run recorded, so the two are directly comparable.
    On a ~1:4 split, plain accuracy is flattered (predicting all-benign
    scores 0.80), which is why balanced accuracy and MCC are both here.

No scikit-learn dependency: ROC-AUC and average precision are computed with
numpy so the script runs on a minimal device image.

Usage:
    # flat layout: weights, tokenizer, metrics.json and the CSVs together
    python3 stage1_device.py --ckpt . \\
        --csv train_text.csv val_text.csv test_text.csv

    # quick check before the real run (samples every split, not just the first)
    python3 stage1_device.py --ckpt . --csv *_text.csv --max-rows 300
"""

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ----------------------------------------------------------------- metrics

def _ranks(x):
    """Average ranks, ties shared. Equivalent to scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def roc_auc(y, p):
    """Mann-Whitney U form: the probability a random positive outranks a
    random negative. Handles ties correctly, which a naive trapezoid over
    unique thresholds does not."""
    y = np.asarray(y)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = _ranks(np.asarray(p, dtype=float))
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, p):
    """Area under precision-recall, summed as a step function.

    Reported alongside ROC-AUC because the classes are imbalanced, and
    ROC-AUC stays optimistic under imbalance in a way that PR does not.
    """
    y = np.asarray(y)
    if (y == 1).sum() == 0:
        return None
    order = np.argsort(-np.asarray(p, dtype=float), kind="mergesort")
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / (y == 1).sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def confusion(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return (int(((y == 1) & (pred == 1)).sum()),
            int(((y == 0) & (pred == 1)).sum()),
            int(((y == 1) & (pred == 0)).sum()),
            int(((y == 0) & (pred == 0)).sum()))


def metrics(y, p, thr):
    """Same keys metrics.json records, so the two tables can be compared."""
    tp, fp, fn, tn = confusion(y, (np.asarray(p) >= thr).astype(int))
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0          # TPR / sensitivity
    spec = tn / (tn + fp) if tn + fp else 0.0         # TNR
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": n, "n_attack": tp + fn, "n_benign": fp + tn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / n if n else 0.0,
        # the split is ~1:4, so plain accuracy is flattered: all-benign
        # scores 0.80. Balanced accuracy and MCC are the honest headlines.
        "balanced_accuracy": (rec + spec) / 2,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "mcc": ((tp * tn - fp * fn) / den) if den else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "roc_auc": roc_auc(y, p),
        "pr_auc": average_precision(y, p),
    }


HDR = (f"  {'split':16s} {'n':>7s} {'acc%':>7s} {'bal%':>7s} {'prec%':>7s} "
       f"{'rec%':>7s} {'F1':>7s} {'MCC':>7s} {'FPR%':>6s} {'AUC':>7s} "
       f"{'PR-AUC':>7s}   TP/FP/FN/TN")


def show(name, m):
    print(f"  {name:16s} {m['n']:7d} {m['accuracy']*100:7.2f} "
          f"{m['balanced_accuracy']*100:7.2f} {m['precision']*100:7.2f} "
          f"{m['recall']*100:7.2f} {m['f1']*100:7.2f} {m['mcc']:7.4f} "
          f"{m['false_positive_rate']*100:6.2f} "
          f"{(m['roc_auc'] or 0)*100:7.2f} {(m['pr_auc'] or 0)*100:7.2f}"
          f"   {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")


# ------------------------------------------------------------------- setup

def peak_rss_mb():
    """ru_maxrss is kilobytes on Linux and BYTES on macOS. Getting this
    wrong reports the Pi as using 1000x less memory than the Mac."""
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 if sys.platform != "darwin" else kb / 1024 / 1024


def load(ckpt, dtype, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    try:
        mdl = AutoModelForSequenceClassification.from_pretrained(
            ckpt, dtype=dtype)
    except TypeError:
        # older transformers spell it torch_dtype. Falling back rather than
        # letting the kwarg be swallowed, which would load float32 while the
        # summary claimed otherwise.
        mdl = AutoModelForSequenceClassification.from_pretrained(
            ckpt, torch_dtype=dtype)
    mdl.to(device).eval()
    return tok, mdl


@torch.no_grad()
def infer(tok, mdl, texts, max_length, batch, device):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding="max_length", truncation=True,
                  max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = mdl(**enc).logits.float()
        out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=".",
                    help="directory holding config.json and the weights. "
                         "metrics.json is read from here or its parent for "
                         "the tuned threshold")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="one or more CSVs. Each contributes a 'split' column "
                         "named after its file stem, so results stay separable")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--max-length", type=int, default=128,
                    help="pinned to 128 to match the published runs. The "
                         "tokenizer config says 192, which would pad every "
                         "sequence half as far again for identical "
                         "predictions -- see the module docstring")
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: tuned_threshold from metrics.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--latency-n", type=int, default=200,
                    help="flows timed one at a time for the online latency "
                         "figure")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads. Default: torch's own choice. "
                         "Worth sweeping: on four cores the optimum at batch "
                         "size 1 is often 2, not 4")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cpu",
                    help="cpu on the Pi. Use mps/cuda on the dev machine to "
                         "produce the comparison run")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="rows per split, not in total: a smoke test that "
                         "took the head of the concatenated frame would "
                         "sample only the first file")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None,
                    help="label for this run, e.g. pi4 or macbook. Default: "
                         "hostname")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    tag = args.tag or platform.node().split(".")[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"host       {platform.node()}  {platform.machine()}  "
          f"{platform.system()}")
    print(f"python     {platform.python_version()}   torch {torch.__version__}"
          f"   threads {torch.get_num_threads()}")

    # ---- threshold ------------------------------------------------------
    ckpt = Path(args.ckpt).resolve()
    thr = args.threshold
    if thr is None:
        for cand in (ckpt / "metrics.json", ckpt.parent / "metrics.json"):
            if cand.exists():
                thr = json.loads(cand.read_text()).get("tuned_threshold")
                if thr is not None:
                    print(f"threshold  {thr:.4f}  (from {cand})")
                    break
    if thr is None:
        raise SystemExit(
            "no threshold: metrics.json not found or has no 'tuned_threshold'. "
            "Pass --threshold explicitly. Do not fall back to 0.5 silently -- "
            "it changes every verdict.")

    # ---- data -----------------------------------------------------------
    frames = []
    for f in args.csv:
        p = Path(f)
        t = pd.read_csv(p)
        missing = [c for c in (args.text_col, args.label_col)
                   if c not in t.columns]
        if missing:
            raise SystemExit(
                f"{p.name} has no column {missing}. Columns present: "
                f"{list(t.columns)[:12]}. Set --text-col / --label-col.")
        t["split"] = p.stem
        if args.max_rows:
            t = t.head(args.max_rows)
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)

    print(f"\ndata       {len(df)} flows from {len(args.csv)} files")
    for s, g in df.groupby("split", sort=False):
        n_att = int((g[args.label_col] == 1).sum())
        print(f"             {s:22s} {len(g):7d}   attack {n_att:6d} "
              f"({100*n_att/len(g):4.1f}%)")

    # ---- model ----------------------------------------------------------
    t0 = time.perf_counter()
    tok, mdl = load(str(ckpt), dtype, args.device)
    load_s = time.perf_counter() - t0
    n_par = sum(p.numel() for p in mdl.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in mdl.parameters()) / 1e6
    # what actually loaded, not what was asked for
    real_dtype = str(next(mdl.parameters()).dtype).replace("torch.", "")
    max_length = args.max_length

    print(f"\nmodel      {n_par/1e6:.1f}M parameters, {size_mb:.1f} MB of "
          f"weights in {real_dtype}, loaded in {load_s:.1f}s")
    if real_dtype != args.dtype:
        print(f"           WARNING requested {args.dtype}, got {real_dtype}")
    print(f"           max_length {max_length}, device {args.device}")

    texts = df[args.text_col].astype(str).tolist()

    # ---- warm-up --------------------------------------------------------
    # excluded from every timing below: the first passes pay for lazy
    # allocation and kernel selection, which is a one-off, not a per-flow cost
    infer(tok, mdl, texts[:args.warmup], max_length, 1, args.device)

    # ---- batched throughput ---------------------------------------------
    t0 = time.perf_counter()
    probs = infer(tok, mdl, texts, max_length, args.batch_size, args.device)
    batch_s = time.perf_counter() - t0

    # ---- single-flow latency --------------------------------------------
    n_lat = min(args.latency_n, len(texts))
    lat = []
    for t in texts[:n_lat]:
        t0 = time.perf_counter()
        infer(tok, mdl, [t], max_length, 1, args.device)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    rss = peak_rss_mb()

    print(f"\nlatency    batched   {batch_s:.1f}s for {len(texts)} flows = "
          f"{1000*batch_s/len(texts):.2f} ms/flow, "
          f"{len(texts)/batch_s:.0f} flows/s (batch {args.batch_size})")
    print(f"           single    median {np.median(lat):.2f} ms, "
          f"p95 {np.percentile(lat, 95):.2f} ms, "
          f"p99 {np.percentile(lat, 99):.2f} ms, max {lat.max():.2f} ms "
          f"(n={n_lat}, batch 1)")
    print(f"           note      timings include tokenisation, which is the "
          f"real per-flow cost")
    print(f"memory     peak RSS {rss:.0f} MB for the whole process "
          f"(interpreter and libraries included, not just weights)")

    # ---- metrics --------------------------------------------------------
    df["prob"] = probs
    df["pred"] = (probs >= thr).astype(int)

    print(f"\nDETECTION at threshold {thr:.4f}")
    print(HDR)
    per_split = {}
    for s, g in df.groupby("split", sort=False):
        m = metrics(g[args.label_col].to_numpy(), g.prob.to_numpy(), thr)
        per_split[s] = m
        show(s, m)

    combined = metrics(df[args.label_col].to_numpy(), df.prob.to_numpy(), thr)
    print("  " + "-" * 108)
    show("COMBINED", combined)

    print()
    print("  The model was fitted on train, so only the test row is evidence")
    print("  of detection performance. COMBINED is dominated by training")
    print("  flows -- use it for the cross-device comparison and for a")
    print("  whole-dataset throughput figure, not as a headline result.")
    print("  On a ~1:4 split quote balanced accuracy or MCC, not accuracy:")
    print("  predicting all-benign would score about 0.80.")

    if "scenario" in df.columns:
        te = df[df.split.str.contains("test", case=False)]
        if len(te):
            print(f"\n  BY SCENARIO (test split only)")
            print(HDR)
            for s, g in te.groupby("scenario", sort=False):
                m = metrics(g[args.label_col].to_numpy(), g.prob.to_numpy(),
                            thr)
                show(str(s)[:16], m)

    # ---- outputs --------------------------------------------------------
    keep = [c for c in ("flow_key", "scenario", "split") if c in df.columns]
    probs_path = out / f"stage1_probs_{tag}.csv"
    df[keep + [args.label_col, "prob", "pred"]].to_csv(probs_path, index=False)

    summary = {
        "tag": tag,
        "host": {"node": platform.node(), "machine": platform.machine(),
                 "system": platform.system(),
                 "python": platform.python_version(),
                 "torch": torch.__version__,
                 "threads": torch.get_num_threads()},
        "model": {"checkpoint": str(ckpt), "parameters": int(n_par),
                  "weights_mb": round(size_mb, 1),
                  "dtype_requested": args.dtype, "dtype_loaded": real_dtype,
                  "device": args.device, "max_length": int(max_length),
                  "load_seconds": round(load_s, 2)},
        "threshold": float(thr),
        "latency": {
            "includes_tokenisation": True,
            "batched_ms_per_flow": round(1000 * batch_s / len(texts), 3),
            "batched_flows_per_second": round(len(texts) / batch_s, 1),
            "batch_size": args.batch_size,
            "single_ms_median": round(float(np.median(lat)), 3),
            "single_ms_p95": round(float(np.percentile(lat, 95)), 3),
            "single_ms_p99": round(float(np.percentile(lat, 99)), 3),
            "single_ms_max": round(float(lat.max()), 3),
            "single_n": int(n_lat)},
        "peak_rss_mb": round(rss, 1),
        "metrics_by_split": per_split,
        "metrics_combined": combined,
    }
    summary_path = out / f"stage1_summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nwritten    {probs_path}")
    print(f"           {summary_path}")


if __name__ == "__main__":
    main()