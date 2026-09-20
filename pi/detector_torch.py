#!/usr/bin/env python3
"""
Stage 1 Slowloris detector, PyTorch backend.

A drop-in replacement for detector.py. Same class name, same methods, same
serialisation, same threshold source -- only the runtime differs, so calling
code does not care which one it was handed.

WHY THIS EXISTS

    Integrated Gradients needs a backward pass through the embedding layer,
    which an ONNX Runtime session cannot provide. Running detection under
    ONNX and attribution under PyTorch therefore means two runtimes and two
    resident copies of the same weights, and it means the model that made
    the decision is not the object that explains it.

    Those two models agree to ~4e-06 in P(attack) with zero verdict
    disagreements across the 4,643-flow test set, so the split is defensible.
    It is simpler not to have to defend it. Detecting in PyTorch makes the
    explained model and the deciding model the same object by construction,
    and lets one load serve both stages.

    The cost is measured, not assumed: PyTorch is about 1.4x slower than
    ONNX Runtime at batch size 1 (p50 7.94 ms against 5.62 ms on an x86
    reference; re-measure on the deployment device). Stage 1 latency is a
    few milliseconds either way and generation dominates the pipeline by
    three orders of magnitude, so the simplification is cheap.

THE MODEL DIRECTORY

    Needs the HuggingFace checkpoint (config.json, model.safetensors, the
    tokeniser files) plus two files from the training run:

        metrics.json            for tuned_threshold. Do NOT substitute 0.5
        selected_features.json  for feature order and the label strings

    Feature order is read rather than hardcoded because it must match what
    the model was trained on. If the two diverge the model sees text it was
    never trained on and the predictions become meaningless without any
    error being raised.

Usage as a module:

    from detector_torch import SlowlorisDetector

    det = SlowlorisDetector("~/payload/bert-mini_torch", threads=4)
    print(det.predict_text("client max packet size is 234 | ...").verdict)

    # the same objects IG needs -- one load serves both stages
    det.model, det.tokenizer

Usage from the command line:

    python3 detector_torch.py --model-dir payload/bert-mini_torch --self-test
    python3 detector_torch.py --model-dir payload/bert-mini_torch \\
                              --data payload/test_text.csv
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def peak_rss_mb():
    """VmHWM: peak resident set size, whole process.

    A high-water mark, so it needs no sampling loop. Note this is NOT the
    metric src/profile_model.py reports -- that one is an RSS *delta* across
    the model load, which is tens of MB where this is hundreds, because most
    of the footprint here is torch and transformers rather than weights.
    Do not put the two in one column.
    """
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


@dataclass
class Detection:
    """One flow's verdict. Identical to detector.Detection."""
    verdict: str            # "attack" or "benign"
    probability: float      # P(attack)
    is_attack: bool
    latency_ms: float

    def __str__(self):
        return (f"{self.verdict:6s}  p={self.probability:.4f}  "
                f"({self.latency_ms:.1f} ms)")


class SlowlorisDetector:
    """PyTorch-backed flow classifier.

    Parameters
    ----------
    model_dir : path holding the checkpoint, metrics.json and
        selected_features.json
    threads : intra-op thread count. Set globally via torch.set_num_threads,
        so it affects anything else using torch in this process. On a Pi 4
        measure 2 against 4 before fixing it: at batch size 1 the sequence
        is short enough that thread coordination can cost more than the
        parallelism returns.
    max_length : must not be below the longest tokenised record, or the
        lowest-ranked features are silently truncated away. 128 is what the
        published IG run used; the longest record is 121 tokens. Padding
        length beyond that does not change the output, because padded
        positions are masked out of attention.
    threshold : from metrics.json "tuned_threshold" (0.24). Passing 0.5
        silently changes every verdict.
    """

    def __init__(self, model_dir, threads=4, max_length=128,
                 threshold=None, device="cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, \
            AutoTokenizer

        self.dir = Path(model_dir).expanduser()
        if not (self.dir / "config.json").exists():
            raise FileNotFoundError(f"no config.json in {self.dir}")

        self.torch = torch
        torch.set_num_threads(int(threads))
        self.threads = int(threads)
        self.device = device

        spec_path = self.dir / "selected_features.json"
        if spec_path.exists():
            spec = json.loads(spec_path.read_text())
            self.features = spec["selected"]
            self.labels = spec.get("pretty_names",
                                   {f: f for f in self.features})
        else:
            self.features, self.labels = None, None
            print("warning: selected_features.json missing; predict() from a "
                  "feature dict is unavailable, predict_text() still works",
                  file=sys.stderr)

        if threshold is None:
            mfile = self.dir / "metrics.json"
            if mfile.exists():
                threshold = json.loads(mfile.read_text())["tuned_threshold"]
            else:
                threshold = 0.5
                print("warning: no metrics.json; defaulting threshold to 0.5",
                      file=sys.stderr)
        self.threshold = float(threshold)
        self.max_length = max_length

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.dir), model_max_length=max_length)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.dir)).to(device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - t0

        # the first forward pass allocates and picks kernels, so absorb it
        # here rather than in the first real prediction
        self._warm()

    def _warm(self):
        self.predict_batch_text(["warmup"])

    def _encode(self, texts):
        return self.tokenizer(texts, truncation=True, padding="max_length",
                              max_length=self.max_length,
                              return_tensors="pt").to(self.device)

    # ---------------------------------------------------------------- api --

    def serialise(self, flow: dict) -> str:
        """Turn a feature dict into the text the model expects.

        Byte-identical to detector.SlowlorisDetector.serialise: field order
        from selected_features.json, milliseconds rendered as seconds to one
        decimal place.
        """
        if self.features is None:
            raise RuntimeError("selected_features.json not available")
        parts = []
        for col in self.features:
            if col not in flow:
                raise KeyError(f"flow is missing feature '{col}'")
            v = float(flow[col])
            if col.endswith("_ms"):
                s = f"{v / 1000:.1f}s"
            elif v.is_integer():
                s = str(int(v))
            else:
                s = f"{v:.2f}"
            parts.append(f"{self.labels.get(col, col)} is {s}")
        return " | ".join(parts)

    def predict_text(self, text) -> Detection:
        return self.predict_batch_text([text])[0]

    def predict(self, flow: dict) -> Detection:
        return self.predict_text(self.serialise(flow))

    def predict_batch_text(self, texts) -> list:
        texts = list(texts)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            logits = self.model(**self._encode(texts)).logits
            probs = self.torch.softmax(logits, -1)[:, 1].cpu().numpy()
        elapsed = (time.perf_counter() - t0) * 1000
        per = elapsed / len(texts)
        return [Detection("attack" if p >= self.threshold else "benign",
                          float(p), bool(p >= self.threshold), per)
                for p in probs]

    def predict_batch(self, flows) -> list:
        return self.predict_batch_text([self.serialise(f) for f in flows])

    # ------------------------------------------------------------- helpers --

    def info(self) -> dict:
        n_params = sum(p.numel() for p in self.model.parameters())
        weights = self.dir / "model.safetensors"
        if not weights.exists():
            weights = self.dir / "pytorch_model.bin"
        return {
            "model_dir": str(self.dir),
            "runtime": f"pytorch {self.torch.__version__}",
            "weights_mb": round(weights.stat().st_size / 1e6, 2)
                          if weights.exists() else None,
            "n_params": n_params,
            "max_length": self.max_length,
            "threshold": self.threshold,
            "threads": self.threads,
            "n_features": len(self.features) if self.features else None,
            "load_seconds": round(self.load_seconds, 3),
        }


# ------------------------------------------------------------------- cli ---
# The same two reference flows detector.py uses, so a mismatch between the
# two backends shows up immediately.

SELF_TEST_ATTACK = (
    "client max packet size is 234 | server mean packet size is 67.14 | "
    "server bytes is 470 | server stddev packet size is 3.02 | "
    "client max inter-arrival is 15.1s | server stddev inter-arrival is 8.2s | "
    "server psh packets is 0 | client min inter-arrival is 0.0s | "
    "total min packet size is 66 | server min inter-arrival is 0.0s | "
    "total min inter-arrival is 0.0s | server rst packets is 0 | "
    "total rst packets is 0"
)
SELF_TEST_BENIGN = (
    "client max packet size is 70 | server mean packet size is 60 | "
    "server bytes is 480 | server stddev packet size is 0 | "
    "client max inter-arrival is 10.3s | server stddev inter-arrival is 4.8s | "
    "server psh packets is 1 | client min inter-arrival is 0.2s | "
    "total min packet size is 60 | server min inter-arrival is 0.0s | "
    "total min inter-arrival is 0.0s | server rst packets is 0 | "
    "total rst packets is 0"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", help="CSV with a 'text' column, and optionally "
                                   "'label' for accuracy reporting")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--batch", type=int, default=1,
                    help="1 is the deployment condition; larger values "
                         "measure throughput, not per-flow latency")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=None, help="write predictions to CSV")
    args = ap.parse_args()

    det = SlowlorisDetector(args.model_dir, threads=args.threads,
                            max_length=args.max_length,
                            threshold=args.threshold)
    for k, v in det.info().items():
        print(f"  {k:<14} {v}")
    print()

    if args.self_test:
        print("self-test on two reference flows")
        for name, text, expect in [("attack", SELF_TEST_ATTACK, "attack"),
                                   ("benign", SELF_TEST_BENIGN, "benign")]:
            r = det.predict_text(text)
            ok = "ok" if r.verdict == expect else "MISMATCH"
            print(f"  {name:7s} -> {r}   [{ok}]")
        print("\nA mismatch here means the serialisation does not match what")
        print("the model was trained on - check selected_features.json and")
        print("the max_length setting before trusting any other output.")
        return

    if not args.data:
        ap.error("--data or --self-test required")

    import pandas as pd
    df = pd.read_csv(Path(args.data).expanduser())
    if args.limit:
        df = df.head(args.limit)
    texts = df["text"].tolist()
    print(f"scoring {len(texts)} flows, batch size {args.batch}\n")

    t0 = time.perf_counter()
    results, lat = [], []
    for i in range(0, len(texts), args.batch):
        rs = det.predict_batch_text(texts[i:i + args.batch])
        results.extend(rs)
        lat.extend(r.latency_ms for r in rs)
        done = i + len(rs)
        if done % 500 == 0 or done == len(texts):
            el = time.perf_counter() - t0
            print(f"  {done}/{len(texts)}  {el:.0f}s elapsed, "
                  f"{done / el:.1f} flows/s", flush=True)
    total = time.perf_counter() - t0

    lat = np.array(lat)
    print(f"\nlatency per flow (ms): "
          f"mean {lat.mean():.2f}  p50 {np.percentile(lat, 50):.2f}  "
          f"p95 {np.percentile(lat, 95):.2f}  "
          f"p99 {np.percentile(lat, 99):.2f}  max {lat.max():.2f}")
    print(f"throughput: {len(texts) / total:.1f} flows/s  "
          f"({total:.1f} s wall clock)")
    hwm = peak_rss_mb()
    if hwm:
        print(f"peak RSS: {hwm:.1f} MB (whole process, VmHWM -- includes "
              f"torch and transformers, not just the model)")

    n_attack = sum(r.is_attack for r in results)
    print(f"flagged as attack: {n_attack} of {len(results)} "
          f"({n_attack / len(results) * 100:.1f}%)")

    if "label" in df.columns:
        y = df["label"].values
        p = np.array([r.is_attack for r in results]).astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        n = tp + fp + tn + fn
        acc = (tp + tn) / n if n else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        # balanced accuracy, because the split is ~1:4. Plain accuracy is
        # flattered by the benign majority: calling everything benign would
        # score 0.80 here.
        bal = (rec + spec) / 2
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / den) if den else 0.0
        print(f"\naccuracy           {acc:.4f}")
        print(f"balanced accuracy  {bal:.4f}")
        print(f"precision          {prec:.4f}")
        print(f"recall (TPR)       {rec:.4f}")
        print(f"specificity (TNR)  {spec:.4f}")
        print(f"f1                 {f1:.4f}")
        print(f"mcc                {mcc:.4f}")
        print(f"\ntp {tp}  fp {fp}  tn {tn}  fn {fn}   "
              f"(n={n}, attack={tp+fn}, benign={tn+fp})")
        print("\nCompare against the training run's metrics.json. A mismatch")
        print("means the deployed model is not the model you evaluated.")

    if args.out:
        df["prob_attack"] = [r.probability for r in results]
        df["pred"] = [int(r.is_attack) for r in results]
        # keep the per-flow latency: percentiles cannot be recovered from a
        # file that only stores predictions
        df["latency_ms"] = lat
        df.to_csv(args.out, index=False)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
