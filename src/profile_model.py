#!/usr/bin/env python3
"""
Usage:
    python src/profile.py --run experiments/stage1_full_13/electra-small_s42
    python src/profile.py --run <dir> --skip-quant --skip-onnx
"""

import argparse
import gc
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import compute_metrics, load_configs, load_data, write_json


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def time_inference(model, tok, texts, batch_size, max_length,
                   n_runs, warmup, threads):
    torch.set_num_threads(threads)
    model.eval()

    batches = []
    for i in range(0, min(len(texts), (n_runs + warmup) * batch_size),
                   batch_size):
        chunk = texts[i:i + batch_size]
        if len(chunk) < batch_size:
            chunk = chunk + texts[:batch_size - len(chunk)]
        batches.append(tok(chunk, truncation=True, padding="max_length",
                           max_length=max_length, return_tensors="pt"))
        if len(batches) >= n_runs + warmup:
            break

    with torch.no_grad():
        for b in batches[:warmup]:
            model(**b)

        times = []
        rss_before = peak_rss_mb()
        peak = rss_before
        for b in batches[warmup:warmup + n_runs]:
            t0 = time.perf_counter()
            model(**b)
            times.append((time.perf_counter() - t0) * 1000)
            peak = max(peak, peak_rss_mb())

    t = np.array(times)
    return {
        "batch_size": batch_size,
        "threads": threads,
        "n_runs": len(t),
        "mean_ms": float(t.mean()),
        "p50_ms": float(np.percentile(t, 50)),
        "p90_ms": float(np.percentile(t, 90)),
        "p95_ms": float(np.percentile(t, 95)),
        "p99_ms": float(np.percentile(t, 99)),
        "std_ms": float(t.std()),
        "per_flow_ms": float(t.mean() / batch_size),
        "flows_per_second": float(batch_size / (t.mean() / 1000)),
        "peak_rss_mb": float(peak),
        "rss_delta_mb": float(peak - rss_before),
    }


@torch.no_grad()
def score(model, tok, texts, labels, max_length, threshold, batch_size=64):
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True,
                  padding="max_length", max_length=max_length,
                  return_tensors="pt")
        probs.append(torch.softmax(model(**enc).logits, -1)[:, 1].numpy())
    p = np.concatenate(probs)
    return compute_metrics(labels, (p >= threshold).astype(int), p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an experiments/... run dir")
    ap.add_argument("--root", default=".")
    ap.add_argument("--skip-quant", action="store_true")
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    run = Path(args.run).resolve()
    ckpt = run / "checkpoint"
    if not ckpt.exists():
        sys.exit(f"no checkpoint at {ckpt}")

    cfg = json.loads((run / "run_config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    _, exp_cfg = load_configs(root)
    prof_cfg = exp_cfg["profiling"]
    max_length = cfg["hyperparameters"]["max_length"]
    threshold = metrics["tuned_threshold"]

    print(f"profiling {cfg['model']} ({args.run})")
    torch.set_num_threads(1)

    splits, _ = load_data(root, exp_cfg, cfg["feature_set"])
    texts = splits["test"]["text"].tolist()
    labels = splits["test"]["label"].values
    gc.collect()
    rss0 = peak_rss_mb()
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(ckpt, model_max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.eval()
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        model(**tok(texts[0], truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt"))
    first_infer_ms = (time.perf_counter() - t0) * 1000
    model_rss = peak_rss_mb() - rss0

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = dir_size_mb(ckpt)
    print(f"  params={n_params:,}  disk={size_mb:.1f} MB  "
          f"load={load_s:.2f}s  first_inference={first_infer_ms:.1f}ms")

    report = {
        "model": cfg["model"],
        "feature_set": cfg["feature_set"],
        "seed": cfg["seed"],
        "n_params": n_params,
        "n_features": cfg["n_features"],
        "max_length": max_length,
        "disk_mb_fp32": size_mb,
        "model_rss_mb": model_rss,
        "cold_start_load_s": load_s,
        "cold_start_first_inference_ms": first_infer_ms,
        "cold_start_total_s": load_s + first_infer_ms / 1000,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "torch": torch.__version__,
        },
        "note": "CPU only; the deployment target has no GPU.",
    }

    print("\n  latency (fp32)")
    rows = []
    for threads in prof_cfg["threads"]:
        for bs in prof_cfg["batch_sizes"]:
            r = time_inference(model, tok, texts, bs, max_length,
                               prof_cfg["latency_runs"] if bs == 1 else 200,
                               prof_cfg["latency_warmup"], threads)
            r["precision"] = "fp32"
            rows.append(r)
            print(f"    threads={threads} batch={bs:2d}  "
                  f"p50={r['p50_ms']:7.2f}ms  p95={r['p95_ms']:7.2f}ms  "
                  f"per_flow={r['per_flow_ms']:6.2f}ms  "
                  f"{r['flows_per_second']:8.1f} flows/s")

    if not args.skip_quant and prof_cfg.get("measure_quantised", True):
        print("\n  int8 dynamic quantisation")
        try:
            qmodel = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
            qdir = run / "checkpoint_int8"
            qdir.mkdir(exist_ok=True)
            torch.save(qmodel.state_dict(), qdir / "model_int8.pt")
            qsize = (qdir / "model_int8.pt").stat().st_size / 1e6

            for threads in prof_cfg["threads"]:
                r = time_inference(qmodel, tok, texts, 1, max_length,
                                   prof_cfg["latency_runs"],
                                   prof_cfg["latency_warmup"], threads)
                r["precision"] = "int8"
                rows.append(r)
                print(f"    threads={threads} batch= 1  "
                      f"p50={r['p50_ms']:7.2f}ms  p95={r['p95_ms']:7.2f}ms  "
                      f"{r['flows_per_second']:8.1f} flows/s")

            qm = score(qmodel, tok, texts, labels, max_length, threshold)
            report["int8"] = {
                "disk_mb": qsize,
                "size_reduction_pct": 100 * (1 - qsize / size_mb),
                "test_metrics": qm,
                "f1_delta_vs_fp32": qm["f1"] - metrics["test_at_tuned"]["f1"],
                "mcc_delta_vs_fp32": qm["mcc"] - metrics["test_at_tuned"]["mcc"],
            }
            print(f"    disk {size_mb:.1f} -> {qsize:.1f} MB  "
                  f"({report['int8']['size_reduction_pct']:.0f}% smaller)")
            print(f"    test f1 {metrics['test_at_tuned']['f1']:.4f} -> "
                  f"{qm['f1']:.4f}  "
                  f"(delta {report['int8']['f1_delta_vs_fp32']:+.4f})")
        except Exception as exc:                              # noqa: BLE001
            print(f"    quantisation failed: {exc}")
            report["int8"] = {"error": str(exc)}

    if not args.skip_onnx:
        print("\n  onnx export")
        try:
            onnx_dir = root / "deployment/onnx" / \
                f"{cfg['model']}_{cfg['feature_set']}_s{cfg['seed']}"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            enc = tok("sample", truncation=True, padding="max_length",
                      max_length=max_length, return_tensors="pt")
            inputs = tuple(enc[k] for k in ["input_ids", "attention_mask"])
            torch.onnx.export(
                model, inputs, str(onnx_dir / "model.onnx"),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={"input_ids": {0: "batch"},
                              "attention_mask": {0: "batch"},
                              "logits": {0: "batch"}},
                opset_version=17,
            )
            tok.save_pretrained(onnx_dir)
            osize = (onnx_dir / "model.onnx").stat().st_size / 1e6
            report["onnx"] = {"path": str(onnx_dir), "disk_mb": osize}
            print(f"    exported {osize:.1f} MB -> {onnx_dir}")
        except Exception as exc:                              
            print(f"    onnx export failed: {exc}")
            report["onnx"] = {"error": str(exc)}

    report["latency"] = rows
    write_json(run / "profile.json", report)
    pd.DataFrame(rows).to_csv(run / "latency.csv", index=False)

    b1 = next(r for r in rows
              if r["batch_size"] == 1 and r["threads"] == max(prof_cfg["threads"])
              and r["precision"] == "fp32")
    print(f"\n  headline: {b1['p50_ms']:.2f} ms per flow "
          f"(batch 1, {b1['threads']} threads), "
          f"{size_mb:.1f} MB on disk, {model_rss:.0f} MB resident")
    print(f"written to {run / 'profile.json'}")


if __name__ == "__main__":
    main()
