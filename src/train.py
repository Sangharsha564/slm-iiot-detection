#!/usr/bin/env python3
"""
Usage:
    python src/train.py --model electra-small --feature-set full_13 --seed 42
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (compute_metrics, best_threshold, load_configs, load_data,
                    pick_device, run_dir, set_seed, write_json)


class FlowDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                             max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    probs, trues, total_loss = [], [], 0.0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        total_loss += loss_fn(logits, labels).item() * len(labels)
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        trues.append(labels.cpu().numpy())
    return (np.concatenate(probs), np.concatenate(trues),
            total_loss / len(loader.dataset))


def plot_curve(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h.epoch, h.train_loss, marker="o", label="train")
    ax[0].plot(h.epoch, h.val_loss, marker="o", label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_mcc, marker="o", color="tab:green")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val MCC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--feature-set", default="full_13")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "mps", "cuda", "auto"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--exp-config", default="experiment.yaml",
                    help="experiment config filename inside configs/")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    models_cfg, exp_cfg = load_configs(root, args.exp_config)
    if args.model not in models_cfg["models"]:
        sys.exit(f"unknown model: {args.model}")

    cfg = dict(models_cfg["defaults"])
    cfg.update(models_cfg["models"][args.model].get("overrides", {}) or {})
    if args.epochs:
        cfg["epochs"] = args.epochs
    repo = models_cfg["models"][args.model]["repo"]

    set_seed(args.seed)
    device = pick_device(args.device)

    variant = Path(args.exp_config).stem
    suffix = "" if variant == "experiment" else variant.replace("experiment_", "_")
    out = run_dir(root, args.feature_set + suffix, args.model, args.seed)
    out.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model}  features={args.feature_set}{suffix}  "
          f"seed={args.seed}  device={device}")
    print(f"  data: {exp_cfg['data']['processed_dir']}")

    splits, meta = load_data(root, exp_cfg, args.feature_set)
    tok = AutoTokenizer.from_pretrained(repo, model_max_length=cfg["max_length"])

    ds = {name: FlowDataset(d["text"], d["label"], tok, cfg["max_length"])
          for name, d in splits.items()}
    g = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(ds["train"], batch_size=cfg["batch_size"],
                            shuffle=True, generator=g),
        "val": DataLoader(ds["val"], batch_size=cfg["eval_batch_size"]),
        "test": DataLoader(ds["test"], batch_size=cfg["eval_batch_size"]),
    }
    for k, d in splits.items():
        print(f"  {k:5s} n={len(d):6d}  attack={int(d.label.sum()):5d}")

    conf = AutoConfig.from_pretrained(repo, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo, config=conf).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,} ({n_trainable:,} trainable)")

    w = exp_cfg["class_weighting"].get("attack_weight")
    if w is None:
        n1 = int(splits["train"].label.sum())
        n0 = len(splits["train"]) - n1
        w = n0 / n1
    weights = torch.tensor([1.0, w], dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print(f"  attack class weight: {w:.3f}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                              weight_decay=cfg["weight_decay"])
    total_steps = len(loaders["train"]) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * cfg["warmup_ratio"]), total_steps)

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(exp_cfg["mlflow"]["tracking_uri"])
            mlflow.set_experiment(exp_cfg["mlflow"]["experiment_name"])
            mlflow.start_run(run_name=f"{args.model}_{args.feature_set}_s{args.seed}")
            mlflow.log_params({
                "model": args.model, "repo": repo, "seed": args.seed,
                "feature_set": args.feature_set,
                "n_features": meta["n_features"],
                "n_params": n_params, "device": str(device),
                "attack_weight": round(w, 4), **cfg,
            })
        except Exception as exc:                             
            print(f"  mlflow disabled: {exc}")
            use_mlflow = False

    history, best_mcc, best_epoch, patience = [], -1.0, 0, 0
    train_start = time.perf_counter()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running, t0 = 0.0, time.perf_counter()
        for batch in loaders["train"]:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            optim.step(); sched.step(); optim.zero_grad()
            running += loss.item() * len(labels)
        train_loss = running / len(ds["train"])
        epoch_s = time.perf_counter() - t0

        vp, vt, val_loss = evaluate(model, loaders["val"], device, loss_fn)
        vm = compute_metrics(vt, (vp >= 0.5).astype(int), vp)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "epoch_seconds": epoch_s,
               **{f"val_{k}": v for k, v in vm.items()
                  if isinstance(v, float)}}
        history.append(row)
        print(f"  epoch {epoch:2d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_mcc={vm['mcc']:.4f}  "
              f"val_f1={vm['f1']:.4f}  ({epoch_s:.1f}s)")

        if use_mlflow:
            import mlflow
            mlflow.log_metrics({k: v for k, v in row.items()
                                if k != "epoch"}, step=epoch)

        if vm["mcc"] > best_mcc:
            best_mcc, best_epoch, patience = vm["mcc"], epoch, 0
            model.save_pretrained(out / "checkpoint")
            tok.save_pretrained(out / "checkpoint")
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                print(f"  early stop at epoch {epoch} "
                      f"(best was {best_epoch})")
                break

    train_seconds = time.perf_counter() - train_start
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    plot_curve(history, out / "training_curve.png")

    model = AutoModelForSequenceClassification.from_pretrained(
        out / "checkpoint").to(device)

    vp, vt, _ = evaluate(model, loaders["val"], device, loss_fn)
    thr, thr_val = best_threshold(vt, vp, exp_cfg["metrics"]["primary"])
    print(f"  tuned threshold (val): {thr:.2f}  -> mcc={thr_val:.4f}")

    tp_, tt_, _ = evaluate(model, loaders["test"], device, loss_fn)

    metrics = {
        "val_at_0.5": compute_metrics(vt, (vp >= 0.5).astype(int), vp),
        "val_at_tuned": compute_metrics(vt, (vp >= thr).astype(int), vp),
        "test_at_0.5": compute_metrics(tt_, (tp_ >= 0.5).astype(int), tp_),
        "test_at_tuned": compute_metrics(tt_, (tp_ >= thr).astype(int), tp_),
        "tuned_threshold": thr,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_seconds": train_seconds,
    }
    write_json(out / "metrics.json", metrics)

    m = metrics["test_at_tuned"]
    print(f"\n  TEST  f1={m['f1']:.4f}  mcc={m['mcc']:.4f}  "
          f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
          f"pr_auc={m['pr_auc']:.4f}")
    print(f"        tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")

    preds = pd.DataFrame({
        "split": "test",
        "prob_attack": tp_,
        "pred": (tp_ >= thr).astype(int),
        "label": tt_,
        "device": splits["test"]["device"].values,
        "scenario": splits["test"]["scenario"].values,
        "text": splits["test"]["text"].values,
    })
    preds.to_csv(out / "predictions.csv", index=False)
    errors = preds[preds.pred != preds.label]
    errors.to_csv(out / "errors.csv", index=False)
    print(f"  misclassified: {len(errors)} of {len(preds)}")

    by_scen = (preds.groupby("scenario")
               .apply(lambda g: pd.Series({
                   "n": len(g),
                   "n_attack": int(g.label.sum()),
                   "recall": float((g[g.label == 1].pred == 1).mean())
                             if (g.label == 1).any() else np.nan,
               }), include_groups=False)
               .reset_index())
    by_scen.to_csv(out / "per_scenario.csv", index=False)
    print("\n" + by_scen.to_string(index=False))
    write_json(out / "run_config.json", {
        "model": args.model, "repo": repo, "seed": args.seed,
        "feature_set": args.feature_set,
        "serialisation": variant,
        "processed_dir": exp_cfg["data"]["processed_dir"],
        "n_features": meta["n_features"],
        "features": meta["features"], "dropped": meta["dropped"],
        "mechanisms": meta["mechanisms"],
        "hyperparameters": cfg,
        "attack_class_weight": w,
        "n_params": n_params, "n_trainable": n_trainable,
        "device": str(device),
        "split_sizes": {k: {"n": len(v), "attack": int(v.label.sum())}
                        for k, v in splits.items()},
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    })

    if use_mlflow:
        import mlflow
        flat = {f"{k}_{mk}": mv
                for k in ["val_at_tuned", "test_at_tuned"]
                for mk, mv in metrics[k].items() if isinstance(mv, float)}
        mlflow.log_metrics({**flat, "tuned_threshold": thr,
                            "train_seconds": train_seconds})
        for f in ["metrics.json", "run_config.json", "history.csv",
                  "per_scenario.csv", "errors.csv", "training_curve.png"]:
            p = out / f
            if p.exists():
                mlflow.log_artifact(str(p))
        mlflow.end_run()

    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()