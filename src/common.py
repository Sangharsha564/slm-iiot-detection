"""Shared utilities: config loading, data loading, seeding, metrics."""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)




def set_seed(seed: int):
    """Seed every source of randomness we can reach."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)  


def load_configs(root: Path, exp_file: str = "experiment.yaml"):
    """Load model and experiment configs.

    exp_file allows swapping the experiment config, which is how the
    feature-naming ablation points at a different processed_dir without
    duplicating anything else.
    """
    with open(root / "configs/models.yaml") as f:
        models = yaml.safe_load(f)
    with open(root / "configs" / exp_file) as f:
        exp = yaml.safe_load(f)
    return models, exp


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cpu")          
    if requested == "mps" and not torch.backends.mps.is_available():
        print("mps requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def drop_features_from_text(text: str, drop_labels: list[str]) -> str:
    """Remove named fields from a pipe-separated serialisation."""
    if not drop_labels:
        return text
    parts = [p.strip() for p in text.split("|")]
    kept = [p for p in parts
            if not any(p.startswith(f"{lab} is") for lab in drop_labels)]
    return " | ".join(kept)


def load_split(processed_dir: Path, filename: str, drop_cols: list[str],
               feature_spec: dict) -> pd.DataFrame:
    df = pd.read_csv(processed_dir / filename)
    if drop_cols:
        pretty = feature_spec.get("pretty_names", {})
        drop_labels = [pretty.get(c, c) for c in drop_cols]
        df["text"] = df["text"].map(
            lambda t: drop_features_from_text(t, drop_labels))
    return df


def load_data(root: Path, exp_cfg: dict, feature_set: str):
    """Return train, val, test frames plus the resolved feature metadata."""
    d = root / exp_cfg["data"]["processed_dir"]
    with open(d / exp_cfg["data"]["feature_spec"]) as f:
        spec = json.load(f)

    drop_cols = exp_cfg["feature_sets"][feature_set]["drop"]
    kept = [c for c in spec["selected"] if c not in drop_cols]

    splits = {}
    for name, key in [("train", "train_file"), ("val", "val_file"),
                      ("test", "test_file")]:
        splits[name] = load_split(d, exp_cfg["data"][key], drop_cols, spec)

    meta = {
        "feature_set": feature_set,
        "n_features": len(kept),
        "features": kept,
        "dropped": drop_cols,
        "mechanisms": {c: spec["mechanisms"][c] for c in kept},
    }
    return splits, meta

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """All reported metrics. Accuracy is included but should not be led with:
    at ~1:3.9 imbalance a majority-class predictor already scores ~0.80."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }
    if y_prob is not None:
        m["pr_auc"] = average_precision_score(y_true, y_prob)
        m["roc_auc"] = roc_auc_score(y_true, y_prob)
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in m.items()}


def best_threshold(y_true, y_prob, metric: str = "mcc") -> tuple[float, float]:
    """Sweep the decision threshold. Fitted on validation only."""
    fn = {"mcc": matthews_corrcoef, "f1": f1_score}[metric]
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        v = fn(y_true, (y_prob >= t).astype(int))
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v


def run_dir(root: Path, feature_set: str, model: str, seed: int) -> Path:
    return root / "experiments" / f"stage1_{feature_set}" / f"{model}_s{seed}"


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))