"""
XGBoost Gate — Full Training Script
=====================================
Trains a 6-class XGBoost classifier on the preprocessed CIC IIoT 2025 dataset.
Logs everything to MLflow. Saves model and feature importance.

Run from project root:
    python src/training/train_xgboost.py

Outputs (in models/xgboost/):
    xgboost_gate.json          ← trained model
    feature_importance.csv     ← ranked feature importance
    feature_importance_top30.png
    classification_report.txt

MLflow:
    experiment → 'xgboost-gate'
    run        → logged params, metrics, artifacts
"""

import os, sys, time, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import xgboost as xgb
import yaml
from sklearn.metrics import (
    classification_report, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

# ── project root on path ───────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.utils.reproducibility import set_seed, get_device

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
with open(os.path.join(ROOT, 'configs', 'project_config.yaml')) as f:
    cfg = yaml.safe_load(f)

PREP_DIR  = os.path.join(ROOT, cfg['paths']['preprocessed_dir'])
MODEL_DIR = os.path.join(ROOT, cfg['paths']['xgboost_dir'])
FIG_DIR   = os.path.join(ROOT, cfg['paths']['figures_dir'])
LOG_DIR   = os.path.join(ROOT, cfg['paths']['logs_dir'])
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

SEED      = cfg['project']['seed']
LABEL_MAP = {int(k): v for k, v in cfg['dataset']['label_map'].items()}
N_CLASSES = cfg['dataset']['n_classes']

set_seed(SEED)
device = get_device()

print("\n" + "═"*65)
print("  XGBoost Gate — Training")
print("═"*65)
print(f"  Device  : {device}  (XGBoost uses CPU — MPS is for SLM)")
print(f"  Seed    : {SEED}")
print(f"  Classes : {list(LABEL_MAP.values())}")

# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading preprocessed data ...")

X_train_full = np.load(os.path.join(PREP_DIR, 'X_train_resampled.npy'))
y_train_full = np.load(os.path.join(PREP_DIR, 'y_train_resampled.npy'))
X_test       = np.load(os.path.join(PREP_DIR, 'X_test.npy'))
y_test       = np.load(os.path.join(PREP_DIR, 'y_test.npy'))

with open(os.path.join(PREP_DIR, 'feature_names.txt')) as f:
    feature_names = f.read().splitlines()

print(f"  Train (SMOTE balanced): {X_train_full.shape}")
print(f"  Test  (original dist) : {X_test.shape}")
print(f"  Features              : {len(feature_names)}")

# Split a validation set from training for early stopping
# 90% train, 10% validation — stratified
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.1,
    random_state=SEED,
    stratify=y_train_full
)
print(f"  Train split : {X_train.shape[0]:,}  |  Val split : {X_val.shape[0]:,}")

# Class distribution in test
print(f"\n  Test class distribution:")
for cls_id in sorted(np.unique(y_test)):
    n = (y_test == cls_id).sum()
    print(f"    {cls_id} ({LABEL_MAP[cls_id]:<12}) : {n:>4}  ({n/len(y_test)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════
params = {
    'n_estimators'      : cfg['xgboost']['n_estimators'],
    'max_depth'         : cfg['xgboost']['max_depth'],
    'learning_rate'     : cfg['xgboost']['learning_rate'],
    'subsample'         : cfg['xgboost']['subsample'],
    'colsample_bytree'  : cfg['xgboost']['colsample_bytree'],
    'eval_metric'       : 'mlogloss',
    'nthread'           : cfg['xgboost']['nthread'],
    'random_state'      : SEED,
    'verbosity'         : 0,
    # Regularisation — reduces overfitting on the SMOTE-expanded training set
    'reg_alpha'         : 0.1,    # L1 regularisation
    'reg_lambda'        : 1.0,    # L2 regularisation (default)
    'min_child_weight'  : 5,      # minimum samples per leaf
    'gamma'             : 0.1,    # minimum loss reduction for a split
    'early_stopping_rounds': 30,  # stop if val loss doesn't improve for 30 rounds
}

print(f"\n[2/6] Model parameters:")
for k, v in params.items():
    print(f"  {k:<25}: {v}")

# ══════════════════════════════════════════════════════════════════════════
# MLFLOW SETUP
# ══════════════════════════════════════════════════════════════════════════
mlflow.set_experiment(cfg['mlflow']['experiment_xgboost'])

# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[3/6] Training XGBoost ...")
print(f"  Max trees       : {params['n_estimators']}")
print(f"  Early stopping  : {params['early_stopping_rounds']} rounds")
print(f"  All CPU cores   : nthread={params['nthread']}")

with mlflow.start_run(run_name="xgboost-full-run-01") as run:

    # Log all parameters
    mlflow.log_params({k: v for k, v in params.items()
                       if k != 'early_stopping_rounds'})
    mlflow.log_param('early_stopping_rounds', params['early_stopping_rounds'])
    mlflow.log_param('train_rows',   X_train.shape[0])
    mlflow.log_param('val_rows',     X_val.shape[0])
    mlflow.log_param('test_rows',    X_test.shape[0])
    mlflow.log_param('n_features',   len(feature_names))
    mlflow.log_param('smote_applied', True)
    mlflow.set_tag('dataset', 'CIC-IIoT-2025')
    mlflow.set_tag('type',    'full-training')

    # Train
    model = xgb.XGBClassifier(**params)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50    # print loss every 50 trees
    )
    train_time = time.time() - t0

    actual_trees = model.best_iteration + 1 if hasattr(model, 'best_iteration') else params['n_estimators']
    print(f"\n  Training complete in {train_time:.1f}s")
    print(f"  Best iteration  : {actual_trees} trees")

    # ── Evaluate on test set ───────────────────────────────────────────
    print(f"\n[4/6] Evaluating on test set ...")

    preds      = model.predict(X_test)
    probs      = model.predict_proba(X_test)

    f1_macro   = f1_score(y_test, preds, average='macro')
    f1_weighted= f1_score(y_test, preds, average='weighted')
    f1_per_cls = f1_score(y_test, preds, average=None, labels=list(range(N_CLASSES)))
    f1_slowloris = f1_per_cls[1]

    # Accuracy
    accuracy = (preds == y_test).mean()

    print(f"\n  ── Test Results ──────────────────────────────────")
    print(f"  Accuracy         : {accuracy:.4f}")
    print(f"  F1 macro         : {f1_macro:.4f}")
    print(f"  F1 weighted      : {f1_weighted:.4f}")
    print(f"  F1 Slowloris     : {f1_slowloris:.4f}  ← key metric")
    print(f"  Train time       : {train_time:.1f}s")

    print(f"\n  ── Per-Class F1 ──────────────────────────────────")
    target_names = [LABEL_MAP[i] for i in range(N_CLASSES)]
    report_str = classification_report(
        y_test, preds,
        target_names=target_names,
        zero_division=0
    )
    print(report_str)

    # Log all metrics to MLflow
    metrics = {
        'accuracy'        : round(accuracy,      4),
        'f1_macro'        : round(f1_macro,       4),
        'f1_weighted'     : round(f1_weighted,    4),
        'f1_slowloris'    : round(f1_slowloris,   4),
        'train_time_s'    : round(train_time,     2),
        'best_iteration'  : actual_trees,
    }
    for i, cls_name in LABEL_MAP.items():
        metrics[f'f1_{cls_name}'] = round(float(f1_per_cls[i]), 4)
    mlflow.log_metrics(metrics)

    # ── Feature Importance ────────────────────────────────────────────
    print(f"\n[5/6] Computing feature importance ...")

    importance_scores = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature'    : feature_names,
        'importance' : importance_scores,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    importance_df['rank'] = importance_df.index + 1

    print(f"\n  Top 20 features by importance:")
    print(f"  {'Rank':<5} {'Feature':<45} {'Importance':>10}")
    print(f"  {'─'*63}")
    for _, row in importance_df.head(20).iterrows():
        print(f"  {int(row['rank']):<5} {row['feature']:<45} {row['importance']:>10.4f}")

    # Identify low-importance features (log_* group)
    log_features = importance_df[importance_df['feature'].str.startswith('log_')]
    print(f"\n  Log feature importances (expected near-zero):")
    for _, row in log_features.iterrows():
        print(f"    {row['feature']:<45} {row['importance']:.4f}  (rank {int(row['rank'])})")

    # Save importance CSV
    imp_csv = os.path.join(MODEL_DIR, 'feature_importance.csv')
    importance_df.to_csv(imp_csv, index=False)
    mlflow.log_artifact(imp_csv)

    # Plot top 30 features
    top30 = importance_df.head(30)
    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(
        top30['feature'][::-1],
        top30['importance'][::-1],
        color='steelblue', edgecolor='white'
    )
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title(f'XGBoost — Top 30 Feature Importances\n'
                 f'F1 macro={f1_macro:.3f}  F1 Slowloris={f1_slowloris:.3f}',
                 fontsize=13)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    imp_plot = os.path.join(FIG_DIR, 'xgboost_feature_importance_top30.png')
    plt.savefig(imp_plot, dpi=150, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(imp_plot)
    print(f"\n  Feature importance plot saved.")

    # Plot confusion matrix
    cm   = confusion_matrix(y_test, preds)
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )
    disp.plot(ax=ax2, cmap='Blues', colorbar=False)
    ax2.set_title(f'XGBoost Confusion Matrix  (F1 macro={f1_macro:.3f})', fontsize=13)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    cm_plot = os.path.join(FIG_DIR, 'xgboost_confusion_matrix.png')
    plt.savefig(cm_plot, dpi=150, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_plot)
    print(f"  Confusion matrix plot saved.")

    # ── Save model and report ─────────────────────────────────────────
    print(f"\n[6/6] Saving model and report ...")

    model_path = os.path.join(MODEL_DIR, 'xgboost_gate.json')
    model.save_model(model_path)
    mlflow.xgboost.log_model(model, artifact_path='xgboost_model')
    print(f"  Model saved : {model_path}")

    report_path = os.path.join(MODEL_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"XGBoost Gate — Classification Report\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Accuracy    : {accuracy:.4f}\n")
        f.write(f"F1 macro    : {f1_macro:.4f}\n")
        f.write(f"F1 weighted : {f1_weighted:.4f}\n")
        f.write(f"F1 Slowloris: {f1_slowloris:.4f}\n")
        f.write(f"Train time  : {train_time:.1f}s\n")
        f.write(f"Trees used  : {actual_trees}\n\n")
        f.write(report_str)
        f.write(f"\n\nTop 20 Features by Importance:\n")
        f.write(importance_df.head(20).to_string(index=False))
    mlflow.log_artifact(report_path)
    print(f"  Report saved: {report_path}")

    print(f"\n  MLflow Run ID : {run.info.run_id}")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  TRAINING COMPLETE")
print(f"{'═'*65}")
print(f"  Accuracy         : {accuracy:.4f}")
print(f"  F1 macro         : {f1_macro:.4f}")
print(f"  F1 Slowloris     : {f1_slowloris:.4f}")
print(f"  Trees used       : {actual_trees} / {params['n_estimators']} max")
print(f"  Train time       : {train_time:.1f}s")
print(f"\n  Outputs in : models/xgboost/")
print(f"    ✓ xgboost_gate.json")
print(f"    ✓ feature_importance.csv")
print(f"    ✓ classification_report.txt")
print(f"  Plots in   : reports/figures/")
print(f"    ✓ xgboost_feature_importance_top30.png")
print(f"    ✓ xgboost_confusion_matrix.png")
print(f"\n  View in MLflow UI:")
print(f"    cd ~/Desktop/slm && mlflow ui --port 5001 &")
print(f"    http://127.0.0.1:5001\n")
