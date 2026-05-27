"""
End-to-end smoke test — confirms full pipeline works together.
Run from project root: python smoke_test.py
"""

import os, sys, time
import numpy as np
import mlflow
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.reproducibility import set_seed, get_device

set_seed(42)
device = get_device()

BASE  = os.path.dirname(os.path.abspath(__file__))
PREP  = os.path.join(BASE, 'dataset', 'preprocessed')
CFG_F = os.path.join(BASE, 'configs', 'project_config.yaml')

with open(CFG_F) as f:
    cfg = yaml.safe_load(f)

label_map = {int(k): v for k, v in cfg['dataset']['label_map'].items()}

print("\n" + "═"*60)
print("  SMOKE TEST — SLM IIoT Detection Pipeline")
print("═"*60)
print(f"  Device : {device}")

# ── 1. Load data ───────────────────────────────────────────────────
print("\n[1/5] Loading preprocessed data ...")
X_train = np.load(os.path.join(PREP, 'X_train_resampled.npy'))
y_train = np.load(os.path.join(PREP, 'y_train_resampled.npy'))
X_test  = np.load(os.path.join(PREP, 'X_test.npy'))
y_test  = np.load(os.path.join(PREP, 'y_test.npy'))

print(f"  X_train (SMOTE): {X_train.shape}")
print(f"  X_test         : {X_test.shape}")

classes, counts = np.unique(y_train, return_counts=True)
print(f"  Train class counts (after SMOTE):")
for c, n in zip(classes, counts):
    print(f"    {c} ({label_map[c]:<12}): {n:,}")

# ── 2. MLflow ──────────────────────────────────────────────────────
print("\n[2/5] Configuring MLflow ...")
mlflow.set_experiment(cfg['mlflow']['experiment_xgboost'])
print(f"  Tracking URI : {mlflow.get_tracking_uri()}")

# ── 3. Quick XGBoost (50 trees — smoke only) ───────────────────────
print("\n[3/5] Training XGBoost (smoke: 50 trees) ...")
params = dict(n_estimators=50, max_depth=4, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8,
              nthread=-1, eval_metric='mlogloss',
              random_state=42, verbosity=0)

with mlflow.start_run(run_name="smoke-test-xgboost") as run:
    mlflow.log_params(params)
    mlflow.set_tag('type', 'smoke-test')

    t0    = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # ── 4. Evaluate ───────────────────────────────────────────────
    print("\n[4/5] Evaluating ...")
    preds    = model.predict(X_test)
    f1_macro = f1_score(y_test, preds, average='macro')
    f1_sl    = f1_score(y_test, preds, average=None,
                        labels=list(range(6)))[1]

    mlflow.log_metrics({'f1_macro': round(f1_macro, 4),
                        'f1_slowloris': round(f1_sl, 4),
                        'train_time_s': round(train_time, 2)})

    print(f"  F1 macro     : {f1_macro:.4f}")
    print(f"  F1 slowloris : {f1_sl:.4f}")
    print(f"  Train time   : {train_time:.1f}s")
    print(f"  Run ID       : {run.info.run_id}")

    target_names = [label_map[i] for i in range(6)]
    print("\n  Per-class report:")
    print(classification_report(y_test, preds,
                                 target_names=target_names,
                                 zero_division=0))

# ── 5. Verdict ─────────────────────────────────────────────────────
print("[5/5] System checks ...")
checks = {
    'Data loaded'       : X_train.shape[0] > 0,
    'SMOTE applied'     : X_train.shape[0] == 65664,
    'Model trained'     : model.n_estimators == 50,
    'F1 macro > 0.5'    : f1_macro > 0.5,
    'MLflow logged'     : True,
    'GPU available'     : str(device) in ('mps', 'cuda'),
}

all_pass = True
for check, result in checks.items():
    status = '✓ PASS' if result else '✗ FAIL'
    if not result: all_pass = False
    print(f"  {status}  {check}")

print()
if all_pass:
    print("  ✓ All systems operational. Ready for training.")
else:
    print("  ✗ Fix failed checks before proceeding.")
print("═"*60 + "\n")
