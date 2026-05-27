"""
Preprocessing Pipeline — combined_shuffled.csv
===============================================
Steps:
  1. Load dataset
  2. Drop irrelevant / identifier columns
  3. Encode labels (6-class model_label)
  4. Separate features X and target y
  5. Stratified 80/20 train/test split
  6. RobustScaler  (fit on train only, transform both)
  7. Class weights + SMOTE on training set
  8. Save all artifacts to preprocessed/

Output files (in dataset/preprocessed/):
  X_train.npy, X_test.npy
  y_train.npy, y_test.npy
  X_train_resampled.npy, y_train_resampled.npy   ← after SMOTE
  scaler.pkl
  class_weights.pkl
  feature_names.txt
  label_map.json
  preprocessing_report.txt

Run:
  pip install scikit-learn imbalanced-learn pandas numpy --break-system-packages
  python3 preprocess.py
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_FILE  = os.path.join(BASE_DIR, 'combined_shuffled.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'preprocessed')
REPORT    = os.path.join(OUT_DIR,  'preprocessing_report.txt')

os.makedirs(OUT_DIR, exist_ok=True)

lines = []
def p(msg=''):   print(msg); lines.append(str(msg))
def sec(title):  p(); p(f"{'═'*65}"); p(f"  {title}"); p(f"{'═'*65}")

# ── check input ────────────────────────────────────────────────────────────
if not os.path.exists(CSV_FILE):
    print(f"ERROR: {CSV_FILE} not found.\nRun shuffle_combined.py first.")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════
sec("STEP 1 — LOAD DATASET")
# ══════════════════════════════════════════════════════════════════════════
p("  Loading combined_shuffled.csv ...")
df = pd.read_csv(CSV_FILE, low_memory=False)
import numpy as _np
df.replace([_np.inf, -_np.inf], np.nan, inplace=True)

p(f"  Rows    : {len(df):,}")
p(f"  Columns : {len(df.columns)}")
p(f"  Missing : {df.isnull().sum().sum():,}")

# ══════════════════════════════════════════════════════════════════════════
sec("STEP 2 — DROP IRRELEVANT COLUMNS")
# ══════════════════════════════════════════════════════════════════════════
# These columns are: identifiers, timestamps, raw string lists, or
# useless fixed-value features that leak device identity into the model.
DROP_ALWAYS = [
    # device identity
    'device_name', 'device_mac',
    # timestamps  (flow_duration = always 10s — window artifact, useless)
    'timestamp', 'timestamp_start', 'timestamp_end',
    'flow_duration', 'flow_duration_sec',
    # raw IP / MAC / port / protocol string columns
    'network_ips_all',  'network_ips_dst',  'network_ips_src',
    'network_macs_all', 'network_macs_dst', 'network_macs_src',
    'network_ports_all','network_ports_dst','network_ports_src',
    'network_protocols_all','network_protocols_dst','network_protocols_src',
    # zero-variance in all rows and leaks nothing
    'network_tcp-flags-urg_count',
    # raw log string
    'log_data-types',
    # other label columns (we keep label2, label3 for encoding then drop)
    'label_full', 'label1', 'binary_label', '_label',
]
# only drop what actually exists
DROP_ALWAYS = [c for c in DROP_ALWAYS if c in df.columns]
p(f"  Dropping {len(DROP_ALWAYS)} columns:")
for c in DROP_ALWAYS:
    p(f"    - {c}")

df.drop(columns=DROP_ALWAYS, inplace=True)
p(f"\n  Remaining columns: {len(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════
sec("STEP 3 — ENCODE LABELS  (6-class model_label)")
# ══════════════════════════════════════════════════════════════════════════
#
#  Class    ID   Description
#  ───────  ──   ──────────────────────────────────────
#  benign    0   normal traffic
#  slowloris 1   slow-rate DoS / DDoS (key target class)
#  dos_other 2   volumetric DoS (not slowloris)
#  ddos_other 3  volumetric DDoS (not slowloris)
#  recon     4   reconnaissance / scanning
#  other     5   mitm, web, bruteforce, malware
#

LABEL_MAP = {
    0: 'benign',
    1: 'slowloris',
    2: 'dos_other',
    3: 'ddos_other',
    4: 'recon',
    5: 'other',
}

def encode_label(row):
    l2 = str(row['label2']).lower().strip()
    l3 = str(row['label3']).lower().strip()
    if l2 == 'benign':          return 0
    if 'slowloris' in l3:       return 1
    if l2 == 'dos':             return 2
    if l2 == 'ddos':            return 3
    if l2 == 'recon':           return 4
    return 5

p("  Applying label encoding ...")
df['model_label'] = df.apply(encode_label, axis=1)

counts = df['model_label'].value_counts().sort_index()
p(f"\n  {'Class':>3}  {'Name':<15}  {'Count':>7}  {'%':>6}")
p(f"  {'─'*40}")
for cls_id, cnt in counts.items():
    p(f"  {cls_id:>3}  {LABEL_MAP[cls_id]:<15}  {cnt:>7,}  {cnt/len(df)*100:>5.1f}%")

# Save label map
with open(os.path.join(OUT_DIR, 'label_map.json'), 'w') as f:
    json.dump(LABEL_MAP, f, indent=2)
p("\n  Saved: label_map.json")

# ══════════════════════════════════════════════════════════════════════════
sec("STEP 4 — SEPARATE FEATURES AND TARGET")
# ══════════════════════════════════════════════════════════════════════════

# All label columns to exclude from features
LABEL_COLS = ['label2', 'label3', 'label4', 'model_label']
LABEL_COLS = [c for c in LABEL_COLS if c in df.columns]

# Features = all remaining numeric columns
feature_cols = [
    c for c in df.columns
    if c not in LABEL_COLS
    and pd.api.types.is_numeric_dtype(df[c])
]

p(f"  Feature columns : {len(feature_cols)}")
p(f"  Label columns   : {LABEL_COLS}")

# Check for remaining missing values in features
missing_feat = df[feature_cols].isnull().sum().sum()
p(f"  Missing values in features: {missing_feat}")
if missing_feat > 0:
    p("  Filling remaining NaN with column median ...")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

X = df[feature_cols].values.astype(np.float32)
y = df['model_label'].values.astype(np.int32)

p(f"  X shape: {X.shape}")
p(f"  y shape: {y.shape}")

# Save feature names
with open(os.path.join(OUT_DIR, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(feature_cols))
p(f"  Saved: feature_names.txt  ({len(feature_cols)} features)")

# Print full feature list
p(f"\n  Feature list:")
for i, c in enumerate(feature_cols, 1):
    p(f"    {i:3d}. {c}")

# ══════════════════════════════════════════════════════════════════════════
sec("STEP 5 — STRATIFIED TRAIN / TEST SPLIT  (80 / 20)")
# ══════════════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # preserves class proportions in both splits
)

p(f"  Train : {X_train.shape[0]:,} rows  ({X_train.shape[0]/len(X)*100:.0f}%)")
p(f"  Test  : {X_test.shape[0]:,} rows  ({X_test.shape[0]/len(X)*100:.0f}%)")

p(f"\n  Class distribution in TRAIN split:")
for cls_id in sorted(np.unique(y_train)):
    n = (y_train == cls_id).sum()
    p(f"    {cls_id} {LABEL_MAP[cls_id]:<15}  {n:>5,}  ({n/len(y_train)*100:.1f}%)")

p(f"\n  Class distribution in TEST split:")
for cls_id in sorted(np.unique(y_test)):
    n = (y_test == cls_id).sum()
    p(f"    {cls_id} {LABEL_MAP[cls_id]:<15}  {n:>5,}  ({n/len(y_test)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
sec("STEP 6 — ROBUST SCALER  (fit on train, transform both)")
# ══════════════════════════════════════════════════════════════════════════
# RobustScaler uses median and IQR — not affected by extreme outliers
# which are common in network traffic features (e.g. packet counts,
# window sizes during flood attacks can be 1000× normal).
# CRITICAL: fit ONLY on train split to prevent data leakage.

p("  Fitting RobustScaler on training data ...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

p(f"  Train scaled — mean: {X_train_scaled.mean():.4f}  std: {X_train_scaled.std():.4f}")
p(f"  Test  scaled — mean: {X_test_scaled.mean():.4f}  std: {X_test_scaled.std():.4f}")

# Save scaler
with open(os.path.join(OUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
p("  Saved: scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════
sec("STEP 7 — CLASS WEIGHTS + SMOTE")
# ══════════════════════════════════════════════════════════════════════════

# ── 7a. Compute sklearn class weights ─────────────────────────────────────
# Used by: XGBoost (sample_weight), PyTorch CrossEntropyLoss(weight=...)
classes_present = np.unique(y_train)
weights_array   = compute_class_weight(
    class_weight='balanced',
    classes=classes_present,
    y=y_train
)
class_weights = {int(cls): float(w)
                 for cls, w in zip(classes_present, weights_array)}

p("  Computed class weights (balanced):")
for cls_id, w in class_weights.items():
    p(f"    class {cls_id} ({LABEL_MAP[cls_id]:<15})  weight = {w:.4f}")

# XGBoost scale_pos_weight (binary: benign=0 vs attack=1)
n_benign = (y_train == 0).sum()
n_attack = (y_train != 0).sum()
xgb_scale_pos_weight = n_benign / n_attack
p(f"\n  XGBoost scale_pos_weight (benign/attack): {xgb_scale_pos_weight:.3f}")

# scale_pos_weight for Slowloris specifically (binary gate use case)
n_non_sl = (y_train != 1).sum()
n_sl     = (y_train == 1).sum()
xgb_sl_weight = n_non_sl / n_sl if n_sl > 0 else 1.0
p(f"  XGBoost scale_pos_weight (non-SL/SL)   : {xgb_sl_weight:.1f}")

# Save weights
weights_out = {
    'class_weights'         : class_weights,
    'label_map'             : LABEL_MAP,
    'xgb_scale_pos_weight'  : xgb_scale_pos_weight,
    'xgb_sl_weight'         : xgb_sl_weight,
}
with open(os.path.join(OUT_DIR, 'class_weights.pkl'), 'wb') as f:
    pickle.dump(weights_out, f)
p("  Saved: class_weights.pkl")

# ── 7b. SMOTE ──────────────────────────────────────────────────────────────
# Oversample minority classes (especially slowloris class=1) so each class
# reaches the count of the majority class in the training set.
# Applied AFTER scaling so synthetic points are in the same feature space.
# k_neighbors=5 (default) — reduced if a class has fewer than 6 samples.

p("\n  Applying SMOTE to training set ...")
p("  (This may take 30-60 seconds for ~24k rows × 70 features)")

# Determine safe k_neighbors
min_minority = min((y_train == c).sum() for c in classes_present)
k = min(5, min_minority - 1)
p(f"  k_neighbors = {k}  (min class size in train = {min_minority})")

smote = SMOTE(random_state=42, k_neighbors=k)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

p(f"\n  Before SMOTE: {X_train_scaled.shape[0]:,} rows")
p(f"  After  SMOTE: {X_train_res.shape[0]:,} rows")

p(f"\n  Class distribution after SMOTE:")
for cls_id in sorted(np.unique(y_train_res)):
    n = (y_train_res == cls_id).sum()
    p(f"    {cls_id} {LABEL_MAP[cls_id]:<15}  {n:>6,}")

# ══════════════════════════════════════════════════════════════════════════
sec("STEP 8 — SAVE ALL ARTIFACTS")
# ══════════════════════════════════════════════════════════════════════════

artifacts = {
    'X_train.npy'           : X_train_scaled,
    'X_test.npy'            : X_test_scaled,
    'y_train.npy'           : y_train,
    'y_test.npy'            : y_test,
    'X_train_resampled.npy' : X_train_res,
    'y_train_resampled.npy' : y_train_res,
}

for filename, arr in artifacts.items():
    path = os.path.join(OUT_DIR, filename)
    np.save(path, arr)
    p(f"  Saved: {filename}  {arr.shape}")

# ══════════════════════════════════════════════════════════════════════════
sec("SUMMARY")
# ══════════════════════════════════════════════════════════════════════════
p(f"  Input rows          : {len(df):,}")
p(f"  Features            : {len(feature_cols)}")
p(f"  Classes             : {len(LABEL_MAP)}")
p(f"  Train (raw)         : {X_train_scaled.shape[0]:,}")
p(f"  Train (SMOTE)       : {X_train_res.shape[0]:,}")
p(f"  Test                : {X_test_scaled.shape[0]:,}")
p(f"  Scaler              : RobustScaler")
p(f"  Imbalance handling  : SMOTE + class_weights")
p()
p(f"  Output directory    : {OUT_DIR}")
p()
p(f"  ✓ Ready for XGBoost training  →  load X_train_resampled.npy + y_train_resampled.npy")
p(f"  ✓ Ready for Encoder SLM       →  load X_train.npy + y_train.npy + class_weights.pkl")
p(f"  ✓ Evaluation                  →  load X_test.npy + y_test.npy")

# Save report
with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

p()
p(f"  Report saved: {REPORT}")
print(f"\nDone — all artifacts in: {OUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════
# USAGE GUIDE (printed at end)
# ══════════════════════════════════════════════════════════════════════════
print("""
─────────────────────────────────────────────────────────────
  LOADING ARTIFACTS IN YOUR TRAINING SCRIPTS
─────────────────────────────────────────────────────────────

  import numpy as np, pickle, json

  # Features & labels
  X_train = np.load('preprocessed/X_train_resampled.npy')   # SMOTE balanced
  y_train = np.load('preprocessed/y_train_resampled.npy')
  X_test  = np.load('preprocessed/X_test.npy')
  y_test  = np.load('preprocessed/y_test.npy')

  # For Encoder SLM (no SMOTE, use class weights instead):
  X_train_raw = np.load('preprocessed/X_train.npy')
  y_train_raw = np.load('preprocessed/y_train.npy')

  # Scaler (for inference on new data)
  with open('preprocessed/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)
  X_new_scaled = scaler.transform(X_new)

  # Class weights
  with open('preprocessed/class_weights.pkl', 'rb') as f:
      weights = pickle.load(f)
  class_weights = weights['class_weights']          # dict {0:w, 1:w, ...}
  xgb_weight    = weights['xgb_scale_pos_weight']   # for binary XGBoost

  # Feature names
  with open('preprocessed/feature_names.txt') as f:
      feature_names = f.read().splitlines()

  # Label map
  with open('preprocessed/label_map.json') as f:
      label_map = json.load(f)    # {'0':'benign', '1':'slowloris', ...}

─────────────────────────────────────────────────────────────
""")
