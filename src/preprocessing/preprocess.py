"""
Preprocessing Pipeline v2 — SLM-Optimised
==========================================
Improvements over v1:
  - 8-class labels (MitM and malware separated from "other")
  - Correlation removal (drop features with >0.95 pairwise correlation)
  - Log1p transformation for heavily skewed features (before scaling)
  - RobustScaler (after log1p)
  - Combined feature selection: XGBoost importance + Mutual Information (top 20)
  - Boolean domain flags (6 attack-specific binary signals for SLM input)
  - Class weights for SLM training (no SMOTE — imbalance handled via loss function)
  - Key-value verbalization format saved alongside artifacts

Steps:
  1.  Load dataset
  2.  Drop irrelevant columns
  3.  Encode 8-class labels
  4.  Separate features X and target y
  5.  Remove highly correlated features (>0.95)
  6.  Log1p transform on skewed features
  7.  Stratified 80/20 train/test split
  8.  RobustScaler (fit on train only)
  9.  Combined feature selection (XGBoost importance + MI, top 20)
  10. Compute class weights
  11. Compute Boolean domain flags + thresholds
  12. Save all artifacts

Run from project root:
    python src/preprocessing/preprocess.py
"""

import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb

# ── project root ──────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_FILE = os.path.join(ROOT, 'dataset', 'combined_shuffled.csv')
OUT_DIR  = os.path.join(ROOT, 'dataset', 'preprocessed')
REPORT   = os.path.join(OUT_DIR, 'preprocessing_report_v2.txt')
SEED     = 42

os.makedirs(OUT_DIR, exist_ok=True)

lines = []
def p(msg=''):  print(msg); lines.append(str(msg))
def sec(title): p(); p('═'*65); p(f'  {title}'); p('═'*65)

if not os.path.exists(CSV_FILE):
    print(f"ERROR: {CSV_FILE} not found.")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 1 — LOAD DATASET")
# ══════════════════════════════════════════════════════════════════════════════
p("  Loading combined_shuffled.csv ...")
df = pd.read_csv(CSV_FILE, low_memory=False)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
p(f"  Rows    : {len(df):,}")
p(f"  Columns : {len(df.columns)}")
p(f"  Missing : {df.isnull().sum().sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 2 — DROP IRRELEVANT COLUMNS")
# ══════════════════════════════════════════════════════════════════════════════
DROP_ALWAYS = [
    'device_name', 'device_mac',
    'timestamp', 'timestamp_start', 'timestamp_end',
    'flow_duration', 'flow_duration_sec',
    'network_ips_all',  'network_ips_dst',  'network_ips_src',
    'network_macs_all', 'network_macs_dst', 'network_macs_src',
    'network_ports_all','network_ports_dst','network_ports_src',
    'network_protocols_all','network_protocols_dst','network_protocols_src',
    'network_tcp-flags-urg_count',
    'log_data-types',
    'label_full', 'label1', 'binary_label', '_label',
]
DROP_ALWAYS = [c for c in DROP_ALWAYS if c in df.columns]
p(f"  Dropping {len(DROP_ALWAYS)} identifier/non-feature columns")
df.drop(columns=DROP_ALWAYS, inplace=True)
p(f"  Remaining columns: {len(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 3 — ENCODE 8-CLASS LABELS")
# ══════════════════════════════════════════════════════════════════════════════
#
#  Class  ID   label2       Description
#  ─────  ──   ──────────   ────────────────────────────────────
#  benign  0   benign       Normal traffic
#  slowloris 1 dos/ddos     Slow-rate HTTP attack (key target)
#  dos_other 2 dos          Other DoS (not slowloris)
#  ddos_other 3 ddos        Other DDoS (not slowloris)
#  recon   4   recon        Reconnaissance / port scanning
#  mitm    5   mitm         Man-in-the-middle (ARP spoofing)
#  malware 6   malware      Mirai botnet variants
#  other   7   web+brute    Web attacks + bruteforce
#

LABEL_MAP = {
    0: 'benign',
    1: 'slowloris',
    2: 'dos_other',
    3: 'ddos_other',
    4: 'recon',
    5: 'mitm',
    6: 'malware',
    7: 'other',
}
N_CLASSES = len(LABEL_MAP)

def encode_label(row):
    l2 = str(row.get('label2', '')).lower().strip()
    l3 = str(row.get('label3', '')).lower().strip()
    if l2 == 'benign':        return 0
    if 'slowloris' in l3:     return 1
    if l2 == 'dos':           return 2
    if l2 == 'ddos':          return 3
    if l2 == 'recon':         return 4
    if l2 == 'mitm':          return 5
    if l2 == 'malware':       return 6
    return 7  # web + bruteforce

p("  Applying 8-class label encoding ...")
df['model_label'] = df.apply(encode_label, axis=1)

counts = df['model_label'].value_counts().sort_index()
p(f"\n  {'ID':>3}  {'Name':<15}  {'Count':>7}  {'%':>6}")
p(f"  {'─'*40}")
for cls_id, cnt in counts.items():
    p(f"  {cls_id:>3}  {LABEL_MAP[cls_id]:<15}  {cnt:>7,}  {cnt/len(df)*100:>5.1f}%")

with open(os.path.join(OUT_DIR, 'label_map.json'), 'w') as f:
    json.dump(LABEL_MAP, f, indent=2)
p("\n  Saved: label_map.json")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 4 — SEPARATE FEATURES AND TARGET")
# ══════════════════════════════════════════════════════════════════════════════
LABEL_COLS = ['label2', 'label3', 'label4', 'model_label']
LABEL_COLS = [c for c in LABEL_COLS if c in df.columns]

feature_cols = [
    c for c in df.columns
    if c not in LABEL_COLS
    and pd.api.types.is_numeric_dtype(df[c])
]

p(f"  Feature columns : {len(feature_cols)}")

# Fill any remaining NaN with median
missing = df[feature_cols].isnull().sum().sum()
if missing > 0:
    p(f"  Filling {missing} NaN values with column median ...")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# Save raw unscaled values for computing domain flags later
X_raw = df[feature_cols].values.astype(np.float32)
y     = df['model_label'].values.astype(np.int32)

# Keep a copy of ALL features before correlation removal — needed for domain flags
# because some key Slowloris signals (psh_count, payload_avg) may be dropped
X_raw_all     = X_raw.copy()
feature_cols_all = feature_cols.copy()

p(f"  X shape: {X_raw.shape}")
p(f"  y shape: {y.shape}")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 5 — REMOVE HIGHLY CORRELATED FEATURES  (threshold > 0.95)")
# ══════════════════════════════════════════════════════════════════════════════
# Correlated features are redundant — they add noise without information.
# We keep the first feature in each correlated pair.

p("  Computing Pearson correlation matrix ...")
corr_matrix = pd.DataFrame(X_raw, columns=feature_cols).corr().abs()
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
p(f"  Features with correlation > 0.95 to another feature: {len(to_drop)}")
for c in to_drop:
    p(f"    - {c}")

# Remove correlated features
keep_cols = [c for c in feature_cols if c not in to_drop]
keep_idx  = [feature_cols.index(c) for c in keep_cols]
X_raw     = X_raw[:, keep_idx]
feature_cols = keep_cols

p(f"\n  Features after correlation removal: {len(feature_cols)}")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 6 — LOG1P TRANSFORMATION  (skewed features)")
# ══════════════════════════════════════════════════════════════════════════════
# Network traffic features like packet counts and byte counts follow
# power-law distributions (range: 0 to 1,488,000).
# Log1p = log(x+1) handles zero values gracefully.
# Applied before scaling to compress extreme ranges.

p("  Identifying skewed features (skewness > 2) ...")
df_feat   = pd.DataFrame(X_raw, columns=feature_cols)
skewness  = df_feat.skew()
skewed_cols = skewness[skewness.abs() > 2].index.tolist()

p(f"  Skewed features to log-transform: {len(skewed_cols)}")
for c in skewed_cols[:10]:
    p(f"    {c:<45}  skew={skewness[c]:.1f}")
if len(skewed_cols) > 10:
    p(f"    ... and {len(skewed_cols)-10} more")

# Apply log1p to skewed features only (non-negative network traffic values)
X_log = X_raw.copy()
skewed_idx = [feature_cols.index(c) for c in skewed_cols]
X_log[:, skewed_idx] = np.log1p(np.abs(X_raw[:, skewed_idx]))

p(f"\n  Log1p applied to {len(skewed_cols)} features")

# Save which features were log-transformed (needed at inference)
log_transform_info = {
    'skewed_features': skewed_cols,
    'skewness_threshold': 2.0
}
with open(os.path.join(OUT_DIR, 'log_transform_info.json'), 'w') as f:
    json.dump(log_transform_info, f, indent=2)
p("  Saved: log_transform_info.json")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 7 — STRATIFIED TRAIN / TEST SPLIT  (80 / 20)")
# ══════════════════════════════════════════════════════════════════════════════
X_train_log, X_test_log, y_train, y_test, X_train_raw, X_test_raw = train_test_split(
    X_log, y, X_raw,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

p(f"  Train : {X_train_log.shape[0]:,} rows")
p(f"  Test  : {X_test_log.shape[0]:,} rows")
p(f"\n  Class distribution in TRAIN:")
for cls_id in sorted(np.unique(y_train)):
    n = (y_train == cls_id).sum()
    p(f"    {cls_id} ({LABEL_MAP[cls_id]:<12}): {n:>5,}  ({n/len(y_train)*100:.1f}%)")
p(f"\n  Class distribution in TEST:")
for cls_id in sorted(np.unique(y_test)):
    n = (y_test == cls_id).sum()
    p(f"    {cls_id} ({LABEL_MAP[cls_id]:<12}): {n:>5,}  ({n/len(y_test)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 8 — ROBUST SCALER  (fit on train, transform both)")
# ══════════════════════════════════════════════════════════════════════════════
p("  Fitting RobustScaler on log-transformed training data ...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_test_scaled  = scaler.transform(X_test_log)

p(f"  Train — mean: {X_train_scaled.mean():.4f}  std: {X_train_scaled.std():.4f}")
p(f"  Test  — mean: {X_test_scaled.mean():.4f}  std: {X_test_scaled.std():.4f}")

with open(os.path.join(OUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
p("  Saved: scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 9 — FEATURE SELECTION  (XGBoost importance + Mutual Information, top 20)")
# ══════════════════════════════════════════════════════════════════════════════
# Two independent methods — take union of top 20 from each.
# This captures both linear (XGBoost gain) and non-linear (MI) dependencies.

TOP_K = 20

# ── 9a. Mutual Information ────────────────────────────────────────────────────
p(f"  Computing Mutual Information scores on {X_train_scaled.shape[1]} features ...")
mi_scores = mutual_info_classif(
    X_train_scaled, y_train,
    random_state=SEED,
    n_neighbors=3
)
mi_ranking = np.argsort(mi_scores)[::-1]
top_mi_idx = set(mi_ranking[:TOP_K])
top_mi_features = [feature_cols[i] for i in mi_ranking[:TOP_K]]

p(f"\n  Top {TOP_K} features by Mutual Information:")
for rank, idx in enumerate(mi_ranking[:TOP_K], 1):
    p(f"    {rank:2d}. {feature_cols[idx]:<45}  MI={mi_scores[idx]:.4f}")

# ── 9b. XGBoost importance ────────────────────────────────────────────────────
p(f"\n  Training lightweight XGBoost for feature importance ...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=-1,
    random_state=SEED,
    verbosity=0,
    eval_metric='mlogloss',
)
xgb_model.fit(X_train_scaled, y_train)
xgb_importance = xgb_model.feature_importances_
xgb_ranking    = np.argsort(xgb_importance)[::-1]
top_xgb_idx    = set(xgb_ranking[:TOP_K])
top_xgb_features = [feature_cols[i] for i in xgb_ranking[:TOP_K]]

p(f"\n  Top {TOP_K} features by XGBoost importance:")
for rank, idx in enumerate(xgb_ranking[:TOP_K], 1):
    p(f"    {rank:2d}. {feature_cols[idx]:<45}  gain={xgb_importance[idx]:.4f}")

# ── 9c. Union of both methods ─────────────────────────────────────────────────
combined_idx = sorted(top_mi_idx | top_xgb_idx)
selected_features = [feature_cols[i] for i in combined_idx]

# Rank combined features by average rank from both methods
mi_rank_dict  = {i: r for r, i in enumerate(mi_ranking)}
xgb_rank_dict = {i: r for r, i in enumerate(xgb_ranking)}
avg_ranks = {i: (mi_rank_dict.get(i, 99) + xgb_rank_dict.get(i, 99)) / 2
             for i in combined_idx}
combined_idx_sorted = sorted(combined_idx, key=lambda i: avg_ranks[i])
selected_features   = [feature_cols[i] for i in combined_idx_sorted]

p(f"\n  Combined selection (union of top {TOP_K} from each):")
p(f"  Total selected features: {len(selected_features)}")
p(f"\n  {'Rank':<5} {'Feature':<45} {'MI rank':>8} {'XGB rank':>9} {'Avg':>6}")
p(f"  {'─'*75}")
for rank, idx in enumerate(combined_idx_sorted, 1):
    p(f"  {rank:<5} {feature_cols[idx]:<45} "
      f"{mi_rank_dict.get(idx, 99)+1:>8} "
      f"{xgb_rank_dict.get(idx, 99)+1:>9} "
      f"{avg_ranks[idx]+1:>6.1f}")

# Apply feature selection
X_train_sel = X_train_scaled[:, combined_idx_sorted]
X_test_sel  = X_test_scaled[:, combined_idx_sorted]

# Also keep unscaled selected training values for domain flag thresholds
X_train_raw_sel = X_train_raw[:, combined_idx_sorted]

p(f"\n  Train after selection: {X_train_sel.shape}")
p(f"  Test  after selection: {X_test_sel.shape}")

# Save selected feature names
with open(os.path.join(OUT_DIR, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(selected_features))
p("  Saved: feature_names.txt")

# Save full feature importance details
importance_df = pd.DataFrame({
    'feature'   : selected_features,
    'mi_score'  : [mi_scores[i] for i in combined_idx_sorted],
    'xgb_score' : [xgb_importance[i] for i in combined_idx_sorted],
    'mi_rank'   : [mi_rank_dict.get(i, 99)+1 for i in combined_idx_sorted],
    'xgb_rank'  : [xgb_rank_dict.get(i, 99)+1 for i in combined_idx_sorted],
    'avg_rank'  : [avg_ranks[i]+1 for i in combined_idx_sorted],
})
importance_df.to_csv(os.path.join(OUT_DIR, 'selected_feature_importance.csv'), index=False)
p("  Saved: selected_feature_importance.csv")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 10 — CLASS WEIGHTS  (for SLM CrossEntropyLoss)")
# ══════════════════════════════════════════════════════════════════════════════
classes_present = np.unique(y_train)
weights_array   = compute_class_weight(
    class_weight='balanced',
    classes=classes_present,
    y=y_train
)
class_weights = {int(cls): float(w)
                 for cls, w in zip(classes_present, weights_array)}

p("  Class weights (balanced) — higher = rarer class penalised more:")
for cls_id, w in class_weights.items():
    bar = '█' * int(w)
    p(f"    {cls_id} ({LABEL_MAP[cls_id]:<12}): {w:>7.4f}  {bar}")

weights_out = {
    'class_weights': class_weights,
    'label_map'    : LABEL_MAP,
    'n_classes'    : N_CLASSES,
}
with open(os.path.join(OUT_DIR, 'class_weights.pkl'), 'wb') as f:
    pickle.dump(weights_out, f)
p("  Saved: class_weights.pkl")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 11 — BOOLEAN DOMAIN FLAGS  (SLM attack signals)")
# ══════════════════════════════════════════════════════════════════════════════
# 6 binary flags computed from original unscaled training values.
# Thresholds set from Slowloris profile analysis (see final_check_output.txt).
# These flags are appended to key-value verbalized text for SLM input.

# Use the FULL original feature set (before correlation removal) for domain flags
# This ensures psh_count and payload_avg are available even if they were dropped
# as correlated features — they are critical Slowloris signals.
feat_idx_all = {name: i for i, name in enumerate(feature_cols_all)}

def get_col_original(name):
    """Get training column from original unscaled data (before corr removal)."""
    if name in feat_idx_all:
        idx = feat_idx_all[name]
        # X_raw_all has all rows; get train rows using y_train indices
        return X_raw_all[train_indices, idx]
    return np.zeros(len(y_train))

# Get train indices from the split (y_train corresponds to X_train_log rows)
# We reconstruct by matching — use the split indices saved implicitly
# Simpler: re-split X_raw_all with same seed to get train portion
X_raw_all_train, _ = train_test_split(
    X_raw_all, test_size=0.2, random_state=SEED, stratify=y
)

feat_idx_all_dict = {name: i for i, name in enumerate(feature_cols_all)}

def get_col_train(name):
    """Get training column from original unscaled full feature set."""
    if name in feat_idx_all_dict:
        return X_raw_all_train[:, feat_idx_all_dict[name]]
    return np.zeros(len(y_train))

# Compute thresholds from original unscaled training data
psh      = get_col_train('network_tcp-flags-psh_count')
ip_flags = get_col_train('network_ip-flags_min')
interval = get_col_train('network_time-delta_avg')
packets  = get_col_train('network_packets_all_count')
payload  = get_col_train('network_payload-length_avg')
syn      = get_col_train('network_tcp-flags-syn_count')

# Thresholds: use 75th percentile of benign class as boundary
benign_mask = y_train == 0
thresholds = {
    'HIGH_PSH'      : float(np.percentile(psh[benign_mask],     95)),
    'IP_FLAGS_2'    : 1.5,   # exact threshold: ip_flags_min >= 2
    'SLOW_INTERVAL' : float(np.percentile(interval[benign_mask], 25)),
    'HIGH_VOLUME'   : float(np.percentile(packets[benign_mask],  95)),
    'LOW_PAYLOAD'   : float(np.percentile(payload[benign_mask],  25)),
    'HIGH_SYN'      : float(np.percentile(syn[benign_mask],      95)),
}

p("  Boolean domain flag thresholds (derived from benign class percentiles):")
p(f"    HIGH_PSH      (psh_count >{thresholds['HIGH_PSH']:.0f})           "
  f"← Slowloris: avg 812 vs benign avg 1")
p(f"    IP_FLAGS_2    (ip_flags_min >= 2)              "
  f"← Unique Slowloris signature")
p(f"    SLOW_INTERVAL (time_delta_avg < {thresholds['SLOW_INTERVAL']:.4f})      "
  f"← Slow-rate behaviour")
p(f"    HIGH_VOLUME   (packets_total > {thresholds['HIGH_VOLUME']:.0f})     "
  f"← High packet count")
p(f"    LOW_PAYLOAD   (payload_avg < {thresholds['LOW_PAYLOAD']:.1f})          "
  f"← Small payload (Slowloris)")
p(f"    HIGH_SYN      (syn_count > {thresholds['HIGH_SYN']:.0f})              "
  f"← SYN-based attacks")

# Verify flags work — check Slowloris flag activation rate
sl_mask   = y_train == 1
sl_psh    = get_col_train('network_tcp-flags-psh_count')
sl_ipfl   = get_col_train('network_ip-flags_min')

if sl_mask.sum() > 0:
    psh_rate = (sl_psh[sl_mask] > thresholds['HIGH_PSH']).mean()
    ip2_rate = (sl_ipfl[sl_mask] >= thresholds['IP_FLAGS_2']).mean()
    p(f"\n  Slowloris flag activation rates (should be high):")
    p(f"    HIGH_PSH   triggers on {psh_rate*100:.0f}% of Slowloris training samples")
    p(f"    IP_FLAGS_2 triggers on {ip2_rate*100:.0f}% of Slowloris training samples")

with open(os.path.join(OUT_DIR, 'domain_flags_thresholds.json'), 'w') as f:
    json.dump(thresholds, f, indent=2)
p("\n  Saved: domain_flags_thresholds.json")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 12 — SAVE ALL ARTIFACTS")
# ══════════════════════════════════════════════════════════════════════════════

artifacts = {
    'X_train.npy' : X_train_sel,
    'X_test.npy'  : X_test_sel,
    'y_train.npy' : y_train,
    'y_test.npy'  : y_test,
}
for fname, arr in artifacts.items():
    path = os.path.join(OUT_DIR, fname)
    np.save(path, arr)
    p(f"  Saved: {fname}  {arr.shape}")

# ══════════════════════════════════════════════════════════════════════════════
sec("SUMMARY")
# ══════════════════════════════════════════════════════════════════════════════
p(f"  Input rows             : {len(df):,}")
p(f"  Features (after corr)  : {len(feature_cols)}")
p(f"  Features (after sel)   : {len(selected_features)}")
p(f"  Classes                : {N_CLASSES}")
p(f"  Slowloris samples      : {(y_train==1).sum()} train / {(y_test==1).sum()} test")
p(f"  Train rows             : {X_train_sel.shape[0]:,}")
p(f"  Test rows              : {X_test_sel.shape[0]:,}")
p(f"  Scaling                : Log1p → RobustScaler")
p(f"  Imbalance handling     : Class weights (SLM loss function)")
p(f"  Domain flags           : 6 Boolean attack signals")
p()
p(f"  Artifacts saved to: {OUT_DIR}")
p(f"    ✓ X_train.npy, X_test.npy   ({len(selected_features)} selected features)")
p(f"    ✓ y_train.npy, y_test.npy   (8-class labels)")
p(f"    ✓ scaler.pkl                (Log1p + RobustScaler)")
p(f"    ✓ label_map.json            (8-class mapping)")
p(f"    ✓ feature_names.txt         ({len(selected_features)} selected features)")
p(f"    ✓ class_weights.pkl         (for SLM CrossEntropyLoss)")
p(f"    ✓ domain_flags_thresholds.json")
p(f"    ✓ log_transform_info.json")
p(f"    ✓ selected_feature_importance.csv")

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
p(f"\n  Report saved: {REPORT}")
print(f"\nDone — all artifacts in: {OUT_DIR}")
