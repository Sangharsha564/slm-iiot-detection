"""
Preprocessing Pipeline v3 — Binary SLM (Benign vs Slowloris)
=============================================================
Changes from v2:
  - Loads directly from source files: benign_samples_10sec.csv + attack_samples_10sec.csv
  - Binary classification only: benign + Slowloris rows only
  - All other attack classes excluded at load time
  - Binary label encoding: 0 = benign, 1 = slowloris
  - XGBoost feature selection uses binary:logistic objective
  - Class weights recalculated for binary problem (~103x for Slowloris)
  - No XGBoost gate — pure SLM pipeline

Unchanged from v2:
  - Correlation removal (>0.95 threshold, protected features)
  - Log1p transformation (skewness > 2)
  - Stratified 80/20 train/test split (seed=42)
  - RobustScaler (fit on train only)
  - MI + XGBoost union feature selection (top 20 each)
  - 6 Boolean domain flags
  - Key-value verbalization format

Steps:
  1.  Load benign + Slowloris from source files
  2.  Drop irrelevant columns
  3.  Encode binary labels (0=benign, 1=slowloris)
  4.  Separate features X and target y
  5.  Remove highly correlated features (>0.95)
  6.  Log1p transform on skewed features
  7.  Stratified 80/20 train/test split
  8.  RobustScaler (fit on train only)
  9.  Combined feature selection (XGBoost binary + MI, top 20)
  10. Compute binary class weights
  11. Compute Boolean domain flags + thresholds
  12. Save all artifacts

Run from project root:
    python src/preprocessing/preprocess_binary.py
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
ROOT          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ATTACK_FILE   = os.path.join(ROOT, 'dataset', 'attack_samples_10sec.csv')
BENIGN_FILE   = os.path.join(ROOT, 'dataset', 'benign_samples_10sec.csv')
OUT_DIR       = os.path.join(ROOT, 'dataset', 'preprocessed_binary')
REPORT        = os.path.join(OUT_DIR, 'preprocessing_report_binary.txt')
SEED          = 42

os.makedirs(OUT_DIR, exist_ok=True)

lines = []
def p(msg=''):  print(msg); lines.append(str(msg))
def sec(title): p(); p('═'*65); p(f'  {title}'); p('═'*65)

for fpath in [ATTACK_FILE, BENIGN_FILE]:
    if not os.path.exists(fpath):
        print(f"ERROR: {fpath} not found.")
        sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 1 — LOAD BENIGN + SLOWLORIS FROM SOURCE FILES")
# ══════════════════════════════════════════════════════════════════════════════
p("  Loading benign_samples_10sec.csv ...")
df_benign = pd.read_csv(BENIGN_FILE, low_memory=False)
df_benign.replace([np.inf, -np.inf], np.nan, inplace=True)
p(f"  Benign rows    : {len(df_benign):,}")

p("  Loading attack_samples_10sec.csv — filtering to Slowloris only ...")
df_attack = pd.read_csv(ATTACK_FILE, low_memory=False)
df_attack.replace([np.inf, -np.inf], np.nan, inplace=True)
p(f"  Total attack rows : {len(df_attack):,}")

slowloris_mask = df_attack['label3'].str.lower().str.contains('slowloris', na=False)
df_slowloris   = df_attack[slowloris_mask].copy()
p(f"  Slowloris rows    : {len(df_slowloris):,}")
p(f"  Other attacks     : {(~slowloris_mask).sum():,}  ← excluded")
p()
p("  Slowloris breakdown:")
for label, cnt in df_slowloris['label3'].value_counts().items():
    p(f"    {label:<30} {cnt}")

df = pd.concat([df_benign, df_slowloris], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle
p(f"\n  Combined (shuffled) rows : {len(df):,}")
p(f"  Columns                  : {len(df.columns)}")

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
sec("STEP 3 — ENCODE BINARY LABELS")
# ══════════════════════════════════════════════════════════════════════════════
#  Binary label:
#    0 = benign    (normal traffic)
#    1 = slowloris (slow-rate HTTP DoS — the attack we detect)

LABEL_MAP = {0: 'benign', 1: 'slowloris'}
N_CLASSES = 2

# Encode binary labels
df['model_label'] = 0  # benign default
df.loc[df['label3'].str.lower().str.strip().str.contains('slowloris', na=False), 'model_label'] = 1

counts = df['model_label'].value_counts().sort_index()
p(f"\n  {'ID':>3}  {'Name':<12}  {'Count':>7}  {'%':>6}")
p(f"  {'─'*35}")
for cls_id, cnt in counts.items():
    p(f"  {cls_id:>3}  {LABEL_MAP[cls_id]:<12}  {cnt:>7,}  {cnt/len(df)*100:>5.1f}%")
p(f"\n  Class imbalance ratio: {counts[0]/counts[1]:.1f}:1 (benign:slowloris)")

with open(os.path.join(OUT_DIR, 'label_map.json'), 'w') as f:
    json.dump(LABEL_MAP, f, indent=2)
p("  Saved: label_map.json")

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

X_raw = df[feature_cols].values.astype(np.float32)
y     = df['model_label'].values.astype(np.int32)

# Keep a copy of ALL features before correlation removal (needed for domain flags)
X_raw_all        = X_raw.copy()
feature_cols_all = feature_cols.copy()

p(f"  X shape: {X_raw.shape}")
p(f"  y shape: {y.shape}")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 5 — REMOVE HIGHLY CORRELATED FEATURES  (threshold > 0.95)")
# ══════════════════════════════════════════════════════════════════════════════
PROTECTED_FEATURES = [
    'network_tcp-flags-psh_count',   # 812x higher in Slowloris vs benign
    'network_payload-length_avg',    # small payload is a Slowloris characteristic
    'network_ip-flags_min',          # value=2 is unique Slowloris signature
]

p("  Computing Pearson correlation matrix ...")
corr_matrix = pd.DataFrame(X_raw, columns=feature_cols).corr().abs()
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper.columns
           if any(upper[col] > 0.95) and col not in PROTECTED_FEATURES]
p(f"  Protected features (never dropped): {PROTECTED_FEATURES}")
p(f"  Features dropped (correlation > 0.95): {len(to_drop)}")
for c in to_drop:
    p(f"    - {c}")

keep_cols    = [c for c in feature_cols if c not in to_drop]
keep_idx     = [feature_cols.index(c) for c in keep_cols]
X_raw        = X_raw[:, keep_idx]
feature_cols = keep_cols

p(f"\n  Features after correlation removal: {len(feature_cols)}")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 6 — LOG1P TRANSFORMATION  (skewed features)")
# ══════════════════════════════════════════════════════════════════════════════
p("  Identifying skewed features (skewness > 2) ...")
df_feat     = pd.DataFrame(X_raw, columns=feature_cols)
skewness    = df_feat.skew()
skewed_cols = skewness[skewness.abs() > 2].index.tolist()

p(f"  Skewed features to log-transform: {len(skewed_cols)}")
for c in skewed_cols[:10]:
    p(f"    {c:<45}  skew={skewness[c]:.1f}")
if len(skewed_cols) > 10:
    p(f"    ... and {len(skewed_cols)-10} more")

X_log       = X_raw.copy()
skewed_idx  = [feature_cols.index(c) for c in skewed_cols]
X_log[:, skewed_idx] = np.log1p(np.abs(X_raw[:, skewed_idx]))

p(f"\n  Log1p applied to {len(skewed_cols)} features")

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
    p(f"    {cls_id} ({LABEL_MAP[cls_id]:<10}): {n:>5,}  ({n/len(y_train)*100:.2f}%)")
p(f"\n  Class distribution in TEST:")
for cls_id in sorted(np.unique(y_test)):
    n = (y_test == cls_id).sum()
    p(f"    {cls_id} ({LABEL_MAP[cls_id]:<10}): {n:>5,}  ({n/len(y_test)*100:.2f}%)")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 8 — ROBUST SCALER  (fit on train, transform both)")
# ══════════════════════════════════════════════════════════════════════════════
p("  Fitting RobustScaler on log-transformed training data ...")
scaler         = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_test_scaled  = scaler.transform(X_test_log)

p(f"  Train — mean: {X_train_scaled.mean():.4f}  std: {X_train_scaled.std():.4f}")
p(f"  Test  — mean: {X_test_scaled.mean():.4f}  std: {X_test_scaled.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 9 — FEATURE SELECTION  (XGBoost binary + Mutual Information, top 20)")
# ══════════════════════════════════════════════════════════════════════════════
TOP_K = 20

# ── 9a. Mutual Information ────────────────────────────────────────────────────
p(f"  Computing Mutual Information scores on {X_train_scaled.shape[1]} features ...")
mi_scores  = mutual_info_classif(
    X_train_scaled, y_train,
    random_state=SEED,
    n_neighbors=3
)
mi_ranking     = np.argsort(mi_scores)[::-1]
top_mi_idx     = set(mi_ranking[:TOP_K])
top_mi_features = [feature_cols[i] for i in mi_ranking[:TOP_K]]

p(f"\n  Top {TOP_K} features by Mutual Information:")
for rank, idx in enumerate(mi_ranking[:TOP_K], 1):
    p(f"    {rank:2d}. {feature_cols[idx]:<45}  MI={mi_scores[idx]:.4f}")

# ── 9b. XGBoost importance (binary classification) ───────────────────────────
p(f"\n  Training XGBoost (binary:logistic) for feature importance ...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=-1,
    random_state=SEED,
    verbosity=0,
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handles imbalance
)
xgb_model.fit(X_train_scaled, y_train)
xgb_importance  = xgb_model.feature_importances_
xgb_ranking     = np.argsort(xgb_importance)[::-1]
top_xgb_idx     = set(xgb_ranking[:TOP_K])
top_xgb_features = [feature_cols[i] for i in xgb_ranking[:TOP_K]]

p(f"\n  Top {TOP_K} features by XGBoost importance:")
for rank, idx in enumerate(xgb_ranking[:TOP_K], 1):
    p(f"    {rank:2d}. {feature_cols[idx]:<45}  gain={xgb_importance[idx]:.4f}")

# ── 9c. Union of both methods ─────────────────────────────────────────────────
mi_rank_dict  = {i: r for r, i in enumerate(mi_ranking)}
xgb_rank_dict = {i: r for r, i in enumerate(xgb_ranking)}

combined_idx = sorted(top_mi_idx | top_xgb_idx)
avg_ranks    = {i: (mi_rank_dict.get(i, 99) + xgb_rank_dict.get(i, 99)) / 2
                for i in combined_idx}
combined_idx_sorted = sorted(combined_idx, key=lambda i: avg_ranks[i])
selected_features   = [feature_cols[i] for i in combined_idx_sorted]

# Force-add protected features if not already selected
for pf in PROTECTED_FEATURES:
    if pf in feature_cols:
        pf_idx = feature_cols.index(pf)
        if pf_idx not in combined_idx_sorted:
            combined_idx_sorted.append(pf_idx)
            avg_ranks[pf_idx] = 999
            p(f"  ★ Force-added protected feature: {pf}")

combined_idx_sorted = sorted(combined_idx_sorted, key=lambda i: avg_ranks.get(i, 999))
selected_features   = [feature_cols[i] for i in combined_idx_sorted]

p(f"\n  Combined selection (union top {TOP_K} MI + top {TOP_K} XGB + protected):")
p(f"  Total selected features: {len(selected_features)}")
p(f"\n  {'Rank':<5} {'Feature':<45} {'MI rank':>8} {'XGB rank':>9} {'Avg':>6}")
p(f"  {'─'*75}")
for rank, idx in enumerate(combined_idx_sorted, 1):
    flag = ' ★' if avg_ranks.get(idx, 0) == 999 else ''
    p(f"  {rank:<5} {feature_cols[idx]:<45} "
      f"{mi_rank_dict.get(idx, 99)+1:>8} "
      f"{xgb_rank_dict.get(idx, 99)+1:>9} "
      f"{avg_ranks.get(idx,999)+1:>6.1f}{flag}")

# Apply feature selection and refit scaler on selected features
X_train_log_sel = X_train_log[:, combined_idx_sorted]
X_test_log_sel  = X_test_log[:, combined_idx_sorted]

p(f"\n  Refitting RobustScaler on {len(combined_idx_sorted)} selected features ...")
scaler      = RobustScaler()
X_train_sel = scaler.fit_transform(X_train_log_sel)
X_test_sel  = scaler.transform(X_test_log_sel)

with open(os.path.join(OUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
p("  Saved: scaler.pkl")

X_train_raw_sel = X_train_raw[:, combined_idx_sorted]

p(f"\n  Train after selection: {X_train_sel.shape}")
p(f"  Test  after selection: {X_test_sel.shape}")

with open(os.path.join(OUT_DIR, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(selected_features))
p("  Saved: feature_names.txt")

importance_df = pd.DataFrame({
    'feature'   : selected_features,
    'mi_score'  : [mi_scores[i] for i in combined_idx_sorted],
    'xgb_score' : [xgb_importance[i] for i in combined_idx_sorted],
    'mi_rank'   : [mi_rank_dict.get(i, 99)+1 for i in combined_idx_sorted],
    'xgb_rank'  : [xgb_rank_dict.get(i, 99)+1 for i in combined_idx_sorted],
    'avg_rank'  : [avg_ranks.get(i, 999)+1 for i in combined_idx_sorted],
})
importance_df.to_csv(os.path.join(OUT_DIR, 'selected_feature_importance.csv'), index=False)
p("  Saved: selected_feature_importance.csv")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 10 — BINARY CLASS WEIGHTS  (for SLM CrossEntropyLoss)")
# ══════════════════════════════════════════════════════════════════════════════
classes_present = np.unique(y_train)
weights_array   = compute_class_weight(
    class_weight='balanced',
    classes=classes_present,
    y=y_train
)
class_weights = {int(cls): float(w) for cls, w in zip(classes_present, weights_array)}

p("  Binary class weights (balanced):")
for cls_id, w in class_weights.items():
    p(f"    {cls_id} ({LABEL_MAP[cls_id]:<10}): {w:>8.4f}x")
p(f"\n  Slowloris weight / benign weight = "
  f"{class_weights[1]/class_weights[0]:.1f}x heavier penalty on missed Slowloris")

weights_out = {
    'class_weights': class_weights,
    'label_map'    : LABEL_MAP,
    'n_classes'    : N_CLASSES,
}
with open(os.path.join(OUT_DIR, 'class_weights.pkl'), 'wb') as f:
    pickle.dump(weights_out, f)
p("  Saved: class_weights.pkl")

# ══════════════════════════════════════════════════════════════════════════════
sec("STEP 11 — BOOLEAN DOMAIN FLAGS  (SLM Slowloris signals)")
# ══════════════════════════════════════════════════════════════════════════════
# Re-split X_raw_all with same seed to get train portion for threshold calculation
X_raw_all_train, _ = train_test_split(
    X_raw_all, test_size=0.2, random_state=SEED, stratify=y
)

feat_idx_all_dict = {name: i for i, name in enumerate(feature_cols_all)}

def get_col_train(name):
    if name in feat_idx_all_dict:
        return X_raw_all_train[:, feat_idx_all_dict[name]]
    return np.zeros(len(y_train))

psh      = get_col_train('network_tcp-flags-psh_count')
ip_flags = get_col_train('network_ip-flags_min')
interval = get_col_train('network_time-delta_avg')
packets  = get_col_train('network_packets_all_count')
payload  = get_col_train('network_payload-length_avg')
syn      = get_col_train('network_tcp-flags-syn_count')

benign_mask = y_train == 0
thresholds  = {
    'HIGH_PSH'      : float(np.percentile(psh[benign_mask],      95)),
    'IP_FLAGS_2'    : 1.5,
    'SLOW_INTERVAL' : float(np.percentile(interval[benign_mask], 25)),
    'HIGH_VOLUME'   : float(np.percentile(packets[benign_mask],  95)),
    'LOW_PAYLOAD'   : float(np.percentile(payload[benign_mask],  25)),
    'HIGH_SYN'      : float(np.percentile(syn[benign_mask],      95)),
}

p("  Boolean domain flag thresholds (derived from benign class percentiles):")
p(f"    HIGH_PSH      (psh_count > {thresholds['HIGH_PSH']:.0f})")
p(f"    IP_FLAGS_2    (ip_flags_min >= 2)          ← exact Slowloris signature")
p(f"    SLOW_INTERVAL (time_delta_avg < {thresholds['SLOW_INTERVAL']:.5f})")
p(f"    HIGH_VOLUME   (packets_total > {thresholds['HIGH_VOLUME']:.0f})")
p(f"    LOW_PAYLOAD   (payload_avg < {thresholds['LOW_PAYLOAD']:.2f})")
p(f"    HIGH_SYN      (syn_count > {thresholds['HIGH_SYN']:.0f})")

sl_mask = y_train == 1
if sl_mask.sum() > 0:
    p(f"\n  Slowloris flag activation rates on training set:")
    p(f"    HIGH_PSH   : {(psh[sl_mask]      > thresholds['HIGH_PSH']).mean()*100:.0f}%")
    p(f"    IP_FLAGS_2 : {(ip_flags[sl_mask] >= thresholds['IP_FLAGS_2']).mean()*100:.0f}%")
    p(f"    LOW_PAYLOAD: {(payload[sl_mask]   < thresholds['LOW_PAYLOAD']).mean()*100:.0f}%")

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
n_feat = len(selected_features)
p(f"  Source files               : benign_samples_10sec.csv + attack_samples_10sec.csv")
p(f"  Input rows                 : {len(df):,}  ({(y==0).sum():,} benign + {(y==1).sum()} slowloris)")
p(f"  Features (after corr)      : {len(feature_cols)}")
p(f"  Features (after selection) : {n_feat}")
p(f"  Classes                    : 2  (binary: benign=0, slowloris=1)")
p(f"  Train rows                 : {X_train_sel.shape[0]:,}")
p(f"  Test rows                  : {X_test_sel.shape[0]:,}")
p(f"  Slowloris in train         : {(y_train==1).sum()}")
p(f"  Slowloris in test          : {(y_test==1).sum()}")
p(f"  Imbalance ratio            : ~{(y_train==0).sum()/(y_train==1).sum():.0f}:1")
p(f"  Scaling                    : Log1p → RobustScaler")
p(f"  Imbalance handling         : Class weights in SLM loss (no SMOTE)")
p(f"  Domain flags               : 6 Boolean Slowloris signals")
p()
p(f"  Artifacts saved to: {OUT_DIR}")
p(f"    ✓ X_train.npy, X_test.npy   ({n_feat} features)")
p(f"    ✓ y_train.npy, y_test.npy   (binary labels: 0=benign, 1=slowloris)")
p(f"    ✓ scaler.pkl")
p(f"    ✓ label_map.json            {{0: benign, 1: slowloris}}")
p(f"    ✓ feature_names.txt         ({n_feat} features)")
p(f"    ✓ class_weights.pkl")
p(f"    ✓ domain_flags_thresholds.json")
p(f"    ✓ log_transform_info.json")
p(f"    ✓ selected_feature_importance.csv")

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
p(f"\n  Report saved: {REPORT}")
print(f"\nDone — all artifacts in: {OUT_DIR}")
