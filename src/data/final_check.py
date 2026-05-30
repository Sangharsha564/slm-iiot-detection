"""
Final Dataset Confidence Check — combined_shuffled.csv
Run: python3 final_check.py
"""

import os, sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, 'combined_shuffled.csv')
OUT_FILE = os.path.join(BASE_DIR, 'final_check_output.txt')

lines = []
def p(msg=''):   print(msg); lines.append(str(msg))
def sec(t):      p(); p(f"{'─'*65}"); p(f"  {t}"); p(f"{'─'*65}")

if not os.path.exists(CSV_FILE):
    print(f"ERROR: {CSV_FILE} not found."); sys.exit(1)

print("Loading...")
df = pd.read_csv(CSV_FILE, low_memory=False)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
str_cols = [c for c in df.columns if df[c].dtype == object]

# ── 1. basic health ────────────────────────────────────────────
sec("1. BASIC HEALTH CHECK")
p(f"  Rows             : {len(df):,}")
p(f"  Columns          : {len(df.columns)}")
p(f"  Numeric cols     : {len(num_cols)}")
p(f"  String cols      : {len(str_cols)}")
p(f"  Missing values   : {df.isnull().sum().sum():,}")
p(f"  Duplicate rows   : {df.duplicated().sum():,}")
p(f"  Memory usage     : {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

# shuffle check
first5 = df['label2'].head(5).tolist()
p(f"\n  First 5 labels (shuffle check): {first5}")
unique_in_first20 = df['label2'].head(20).nunique()
p(f"  Unique classes in first 20 rows: {unique_in_first20} (good if > 1)")

# ── 2. class distribution ──────────────────────────────────────
sec("2. CLASS DISTRIBUTION (all label columns)")

for col in ['label1', 'label2', 'label3', 'label4']:
    if col not in df.columns:
        continue
    counts = df[col].value_counts()
    if len(counts) > 12:
        p(f"\n  [{col}] — {len(counts)} unique values (showing top 12)")
    else:
        p(f"\n  [{col}] — {len(counts)} unique values")
    for lbl, cnt in counts.head(12).items():
        bar = '█' * max(1, int(cnt / len(df) * 40))
        p(f"    {lbl:<40} {cnt:>6,}  {cnt/len(df)*100:>5.1f}%  {bar}")

# slowloris specifically
p(f"\n  ── Slowloris rows (label3 contains 'slowloris') ──")
sl = df[df['label3'].str.contains('slowloris', case=False, na=False)]
p(f"  Total: {len(sl)}")
for lbl, cnt in sl['label3'].value_counts().items():
    p(f"    {lbl:<40} {cnt:>4}  (dos: {(sl[sl['label3']==lbl]['label2']=='dos').sum()}, "
      f"ddos: {(sl[sl['label3']==lbl]['label2']=='ddos').sum()})")

# ── 3. data quality ────────────────────────────────────────────
sec("3. DATA QUALITY")

# missing per column
missing = df.isnull().sum()
missing = missing[missing > 0]
p(f"  Columns with missing values: {len(missing)}")
if len(missing) > 0:
    for col, cnt in missing.items():
        p(f"    {col}: {cnt:,} ({cnt/len(df)*100:.2f}%)")

# zero variance
zero_var = []
for c in num_cols:
    try:
        if df[c].var(skipna=True) < 1e-10:
            zero_var.append(c)
    except: pass
p(f"  Zero-variance numeric cols ({len(zero_var)}): {zero_var}")

# negative values in rate features
neg_issues = []
for c in num_cols:
    if df[c].min() < -0.001:
        neg_issues.append(f"{c} (min={df[c].min():.4g})")
p(f"  Cols with unexpected negatives: {neg_issues if neg_issues else 'None'}")

# ── 4. numeric feature summary ─────────────────────────────────
sec("4. NUMERIC FEATURE SUMMARY")
stats = df[num_cols].describe().T[['mean','std','min','50%','max']]
stats.columns = ['mean','std','min','median','max']
p(stats.to_string(float_format=lambda x: f"{x:.4g}"))

# ── 5. key features — per class medians ───────────────────────
sec("5. KEY FEATURES — MEDIAN PER CLASS (label2)")

key_features = [
    'network_tcp-flags-syn_count',
    'network_tcp-flags-rst_count',
    'network_tcp-flags-ack_count',
    'network_tcp-flags-fin_count',
    'network_packets_all_count',
    'network_mss_avg',
    'network_ip-flags_min',
    'network_ip-flags_avg',
    'network_time-delta_avg',
    'network_time-delta_max',
    'network_payload-length_avg',
    'network_packet-size_avg',
    'network_interval-packets',
    'network_window-size_min',
    'network_ttl_avg',
]
key_features = [f for f in key_features if f in df.columns]

classes = df['label2'].value_counts().index.tolist()
hdr = f"  {'Feature':<35}"
for c in classes: hdr += f"  {c[:10]:>10}"
p(hdr)
p(f"  {'·'*(35 + 12*len(classes))}")

for feat in key_features:
    row = f"  {feat:<35}"
    for c in classes:
        med = df[df['label2']==c][feat].median()
        row += f"  {med:>10.4g}"
    p(row)

# ── 6. slowloris feature profile ──────────────────────────────
sec("6. SLOWLORIS FEATURE PROFILE vs BENIGN")

sl_mask = df['label3'].str.contains('slowloris', case=False, na=False)
bn_mask = df['label2'] == 'benign'

p(f"  {'Feature':<35} {'Benign med':>12} {'SL med':>12} {'SL mean':>12} {'Ratio':>8}")
p(f"  {'·'*83}")
for feat in key_features:
    bn_med = df[bn_mask][feat].median()
    sl_med = df[sl_mask][feat].median()
    sl_mean= df[sl_mask][feat].mean()
    ratio  = sl_mean / (df[bn_mask][feat].mean() + 1e-9)
    flag   = ' ◀' if abs(ratio) > 10 or ratio < 0.1 else ''
    p(f"  {feat:<35} {bn_med:>12.4g} {sl_med:>12.4g} {sl_mean:>12.4g} {ratio:>7.1f}x{flag}")

# ── 7. string columns overview ─────────────────────────────────
sec("7. STRING COLUMNS (to drop before training)")
for col in str_cols:
    n_unique = df[col].nunique()
    sample   = df[col].iloc[0] if len(df) > 0 else ''
    p(f"  {col:<45} unique={n_unique:>5}   sample: {str(sample)[:40]}")

# ── 8. columns to keep for training ───────────────────────────
sec("8. TRAINING-READY COLUMN LIST")

drop_always = [
    'device_name','device_mac','timestamp','timestamp_start','timestamp_end',
    'network_ips_all','network_ips_dst','network_ips_src',
    'network_macs_all','network_macs_dst','network_macs_src',
    'network_ports_all','network_ports_dst','network_ports_src',
    'network_protocols_all','network_protocols_dst','network_protocols_src',
    'network_tcp-flags-urg_count','log_data-types',
    'label_full','label1','binary_label','_label',
    'flow_duration','flow_duration_sec',
]
drop_always = [c for c in drop_always if c in df.columns]

label_cols = ['label2','label3','label4','model_label']
label_cols = [c for c in label_cols if c in df.columns]

feature_cols = [c for c in df.columns
                if c not in drop_always and c not in label_cols
                and pd.api.types.is_numeric_dtype(df[c])]

p(f"  Columns to DROP  : {len(drop_always)}")
for c in drop_always: p(f"    - {c}")

p(f"\n  Label columns    : {label_cols}")
p(f"\n  Feature columns for training: {len(feature_cols)}")
for i, c in enumerate(feature_cols):
    p(f"    {i+1:3d}. {c}")

# ── 9. final readiness verdict ────────────────────────────────
sec("9. DATASET READINESS VERDICT")

checks = {
    'No missing values'         : df.isnull().sum().sum() == 0,
    'No duplicate rows'         : df.duplicated().sum() == 0,
    'Shuffled (mixed classes)'  : unique_in_first20 > 2,
    'Both benign & attack rows' : ('benign' in df['label2'].values and
                                   'dos' in df['label2'].values),
    'Slowloris rows present'    : sl_mask.sum() > 0,
    'Labels available'          : all(c in df.columns for c in ['label2','label3']),
    'Numeric features present'  : len(feature_cols) > 10,
}

all_pass = True
for check, result in checks.items():
    status = '✓ PASS' if result else '✗ FAIL'
    if not result: all_pass = False
    p(f"  {status}  {check}")

p()
if all_pass:
    p("  ✓ Dataset is ready for preprocessing and model training.")
else:
    p("  ✗ Fix the failed checks before proceeding.")

p(f"\n  Total feature columns for training : {len(feature_cols)}")
p(f"  Total rows                          : {len(df):,}")
p(f"  Slowloris rows                      : {sl_mask.sum()} ({sl_mask.sum()/len(df)*100:.2f}%)")
p(f"  Imbalance ratio (Slowloris)         : {(~sl_mask).sum() // sl_mask.sum()}:1")

# ── save ──────────────────────────────────────────────────────
sec("SAVED")
p(f"  {OUT_FILE}")
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\nDone → final_check_output.txt")
