"""
CIC IIoT 2025 — Dataset Checker (fixed for actual column structure)
Saves output to: dataset_check_output.txt
Run: python check_dataset.py
"""

import os, sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, 'benign_samples_10sec.csv')
OUT_FILE = os.path.join(BASE_DIR, 'dataset_check_output.txt')

lines = []
def p(msg=''):   print(msg);  lines.append(str(msg))
def sec(t):      p();  p(f"{'─'*60}");  p(f"  {t}");  p(f"{'─'*60}")

# ── load ───────────────────────────────────────────────────────
if not os.path.exists(CSV_FILE):
    print(f"ERROR: File not found:\n  {CSV_FILE}"); sys.exit(1)

print("Loading...")
df = pd.read_csv(CSV_FILE, low_memory=False)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ── 1. shape ───────────────────────────────────────────────────
sec("1. SHAPE")
p(f"  Rows    : {len(df):,}")
p(f"  Columns : {len(df.columns)}")

# ── 2. all 5 label columns ─────────────────────────────────────
sec("2. ALL LABEL COLUMNS — unique values & counts")

label_cols = ['label_full', 'label1', 'label2', 'label3', 'label4']
label_cols = [c for c in label_cols if c in df.columns]

for col in label_cols:
    p(f"\n  [ {col} ]  — dtype: {df[col].dtype}")
    vals = df[col].astype(str).str.strip().value_counts()
    for v, cnt in vals.head(30).items():
        p(f"    {v:<50} {cnt:>8,}  {cnt/len(df)*100:>5.1f}%")
    if len(vals) > 30:
        p(f"    ... and {len(vals)-30} more unique values")

# ── 3. identify the string label column ────────────────────────
sec("3. BEST LABEL COLUMN IDENTIFIED")

best_col = None
for col in label_cols:
    sample_vals = df[col].dropna().astype(str).unique()[:5]
    has_text = any(not v.replace('.','').replace('-','').isnumeric() for v in sample_vals)
    if has_text:
        best_col = col
        break

if best_col:
    p(f"  Using '{best_col}' as the main label column")
    df['_label'] = df[best_col].astype(str).str.strip()
    counts = df['_label'].value_counts()
    p(f"  Total classes: {len(counts)}")
    p()
    p(f"  {'Class':<50} {'Count':>8}  {'%':>6}")
    p(f"  {'·'*68}")
    for label, cnt in counts.items():
        p(f"  {label:<50} {cnt:>8,}  {cnt/len(df)*100:>5.1f}%")
    slow = [l for l in counts.index if 'slow' in l.lower()]
    p(f"\n  Slow-rate labels: {slow if slow else 'NONE FOUND — check label values above'}")
else:
    p("  WARNING: No string label column found. All label columns appear numeric.")
    p("  Check section 2 above — label_full or label1 may use numeric codes.")

# ── 4. data quality (safe version) ────────────────────────────
sec("4. DATA QUALITY")

# Only truly numeric columns
numeric_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])]
string_cols  = [c for c in df.columns
                if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]

p(f"  Numeric columns : {len(numeric_cols)}")
p(f"  String  columns : {string_cols}")
p(f"  Missing values  : {df.isnull().sum().sum():,}")
p(f"  Duplicate rows  : {df.duplicated().sum():,}")

zero_var = []
for c in numeric_cols:
    try:
        v = df[c].var(skipna=True)
        if pd.notna(v) and v < 1e-10:
            zero_var.append(c)
    except Exception:
        pass
p(f"  Zero-variance features ({len(zero_var)}): {zero_var[:10]}")

# ── 5. numeric stats for network features only ─────────────────
sec("5. NETWORK FEATURE STATISTICS (numeric summary)")

net_num_cols = [c for c in numeric_cols if c.startswith('network_')]
if net_num_cols:
    stats = df[net_num_cols].describe().T[['mean','std','min','50%','max']]
    stats.columns = ['mean','std','min','median','max']
    p(stats.to_string(float_format=lambda x: f"{x:.4g}"))
else:
    p("  No numeric network_ columns found.")

# ── 6. slow-rate relevant features (using actual col names) ────
sec("6. KEY SLOW-RATE FEATURES vs. LABEL (medians per class)")

# Map using actual column names found in this dataset
slowrate_features = {
    'Time Delta Avg (IAT)'      : 'network_time-delta_avg',
    'Time Delta Max (IAT Max)'  : 'network_time-delta_max',
    'Packet Interval'           : 'network_interval-packets',
    'Packet Count'              : 'network_packets_all_count',
    'Packet Size Avg'           : 'network_packet-size_avg',
    'Payload Length Avg'        : 'network_payload-length_avg',
    'TCP Window Size Avg'       : 'network_window-size_avg',
    'TCP Window Size Min'       : 'network_window-size_min',
    'FIN Flag Count'            : 'network_tcp-flags-fin_count',
    'SYN Flag Count'            : 'network_tcp-flags-syn_count',
    'RST Flag Count'            : 'network_tcp-flags-rst_count',
    'ACK Flag Count'            : 'network_tcp-flags-ack_count',
    'Header Length Avg'         : 'network_header-length_avg',
    'MSS Avg'                   : 'network_mss_avg',
    'Frag Score'                : 'network_fragmentation-score',
}

found_feats = {k: v for k, v in slowrate_features.items() if v in df.columns}
missing_feats = [k for k in slowrate_features if k not in found_feats]

if best_col and found_feats:
    # Get unique classes (limit to manageable number)
    classes = df['_label'].value_counts().head(10).index.tolist()

    # Header row
    header = f"  {'Feature':<28}"
    for cls in classes:
        short = cls.split('/')[-1][:12] if '/' in cls else cls[:12]
        header += f"  {short:>12}"
    p(header)
    p(f"  {'·'*( 28 + 14*len(classes) )}")

    for fname, col in found_feats.items():
        row = f"  {fname:<28}"
        for cls in classes:
            mask = df['_label'] == cls
            med = df[mask][col].median()
            row += f"  {med:>12.4g}"
        p(row)

    if missing_feats:
        p(f"\n  Not found in dataset: {missing_feats}")
else:
    p("  Skipped — label column or features not resolved.")

# ── 7. log (sensor) columns ────────────────────────────────────
sec("7. SENSOR LOG FEATURES (log_ columns)")
log_cols = [c for c in df.columns if c.startswith('log_')]
p(f"  Found {len(log_cols)} sensor log columns: {log_cols}")
if log_cols:
    p()
    p(df[log_cols].describe().T.to_string())

# ── 8. sample rows ─────────────────────────────────────────────
sec("8. SAMPLE ROWS (3 rows, label + first 8 network cols)")
show_cols = label_cols + [c for c in df.columns if c.startswith('network_')][:8]
show_cols = [c for c in show_cols if c in df.columns]
p(df[show_cols].head(3).to_string(index=False))

# ── save ───────────────────────────────────────────────────────
sec("SAVED")
p(f"  {OUT_FILE}")
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\nDone → dataset_check_output.txt")
