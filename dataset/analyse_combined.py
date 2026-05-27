"""
CIC IIoT 2025 — Combined Dataset Analysis
Loads combined_10sec.csv and prints a focused research summary.
Run: python3 analyse_combined.py
"""

import os, sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, 'combined_10sec.csv')
OUT_FILE = os.path.join(BASE_DIR, 'combined_analysis_output.txt')

lines = []
def p(msg=''):   print(msg);  lines.append(str(msg))
def sec(t):      p();  p(f"{'─'*65}");  p(f"  {t}");  p(f"{'─'*65}")

if not os.path.exists(CSV_FILE):
    print(f"ERROR: {CSV_FILE} not found. Run combine.py first."); sys.exit(1)

print("Loading combined_10sec.csv ...")
df = pd.read_csv(CSV_FILE, low_memory=False)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print("Done.\n")

# ── helper masks ───────────────────────────────────────────────
benign_mask    = df['label2'] == 'benign'
slowloris_mask = df['label3'].str.contains('slowloris', case=False, na=False)
dos_mask       = (df['label2'] == 'dos')  & ~slowloris_mask
ddos_mask      = (df['label2'] == 'ddos') & ~slowloris_mask
recon_mask     = df['label2'] == 'recon'
other_mask     = ~benign_mask & ~slowloris_mask & ~dos_mask & ~ddos_mask & ~recon_mask

groups = {
    'BENIGN'      : benign_mask,
    'Slowloris'   : slowloris_mask,
    'DoS (other)' : dos_mask,
    'DDoS (other)': ddos_mask,
    'Recon'       : recon_mask,
    'Other Attack': other_mask,
}

# ── 1. overall class distribution ─────────────────────────────
sec("1. OVERALL CLASS DISTRIBUTION")
total = len(df)
p(f"  {'Group':<20} {'Rows':>7}  {'%':>6}  {'Bar'}")
p(f"  {'·'*55}")
for name, mask in groups.items():
    n   = mask.sum()
    pct = n / total * 100
    bar = '█' * max(1, int(pct / 2))
    p(f"  {name:<20} {n:>7,}  {pct:>5.1f}%  {bar}")
p(f"  {'·'*55}")
p(f"  {'TOTAL':<20} {total:>7,}  100.0%")

# ── 2. slowloris breakdown ─────────────────────────────────────
sec("2. SLOWLORIS BREAKDOWN")
sl_df = df[slowloris_mask]
p(f"  Total Slowloris rows : {len(sl_df)}")
p()
p(f"  {'label3':<35} {'label2':>8}  {'Count':>6}")
p(f"  {'·'*53}")
for (l3, l2), cnt in sl_df.groupby(['label3','label2']).size().items():
    p(f"  {l3:<35} {l2:>8}  {cnt:>6}")

# ── 3. key feature comparison ──────────────────────────────────
sec("3. KEY FEATURE COMPARISON — MEDIANS ACROSS GROUPS")

features = {
    'time-delta_avg (IAT Mean)'     : 'network_time-delta_avg',
    'time-delta_max (IAT Max)'      : 'network_time-delta_max',
    'interval-packets'              : 'network_interval-packets',
    'packets_all_count'             : 'network_packets_all_count',
    'payload-length_avg'            : 'network_payload-length_avg',
    'packet-size_avg'               : 'network_packet-size_avg',
    'tcp-flags-syn_count'           : 'network_tcp-flags-syn_count',
    'tcp-flags-fin_count'           : 'network_tcp-flags-fin_count',
    'tcp-flags-rst_count'           : 'network_tcp-flags-rst_count',
    'tcp-flags-ack_count'           : 'network_tcp-flags-ack_count',
    'window-size_avg'               : 'network_window-size_avg',
    'window-size_min'               : 'network_window-size_min',
    'mss_avg'                       : 'network_mss_avg',
    'fragmentation-score'           : 'network_fragmentation-score',
    'ip-length_avg'                 : 'network_ip-length_avg',
}
features = {k: v for k, v in features.items() if v in df.columns}

# header
hdr = f"  {'Feature':<32}"
for name in groups: hdr += f"  {name[:12]:>12}"
p(hdr)
p(f"  {'·'*(32 + 14*len(groups))}")

for fname, col in features.items():
    row = f"  {fname:<32}"
    meds = {}
    for name, mask in groups.items():
        m = df[mask][col].median()
        meds[name] = m
        row += f"  {m:>12.4g}"

    # flag Slowloris signal
    sl_val = meds.get('Slowloris', np.nan)
    bn_val = meds.get('BENIGN', np.nan)
    if not np.isnan(sl_val) and not np.isnan(bn_val) and abs(bn_val) > 1e-9:
        ratio = sl_val / abs(bn_val)
        if ratio > 3:   row += '  ← SL HIGH'
        elif ratio < 0.3: row += '  ← SL LOW'
    p(row)

# ── 4. slowloris vs benign detailed stats ─────────────────────
sec("4. SLOWLORIS vs BENIGN — DETAILED STATS (mean ± std)")

p(f"  {'Feature':<32}  {'BENIGN mean±std':<28}  {'Slowloris mean±std':<28}  {'Ratio'}")
p(f"  {'·'*100}")

for fname, col in features.items():
    bn = df[benign_mask][col]
    sl = df[slowloris_mask][col]
    bn_str = f"{bn.mean():.4g} ± {bn.std():.4g}"
    sl_str = f"{sl.mean():.4g} ± {sl.std():.4g}"
    ratio  = sl.mean() / (bn.mean() + 1e-9) if bn.mean() != 0 else float('nan')
    p(f"  {fname:<32}  {bn_str:<28}  {sl_str:<28}  {ratio:.2f}x")

# ── 5. class imbalance summary ────────────────────────────────
sec("5. CLASS IMBALANCE SUMMARY")

p(f"  Binary (benign vs attack):")
p(f"    Benign : {benign_mask.sum():,}  ({benign_mask.sum()/total*100:.1f}%)")
p(f"    Attack : {(~benign_mask).sum():,}  ({(~benign_mask).sum()/total*100:.1f}%)")

p(f"\n  Slowloris vs everything else (binary slow-rate task):")
p(f"    Slowloris : {slowloris_mask.sum():,}  ({slowloris_mask.sum()/total*100:.2f}%)")
p(f"    Non-SL    : {(~slowloris_mask).sum():,}  ({(~slowloris_mask).sum()/total*100:.2f}%)")
p(f"    Imbalance ratio : {(~slowloris_mask).sum() / slowloris_mask.sum():.0f}:1")

p(f"\n  label2 distribution (recommended training label):")
for lbl, cnt in df['label2'].value_counts().items():
    p(f"    {lbl:<25} {cnt:>6,}  ({cnt/total*100:.1f}%)")

# ── 6. feature correlation with slowloris label ───────────────
sec("6. TOP FEATURES CORRELATED WITH SLOWLORIS (binary)")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c.startswith('network_') or c.startswith('log_')]

y_sl = slowloris_mask.astype(float)
corr = {}
for col in num_cols:
    try:
        c = df[col].fillna(0).corr(y_sl)
        if not np.isnan(c):
            corr[col] = abs(c)
    except Exception:
        pass

corr_df = pd.Series(corr).sort_values(ascending=False)
p(f"  {'Rank':<5} {'Feature':<45} {'|Correlation|':>14}")
p(f"  {'·'*67}")
for i, (col, val) in enumerate(corr_df.head(20).items(), 1):
    direction = '+' if df[col].fillna(0).corr(y_sl) > 0 else '-'
    p(f"  {i:<5} {col:<45} {direction}{val:>13.4f}")

# ── 7. sensor log comparison ──────────────────────────────────
sec("7. SENSOR LOG FEATURES — SLOWLORIS vs BENIGN (medians)")

log_cols = [c for c in df.columns if c.startswith('log_') and
            pd.api.types.is_numeric_dtype(df[c])]
if log_cols:
    p(f"  {'Feature':<38}  {'BENIGN':>10}  {'Slowloris':>10}  {'Signal'}")
    p(f"  {'·'*65}")
    for col in log_cols:
        bn_med = df[benign_mask][col].median()
        sl_med = df[slowloris_mask][col].median()
        sig = '?'
        if bn_med > 0:
            r = sl_med / bn_med
            sig = '▲' if r > 2 else ('▼' if r < 0.5 else '≈')
        p(f"  {col:<38}  {bn_med:>10.4g}  {sl_med:>10.4g}  {sig}")

# ── 8. recommended label encoding ────────────────────────────
sec("8. RECOMMENDED LABEL ENCODING FOR YOUR MODEL")

p("  For XGBoost gate (binary):")
p("    0 = benign,  1 = attack  →  use 'binary_label' column")

p("\n  For Encoder SLM (multiclass, 5 classes):")
encoding = {
    'benign'             : 0,
    'slowloris (dos+ddos)': 1,
    'dos_other'          : 2,
    'ddos_other'         : 3,
    'recon'              : 4,
    'other (mitm/web/bruteforce/malware)': 5,
}
for lbl, code in encoding.items():
    p(f"    {code} = {lbl}")

p("\n  Python encoding snippet:")
p("""
    def encode_label(row):
        l2 = str(row['label2']).lower()
        l3 = str(row['label3']).lower()
        if l2 == 'benign':          return 0
        if 'slowloris' in l3:       return 1
        if l2 == 'dos':             return 2
        if l2 == 'ddos':            return 3
        if l2 == 'recon':           return 4
        return 5  # mitm, web, bruteforce, malware

    df['model_label'] = df.apply(encode_label, axis=1)
    print(df['model_label'].value_counts().sort_index())
""")

# ── 9. critical warnings ──────────────────────────────────────
sec("9. CRITICAL WARNINGS & RECOMMENDATIONS")

p("  [1] Only 132 Slowloris samples (0.44% of dataset)")
p("      → Must download other window-size files (5sec, 10sec from other scenarios)")
p("      → Or apply SMOTE / class-weighted loss during training")
p("      → Minimum recommended: 500 samples per class for SLM fine-tuning")

p("\n  [2] flow_duration from timestamps = always 10s (useless feature)")
p("      → Drop timestamp columns from ML input")
p("      → Use network_time-delta_avg and network_interval-packets instead")

p("\n  [3] Three zero-variance features in benign data:")
p("      network_fragmentation-score, network_fragmented-packets, network_tcp-flags-urg_count")
p("      → Keep fragmentation features (attack signal), drop urg_count")

p("\n  [4] String columns to drop before training:")
p("      device_name, device_mac, timestamp, timestamp_start, timestamp_end")
p("      network_ips_all/dst/src, network_macs_all/dst/src")
p("      network_ports_all/dst/src, network_protocols_all/dst/src, log_data-types")

# ── save ──────────────────────────────────────────────────────
sec("SAVED")
p(f"  {OUT_FILE}")
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\nDone → combined_analysis_output.txt")
