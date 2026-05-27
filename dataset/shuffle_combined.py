"""
Shuffle combined_10sec.csv and save as combined_shuffled.csv
Run: python3 shuffle_combined.py
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_FILE  = os.path.join(BASE_DIR, 'combined_10sec.csv')
OUT_FILE = os.path.join(BASE_DIR, 'combined_shuffled.csv')

print("Loading...")
df = pd.read_csv(IN_FILE, low_memory=False)
print(f"  Rows before: {len(df):,}  |  Columns: {len(df.columns)}")

# Check order before shuffle
print(f"\n  First 3 labels before shuffle: {df['label2'].head(3).tolist()}")
print(f"  Last  3 labels before shuffle: {df['label2'].tail(3).tolist()}")

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Verify after shuffle
print(f"\n  First 3 labels after shuffle : {df['label2'].head(3).tolist()}")
print(f"  Last  3 labels after shuffle : {df['label2'].tail(3).tolist()}")

# Confirm class counts unchanged
print(f"\n  Class distribution (unchanged):")
for lbl, cnt in df['label2'].value_counts().items():
    print(f"    {lbl:<25} {cnt:>6,}")

print(f"\nSaving to {OUT_FILE} ...")
df.to_csv(OUT_FILE, index=False)
print(f"Done. Rows: {len(df):,}")
