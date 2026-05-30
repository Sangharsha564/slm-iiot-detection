import pandas as pd

attack = pd.read_csv('attack_samples_10sec.csv', low_memory=False)
benign = pd.read_csv('benign_samples_10sec.csv', low_memory=False)

# Add binary label
attack['binary_label'] = 'attack'
benign['binary_label'] = 'benign'

# Merge
df = pd.concat([attack, benign], ignore_index=True)

# Compute flow duration from timestamps
df['flow_duration'] = pd.to_datetime(df['timestamp_end']) - pd.to_datetime(df['timestamp_start'])
df['flow_duration_sec'] = df['flow_duration'].dt.total_seconds()

# Count Slowloris
slowloris = df[df['label3'].str.contains('slowloris', case=False, na=False)]
print(f"Total rows       : {len(df):,}")
print(f"Benign rows      : {len(benign):,}")
print(f"Attack rows      : {len(attack):,}")
print(f"Slowloris rows   : {len(slowloris):,}")
print(f"\nSlowloris by label3:")
print(slowloris['label3'].value_counts())
print(f"\nSlowloris flow_duration_sec (median): {slowloris['flow_duration_sec'].median():.1f}s")
print(f"Benign flow_duration_sec (median)   : {df[df['binary_label']=='benign']['flow_duration_sec'].median():.1f}s")

df.to_csv('combined_10sec.csv', index=False)
print("\nSaved: combined_10sec.csv")
