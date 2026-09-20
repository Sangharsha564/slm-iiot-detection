import re, glob
import pandas as pd
from nfstream import NFStreamer

NF = dict(statistical_analysis=True, idle_timeout=120, active_timeout=1800)

# ---------- ATTACK ----------
files = sorted(glob.glob('dos_slowloris/*.pcap') + glob.glob('ddos_slowloris/*.pcap'))
print(f"Found {len(files)} attack files\n")

parts = []
for f in files:
    name = f.split('/')[-1]
    d = NFStreamer(source=f, **NF).to_pandas()
    d['source_file']  = name
    d['target_port']  = int(re.search(r'port-(\d+)', name).group(1))
    d['variant']      = 'ddos' if '_ddos_' in name else 'dos'
    parts.append(d)
    print(f"  {name:52} {len(d):6} flows")

atk = pd.concat(parts, ignore_index=True)
atk.to_csv('raw_attack.csv', index=False)
print(f"\nraw_attack.csv  -> {len(atk)} flows")

# ---------- BENIGN ----------
print("\nExtracting benign (this one takes longest)...")
ben = NFStreamer(source='benign/benign_whole-network3.pcap', **NF).to_pandas()
ben['source_file'] = 'benign_whole-network3.pcap'
ben['target_port'] = -1
ben['variant']     = 'benign'
ben.to_csv('raw_benign.csv', index=False)
print(f"raw_benign.csv  -> {len(ben)} flows")
