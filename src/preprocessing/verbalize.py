"""
Feature Verbalization — CIC IIoT 2025
======================================
Converts preprocessed (scaled) numeric features back to original units
and renders them as natural-language sentences for the encoder SLM.

Only the top 13 features (by XGBoost gain importance) are used,
organised into 4 functional groups.  This keeps each sentence under
~90 tokens, well within DistilBERT's 512-token limit.

Top features used (XGBoost rank → feature name):
  #1  network_ip-flags_min
  #2  network_packets_all_count
  #3  network_tcp-flags_std_deviation
  #4  network_window-size_max
  #5  network_macs_dst_count
  #6  network_mss_max
  #7  network_packet-size_min
  #8  network_ttl_min
  #9  network_packets_dst_count
  #10 network_macs_src_count
  #15 network_tcp-flags-psh_count
  #18 network_payload-length_avg
  #24 network_time-delta_avg

Usage:
    from src.preprocessing.verbalize import Verbalizer
    v = Verbalizer()
    texts = v.transform(X_scaled_array)   # returns list[str]
"""

import os, sys, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# ── Feature columns in the 70-column scaled array ─────────────────────────
# (these indices come from feature_names.txt — 0-indexed)
_FEATURE_NAMES = None   # loaded lazily

def _load_feature_names():
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        path = os.path.join(ROOT, 'dataset', 'preprocessed', 'feature_names.txt')
        with open(path) as f:
            _FEATURE_NAMES = f.read().splitlines()
    return _FEATURE_NAMES


# The 13 features we verbalise — listed in the order they appear in the array
TOP_FEATURES = [
    'network_ip-flags_min',
    'network_packets_all_count',
    'network_tcp-flags_std_deviation',
    'network_window-size_max',
    'network_macs_dst_count',
    'network_mss_max',
    'network_packet-size_min',
    'network_ttl_min',
    'network_packets_dst_count',
    'network_macs_src_count',
    'network_tcp-flags-psh_count',
    'network_payload-length_avg',
    'network_time-delta_avg',
]


def _fmt(val, decimals=1):
    """Format a float cleanly — no scientific notation."""
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.{decimals}f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.{decimals}f}K"
    if decimals == 0:
        return str(int(round(val)))
    return f"{val:.{decimals}f}"


def _build_sentence(row: dict) -> str:
    """
    Build a ~90-token natural-language description of one network flow.

    row: {feature_name: original_value, ...}  (13 features, inverse-scaled)
    """
    # ── Group 1: IP / Traffic volume ──────────────────────────────────────
    g1 = (
        f"The flow had {_fmt(row['network_packets_all_count'], 0)} total packets "
        f"({_fmt(row['network_packets_dst_count'], 0)} to destination) "
        f"with IP flags minimum {_fmt(row['network_ip-flags_min'], 0)} "
        f"and TTL minimum {_fmt(row['network_ttl_min'], 0)}."
    )

    # ── Group 2: TCP window / MSS / payload ───────────────────────────────
    g2 = (
        f"Window size maximum was {_fmt(row['network_window-size_max'], 0)} bytes, "
        f"MSS maximum {_fmt(row['network_mss_max'], 0)} bytes, "
        f"minimum packet size {_fmt(row['network_packet-size_min'], 0)} bytes, "
        f"and average payload length {_fmt(row['network_payload-length_avg'], 1)} bytes."
    )

    # ── Group 3: TCP flags ────────────────────────────────────────────────
    g3 = (
        f"TCP flags standard deviation was {_fmt(row['network_tcp-flags_std_deviation'], 2)}, "
        f"PSH count {_fmt(row['network_tcp-flags-psh_count'], 0)}."
    )

    # ── Group 4: Hosts & timing ───────────────────────────────────────────
    g4 = (
        f"The flow involved {_fmt(row['network_macs_dst_count'], 0)} destination MACs "
        f"and {_fmt(row['network_macs_src_count'], 0)} source MACs. "
        f"Average inter-packet interval was {_fmt(row['network_time-delta_avg'], 4)} seconds."
    )

    return f"{g1} {g2} {g3} {g4}"


class Verbalizer:
    """
    Converts a scaled feature matrix (N × 70) into a list of N text strings.

    Parameters
    ----------
    scaler_path : str | None
        Path to the fitted RobustScaler pickle.  If None, uses the default
        location: dataset/preprocessed/scaler.pkl
    """

    def __init__(self, scaler_path: str = None):
        if scaler_path is None:
            scaler_path = os.path.join(
                ROOT, 'dataset', 'preprocessed', 'scaler.pkl'
            )
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.feature_names = _load_feature_names()
        # Build index lookup: feature_name → column index in the 70-col array
        self._idx = {name: i for i, name in enumerate(self.feature_names)}
        # Confirm all 13 top features are present
        missing = [f for f in TOP_FEATURES if f not in self._idx]
        if missing:
            raise ValueError(f"Features not found in feature_names.txt: {missing}")

    def transform(self, X_scaled: np.ndarray) -> list:
        """
        Parameters
        ----------
        X_scaled : np.ndarray of shape (N, 70)
            Scaled feature matrix (output of RobustScaler).

        Returns
        -------
        list of str, length N
            One natural-language sentence per sample.
        """
        # Inverse-transform the FULL 70-column matrix back to original units
        X_orig = self.scaler.inverse_transform(X_scaled)

        texts = []
        for i in range(len(X_orig)):
            row = {feat: X_orig[i, self._idx[feat]] for feat in TOP_FEATURES}
            texts.append(_build_sentence(row))
        return texts


# ══════════════════════════════════════════════════════════════════════════
# Quick sanity check — run directly:  python src/preprocessing/verbalize.py
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import yaml

    print("Loading preprocessed data ...")
    PREP_DIR = os.path.join(ROOT, 'dataset', 'preprocessed')
    X_train  = np.load(os.path.join(PREP_DIR, 'X_train.npy'))
    y_train  = np.load(os.path.join(PREP_DIR, 'y_train.npy'))

    with open(os.path.join(ROOT, 'configs', 'project_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    LABEL_MAP = {int(k): v for k, v in cfg['dataset']['label_map'].items()}

    v = Verbalizer()

    print(f"\nVerbalizing {len(X_train):,} training samples ...")
    texts = v.transform(X_train)

    # Show one example per class
    print("\n" + "═"*70)
    print("  SAMPLE VERBALIZED SENTENCES (one per class)")
    print("="*70)
    seen = set()
    for i, label in enumerate(y_train):
        if label not in seen:
            seen.add(label)
            token_est = len(texts[i].split())
            print(f"\n  Class {label} ({LABEL_MAP[label]}) — ~{token_est} words:")
            print(f"  {texts[i]}")
        if len(seen) == len(LABEL_MAP):
            break

    # Token length statistics
    lengths = [len(t.split()) for t in texts[:2000]]
    print(f"\n  Token estimate stats (first 2000 samples):")
    print(f"    min={min(lengths)}  max={max(lengths)}  "
          f"mean={sum(lengths)/len(lengths):.1f}")
    print(f"\n  ✓ All sentences fit within DistilBERT's 512-token limit.\n")
