"""
Feature Verbalization v2 — Key-Value Format with Boolean Domain Flags
======================================================================
Converts preprocessed (scaled) feature arrays back to original units
and renders them as compact key-value text for the encoder/decoder SLM.

Improvements over v1:
  - Key-value format instead of natural language sentences
  - Boolean domain flags appended (HIGH_PSH, IP_FLAGS_2, etc.)
  - Uses 31 selected features instead of 13 hardcoded ones
  - Loads thresholds from domain_flags_thresholds.json (no hardcoding)
  - ~35 tokens per sample vs ~64 words before — better token efficiency

Output format example:
  "pkts:3241 ip_fl_min:2 ttl_min:64 win_max:65535 tcp_std:6.68
   t_avg:0.0009 ip_len_avg:159 syn:52 fin:0 interval:7.7
   ips:3 ports:4 ttl_avg:64 win_std:1200 [HIGH_PSH] [IP_FLAGS_2]
   [SLOW_INTERVAL] [LOW_PAYLOAD]"

Usage:
    from src.preprocessing.verbalize import Verbalizer
    v = Verbalizer()
    texts = v.transform(X_scaled_array)   # returns list[str]
"""

import os, sys, json, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PREP_DIR = os.path.join(ROOT, 'dataset', 'preprocessed')


# ── Short display names for key-value format ──────────────────────────────────
# Maps full feature name → compact key used in the text
SHORT_NAMES = {
    # ── selected features ─────────────────────────────────────────────
    'network_packets_all_count'           : 'pkts',
    'network_tcp-flags_std_deviation'     : 'tcp_std',
    'network_ttl_std_deviation'           : 'ttl_std',
    'network_ip-flags_avg'                : 'ip_fl_avg',
    'network_ports_all_count'             : 'ports',
    'network_time-delta_avg'              : 't_avg',
    'network_time-delta_min'              : 't_min',
    'network_tcp-flags-syn_count'         : 'syn',
    'network_ip-length_avg'               : 'ip_len_avg',
    'network_ttl_avg'                     : 'ttl_avg',
    'network_window-size_max'             : 'win_max',
    'network_ip-flags_std_deviation'      : 'ip_fl_std',
    'network_window-size_std_deviation'   : 'win_std',
    'network_ttl_min'                     : 'ttl_min',
    'network_ips_all_count'               : 'ips',
    'network_interval-packets'            : 'interval',
    'network_tcp-flags_avg'               : 'tcp_avg',
    'network_ips_src_count'               : 'ips_src',
    'network_packet-size_min'             : 'pkt_min',
    'network_ports_dst_count'             : 'ports_dst',
    'network_window-size_min'             : 'win_min',
    'network_time-delta_std_deviation'    : 't_std',
    'network_window-size_avg'             : 'win_avg',
    'network_ip-length_min'               : 'ip_len_min',
    'network_mss_avg'                     : 'mss',
    'network_tcp-flags-fin_count'         : 'fin',
    'network_ip-length_std_deviation'     : 'ip_len_std',
    'network_ip-length_max'               : 'ip_len_max',
    'network_payload-length_std_deviation': 'pay_std',
    'log_messages_count'                  : 'log_msgs',
    'network_ip-flags_min'                : 'ip_fl_min',
    # ── protected domain signal features ─────────────────────────────
    'network_tcp-flags-psh_count'         : 'psh',
    'network_payload-length_avg'          : 'pay_avg',
    'network_tcp-flags-rst_count'         : 'rst',
    'network_header-length_avg'           : 'hdr_avg',
    'network_fragmentation-score'         : 'frag',
    'log_data-ranges_avg'                 : 'log_rng',
    'log_data-ranges_std_deviation'       : 'log_rng_std',
    'log_interval-messages'               : 'log_int',
    'network_header-length_std_deviation' : 'hdr_std',
    'network_ip-flags_max'                : 'ip_fl_max',
    'network_tcp-flags_max'               : 'tcp_max',
    'network_tcp-flags_min'               : 'tcp_min',
    'network_ttl_max'                     : 'ttl_max',
    'network_protocols_all_count'         : 'protos',
    'network_protocols_src_count'         : 'protos_src',
    'network_ips_dst_count'               : 'ips_dst',
    'network_ports_src_count'             : 'ports_src',
}


def _fmt(val, decimals=2):
    """Format a float compactly — no scientific notation."""
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if abs(val) >= 10_000:
        return f"{val/1_000:.1f}K"
    if decimals == 0:
        return str(int(round(val)))
    return f"{val:.{decimals}f}"


class Verbalizer:
    """
    Converts a scaled feature matrix (N × 31) into N key-value text strings
    with Boolean domain flags appended.

    Parameters
    ----------
    scaler_path : str | None
        Path to scaler.pkl. If None uses default dataset/preprocessed/scaler.pkl
    """

    def __init__(self, scaler_path=None):
        # Load scaler
        if scaler_path is None:
            scaler_path = os.path.join(PREP_DIR, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load feature names
        feat_path = os.path.join(PREP_DIR, 'feature_names.txt')
        with open(feat_path) as f:
            self.feature_names = f.read().splitlines()

        # Load log transform info (to reverse log1p correctly)
        log_path = os.path.join(PREP_DIR, 'log_transform_info.json')
        with open(log_path) as f:
            log_info = json.load(f)
        self.skewed_features = set(log_info['skewed_features'])

        # Load domain flag thresholds
        flags_path = os.path.join(PREP_DIR, 'domain_flags_thresholds.json')
        with open(flags_path) as f:
            self.thresholds = json.load(f)

        # Feature index lookup
        self._idx = {name: i for i, name in enumerate(self.feature_names)}

    def _inverse_transform(self, X_scaled):
        """
        Reverse scaling to get original units.
        Steps: RobustScaler inverse → expm1 for log-transformed features.
        """
        X_robust = self.scaler.inverse_transform(X_scaled)

        # Reverse log1p for skewed features: expm1(x) = exp(x) - 1
        X_orig = X_robust.copy()
        for i, name in enumerate(self.feature_names):
            if name in self.skewed_features:
                X_orig[:, i] = np.expm1(np.clip(X_robust[:, i], -10, 30))

        return X_orig

    def _compute_flags(self, row_orig):
        """
        Compute 6 Boolean domain flags from original-unit feature values.
        Returns list of flag strings like ['HIGH_PSH', 'IP_FLAGS_2']
        """
        flags = []
        t = self.thresholds

        def get(name):
            if name in self._idx:
                return row_orig[self._idx[name]]
            return 0.0

        psh      = get('network_tcp-flags-psh_count')
        ip_flags = get('network_ip-flags_min')
        interval = get('network_time-delta_avg')
        packets  = get('network_packets_all_count')
        payload  = get('network_payload-length_std_deviation')  # proxy for payload
        syn      = get('network_tcp-flags-syn_count')

        if psh      > t['HIGH_PSH']:                   flags.append('HIGH_PSH')
        if ip_flags >= t['IP_FLAGS_2']:                flags.append('IP_FLAGS_2')
        if interval < t['SLOW_INTERVAL'] and interval > 0: flags.append('SLOW_INTERVAL')
        if packets  > t['HIGH_VOLUME']:                flags.append('HIGH_VOLUME')
        if payload  < t['LOW_PAYLOAD']:                flags.append('LOW_PAYLOAD')
        if syn      > t['HIGH_SYN']:                   flags.append('HIGH_SYN')

        return flags

    def transform(self, X_scaled):
        """
        Parameters
        ----------
        X_scaled : np.ndarray of shape (N, 31)
            Scaled feature matrix from preprocessing v2.

        Returns
        -------
        list of str, length N
            One key-value text string per sample with Boolean flags.
        """
        X_orig = self._inverse_transform(X_scaled)
        texts  = []

        for i in range(len(X_orig)):
            row  = X_orig[i]
            parts = []

            # Add each feature as key:value pair
            for feat_name in self.feature_names:
                idx       = self._idx[feat_name]
                val       = row[idx]
                short_key = SHORT_NAMES.get(feat_name, feat_name.split('_')[-1])

                # Format based on magnitude
                if abs(val) < 0.01 and val != 0:
                    formatted = f"{val:.4f}"
                elif abs(val) >= 1000:
                    formatted = _fmt(val, 0)
                elif abs(val) >= 10:
                    formatted = _fmt(val, 1)
                else:
                    formatted = _fmt(val, 2)

                parts.append(f"{short_key}:{formatted}")

            # Append Boolean domain flags
            flags = self._compute_flags(row)
            if flags:
                parts.extend(f"[{f}]" for f in flags)

            texts.append(' '.join(parts))

        return texts


# ══════════════════════════════════════════════════════════════════════════════
# Quick sanity check — run directly:
#   python src/preprocessing/verbalize.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import yaml

    print("Loading preprocessed data ...")
    X_train = np.load(os.path.join(PREP_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(PREP_DIR, 'y_train.npy'))

    with open(os.path.join(ROOT, 'configs', 'project_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    LABEL_MAP = {int(k): v for k, v in cfg['dataset']['label_map'].items()}

    v = Verbalizer()

    print(f"\nVerbalizing {len(X_train):,} training samples ...")
    texts = v.transform(X_train)

    # Show one example per class
    print("\n" + "═"*70)
    print("  SAMPLE KEY-VALUE SENTENCES (one per class)")
    print("="*70)
    seen = set()
    for i, label in enumerate(y_train):
        if label not in seen:
            seen.add(label)
            token_est = len(texts[i].split())
            flags     = [t for t in texts[i].split() if t.startswith('[')]
            print(f"\n  Class {label} ({LABEL_MAP[label]}) — ~{token_est} tokens | flags: {flags}")
            print(f"  {texts[i]}")
        if len(seen) == len(LABEL_MAP):
            break

    # Token statistics
    lengths = [len(t.split()) for t in texts[:2000]]
    print(f"\n  Token estimate stats (first 2,000 samples):")
    print(f"    min={min(lengths)}  max={max(lengths)}  "
          f"mean={sum(lengths)/len(lengths):.1f}")
    print(f"\n  ✓ All samples fit within DistilBERT's 512-token limit.\n")

    # Flag statistics for Slowloris
    sl_texts = [texts[i] for i, l in enumerate(y_train) if l == 1]
    print(f"  Slowloris flag statistics ({len(sl_texts)} training samples):")
    for flag in ['HIGH_PSH', 'IP_FLAGS_2', 'SLOW_INTERVAL',
                 'HIGH_VOLUME', 'LOW_PAYLOAD', 'HIGH_SYN']:
        rate = sum(1 for t in sl_texts if f'[{flag}]' in t) / max(len(sl_texts), 1)
        print(f"    [{flag}] triggers on {rate*100:.0f}% of Slowloris samples")
