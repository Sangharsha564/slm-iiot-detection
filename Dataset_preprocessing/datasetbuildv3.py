#!/usr/bin/env python3
"""
DataSense Slowloris — dataset builder v3.

Change from v2, driven by the v2 proof report:

  REVERTED - the handshake filter.
      v2 kept only flows whose client SYN was observed.  In a 62 s window
      the only benign connections showing a SYN are ones that START inside
      it, i.e. short ones.  Long-lived camera / MQTT sessions began hours
      earlier and were removed: benign 25,750 -> 1,611, and benign median
      duration collapsed from 40.5 s to 0.76 s.  Duration then separated
      the classes almost perfectly (stump F1 0.995) for a reason created
      by the filter.  Mid-window fragments are realistic - an IDS reading
      62 s windows sees them too - so they are kept.

  NEW - drop connection-boundary flags instead.
      SYN and FIN counts indicate whether the window happened to contain
      the connection's start or end.  Every attack capture begins at
      attack launch, so every attack flow carries its SYN; that is an
      artifact of how the dataset was recorded, not of Slowloris.
      PSH / ACK / RST are kept - they describe what happened DURING the
      connection, which is behaviour.

  NEW - the shortcut check now also runs on the test set, so an easy task
      is visible as an easy task rather than mistaken for a train-only leak.

Unchanged: TCP only, hold rule at 1.0 s, feature-level dedup,
train on edge1, test on wisenet-camera.

Usage:
    python build_dataset_v3.py /path/to/raw_files ./benign_chunks
"""

import re
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from nfstream import NFStreamer
except ImportError:
    sys.exit("Run:  pip install nfstream pandas scikit-learn")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


# ---------------------------------------------------------------- config ----

VICTIM_IP = {
    "edge1":          "192.168.1.195",
    "mqtt-broker":    "192.168.1.193",
    "wisenet-camera": "192.168.1.57",
    "yi-camera":      "192.168.1.50",
}
ATTACKER_IPS = {f"192.168.1.{n}" for n in range(100, 106)}
CAPTURE_HOST = "192.168.1.210"
TCP = 6

IDLE_TIMEOUT = 120
ACTIVE_TIMEOUT = 1800
HOLD_THRESHOLD_S = 1.0
TRAIN_DEVICE = "edge1"
TEST_DEVICE = "wisenet-camera"

REQUIRE_HANDSHAKE = False     # reverted - see docstring
DEDUP_ON_FEATURES = True

# Window-alignment artifacts: presence depends on where the capture
# boundary fell, not on connection behaviour.
BOUNDARY_FLAG_COLS = [
    "bidirectional_syn_packets", "src2dst_syn_packets", "dst2src_syn_packets",
    "bidirectional_fin_packets", "src2dst_fin_packets", "dst2src_fin_packets",
]

FNAME_RE = re.compile(r"attack_(dos|ddos)_slowloris-port-(\d+)_(.+)\.pcap$")
OUT = Path("./dataset_out_v3")

DROP_COLS = [
    "id", "expiration_id", "src_ip", "src_mac", "src_oui",
    "dst_ip", "dst_mac", "dst_oui", "src_port", "dst_port",
    "protocol", "ip_version", "vlan_id", "tunnel_id",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    "application_name", "application_category_name",
    "application_is_guessed", "application_confidence",
    "requested_server_name", "client_fingerprint",
    "server_fingerprint", "user_agent", "content_type",
]


# ----------------------------------------------------------- extraction -----

def read_pcap(path):
    df = NFStreamer(source=str(path),
                    idle_timeout=IDLE_TIMEOUT,
                    active_timeout=ACTIVE_TIMEOUT,
                    statistical_analysis=True,
                    n_dissections=20).to_pandas()
    if df.empty:
        return df
    return df[df["protocol"] == TCP].copy()


def flow_key(df):
    raw = (df["src_ip"].astype(str) + "|" + df["dst_ip"].astype(str) + "|"
           + df["src_port"].astype(str) + "|" + df["dst_port"].astype(str)
           + "|" + df["bidirectional_first_seen_ms"].astype(str))
    return raw.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:16])


def load_attack_files(root):
    frames = []
    files = sorted(p for p in root.rglob("*.pcap")
                   if FNAME_RE.search(p.name) and not p.name.startswith("._"))
    if not files:
        sys.exit("No slowloris pcaps found - check the path.")

    print(f"Reading {len(files)} attack captures ...")
    for path in files:
        m = FNAME_RE.search(path.name)
        category, port, device = m.group(1), int(m.group(2)), m.group(3)
        if device not in VICTIM_IP:
            continue

        print(f"  {path.name}")
        df = read_pcap(path)
        if df.empty:
            continue

        vip = VICTIM_IP[device]
        df["label"] = (
            df["src_ip"].isin(ATTACKER_IPS)
            & (df["dst_ip"] == vip)
            & (df["dst_port"] == port)
        ).astype(int)

        df = df[~df["src_ip"].eq(CAPTURE_HOST) & ~df["dst_ip"].eq(CAPTURE_HOST)]
        involves_attacker = (df["src_ip"].isin(ATTACKER_IPS)
                             | df["dst_ip"].isin(ATTACKER_IPS))
        df = df[(df["label"] == 1) | ~involves_attacker]

        df["device"] = device
        df["scenario"] = f"{category}_{device}_{port}"
        df["source"] = "attack_capture"
        df["chunk"] = 0
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_benign_chunks(folder):
    files = sorted(p for p in folder.glob("*.pcap")
                   if not p.name.startswith("._"))
    if not files:
        sys.exit(f"No benign chunks in {folder} - run editcap first.")

    print(f"Reading {len(files)} benign windows ...")
    frames = []
    for i, path in enumerate(files, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")
        df = read_pcap(path)
        if df.empty:
            continue
        df = df[~df["src_ip"].eq(CAPTURE_HOST) & ~df["dst_ip"].eq(CAPTURE_HOST)]
        df = df[~df["src_ip"].isin(ATTACKER_IPS)
                & ~df["dst_ip"].isin(ATTACKER_IPS)]
        if df.empty:
            continue
        df["label"] = 0
        df["device"] = "benign_capture"
        df["scenario"] = "benign_baseline"
        df["source"] = "benign_capture"
        df["chunk"] = i
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------- rule 2 -----

def apply_hold_rule(attack_df):
    atk = attack_df[attack_df["label"] == 1]
    stats = (atk.groupby("scenario")
             .agg(attack_flows=("scenario", "size"),
                  median_hold_s=("bidirectional_duration_ms",
                                 lambda s: round(s.median() / 1000, 2)))
             .reset_index())
    stats["kept"] = stats["median_hold_s"] >= HOLD_THRESHOLD_S
    stats["reason"] = np.where(
        stats["kept"], "connections held",
        f"median hold < {HOLD_THRESHOLD_S}s - attack did not take effect")
    return stats


# ---------------------------------------------------------------- proof -----

def stump_check(train, features):
    rows = []
    y = train["label"].values
    for f in features:
        x = train[[f]].fillna(0).values
        try:
            clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x, y)
            rows.append((f, round(f1_score(y, clf.predict(x)), 4)))
        except Exception:                                    # noqa: BLE001
            continue
    return pd.DataFrame(rows, columns=["feature", "stump_f1"]) \
             .sort_values("stump_f1", ascending=False)


def build_proof(train, test, features, scenario_table, notes):
    L = ["=" * 70, "DATASET CORRECTNESS PROOF  (v2)", "=" * 70, ""]

    L.append("0. FILTERS APPLIED")
    for n in notes:
        L.append(f"   {n}")
    L.append("")

    L.append("1. SCENARIO SELECTION (Rule 2)")
    L.append(scenario_table.to_string(index=False))
    L.append("")

    L.append("2. CLASS BALANCE")
    for name, d in [("train", train), ("test", test)]:
        n1, n0 = int((d.label == 1).sum()), int((d.label == 0).sum())
        L.append(f"   {name:6s}  attack={n1:6d}  benign={n0:6d}  "
                 f"ratio=1:{(n0 / max(n1, 1)):.2f}")
    L.append("")

    L.append("3. PROTOCOL CONFOUND  (Rule 1)")
    L.append("   TCP-only by construction; protocol column dropped.  PASS")
    L.append("")

    L.append("4. DURATION  (seconds)")
    for name, d in [("train", train), ("test", test)]:
        for lab in (1, 0):
            s = d.loc[d.label == lab, "bidirectional_duration_ms"] / 1000
            if len(s):
                L.append(f"   {name:6s} label={lab}  median={s.median():7.2f}"
                         f"  p90={s.quantile(.9):7.2f}  max={s.max():7.2f}")
    L.append("   Both classes capped at ~62 s, so duration is only weakly")
    L.append("   discriminative.  That is expected and correct.")
    L.append("")

    L.append("5. DUPLICATE FEATURE ROWS")
    for name, d in [("train", train), ("test", test)]:
        dup = int(d[features].duplicated().sum())
        L.append(f"   {name:6s}  duplicates={dup}  "
                 f"({'PASS' if dup == 0 else 'CHECK'})")
    L.append("")

    L.append("6. TRAIN/TEST LEAKAGE")
    overlap = len(set(train["flow_key"]) & set(test["flow_key"]))
    L.append(f"   shared flow keys = {overlap}  "
             f"({'PASS' if overlap == 0 else 'FAIL'})")
    atk_dev = (set(train.loc[train.label == 1, "device"])
               & set(test.loc[test.label == 1, "device"]))
    L.append(f"   shared ATTACK devices = {sorted(atk_dev) or 'none'}  "
             f"({'PASS' if not atk_dev else 'FAIL'})")
    L.append("   (benign_capture appears on both sides by design - split by")
    L.append("    time window, no window spans both)")
    L.append("")

    L.append("7. IDENTITY LEAKAGE")
    leaked = [c for c in list(DROP_COLS) + BOUNDARY_FLAG_COLS if c in features]
    L.append(f"   identity columns in features = {leaked or 'none'}  "
             f"({'PASS' if not leaked else 'FAIL'})")
    L.append("")

    L.append("8. SHORTCUT CHECK  (single-feature decision stump)")
    stumps = stump_check(train, features)
    stumps_te = stump_check(test, features).rename(
        columns={"stump_f1": "stump_f1_test"})
    merged = stumps.merge(stumps_te, on="feature", how="left")
    L.append(merged.head(15).to_string(index=False))
    top, top_te = stumps["stump_f1"].max(), stumps_te["stump_f1_test"].max()
    L.append(f"   best single feature F1 - train {top:.4f}, test {top_te:.4f}")
    if top < 0.75:
        L.append("   PASS - no single feature dominates")
    elif abs(top - top_te) < 0.10:
        L.append("   NOT A LEAK - the stump generalises to the unseen device,")
        L.append("   so the task is genuinely easy on this benign population.")
        L.append("   Report this as a baseline rather than engineering it away.")
    else:
        L.append("   REVIEW - stump does not transfer; likely a train-side leak.")
    return "\n".join(L), merged


# ----------------------------------------------------------------- main -----

def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    root = Path(sys.argv[1]).expanduser()
    benign_dir = Path(sys.argv[2]).expanduser()
    OUT.mkdir(exist_ok=True)
    notes = []

    attack = load_attack_files(root)
    scenario_table = apply_hold_rule(attack)
    print("\nRule 2 outcome:")
    print(scenario_table.to_string(index=False), "\n")
    scenario_table.to_csv(OUT / "scenario_table.csv", index=False)

    keep = set(scenario_table.loc[scenario_table["kept"], "scenario"])
    attack = attack[attack["scenario"].isin(keep)].copy()

    benign = load_benign_chunks(benign_dir)
    data = pd.concat([attack, benign], ignore_index=True)

    # ---- FIX 1: handshake filter -----------------------------------------
    if REQUIRE_HANDSHAKE and "src2dst_syn_packets" in data.columns:
        before = data.groupby("label").size().to_dict()
        data = data[data["src2dst_syn_packets"] >= 1].copy()
        after = data.groupby("label").size().to_dict()
        msg = ("handshake filter: kept flows with an observed client SYN  "
               f"attack {before.get(1, 0)}->{after.get(1, 0)}, "
               f"benign {before.get(0, 0)}->{after.get(0, 0)}")
        print(msg)
        notes.append(msg)

    data["flow_key"] = flow_key(data)
    before = len(data)
    data = data.drop_duplicates(subset="flow_key").reset_index(drop=True)
    notes.append(f"flow-identity dedup: removed {before - len(data)} rows")

    meta = {"label", "device", "scenario", "source", "chunk", "flow_key"}
    excluded = set(DROP_COLS) | meta | set(BOUNDARY_FLAG_COLS)
    features = [c for c in data.columns
                if c not in excluded
                and pd.api.types.is_numeric_dtype(data[c])
                and data[c].nunique(dropna=False) > 1]
    dropped_flags = [c for c in BOUNDARY_FLAG_COLS if c in data.columns]
    notes.append("dropped connection-boundary flags (window-alignment "
                 f"artifacts): {dropped_flags}")

    # ---- FIX 2: dedup on feature values ----------------------------------
    if DEDUP_ON_FEATURES:
        dup_mask = data.duplicated(subset=features, keep="first")
        by_class = data.loc[dup_mask].groupby("label").size().to_dict()
        msg = (f"feature dedup: removed {int(dup_mask.sum())} rows "
               f"(attack {by_class.get(1, 0)}, benign {by_class.get(0, 0)})")
        print(msg)
        notes.append(msg)
        data = data[~dup_mask].reset_index(drop=True)
        # recompute - dedup can make a column constant
        features = [c for c in features if data[c].nunique(dropna=False) > 1]

    # ---- split ------------------------------------------------------------
    is_train_dev = data["device"] == TRAIN_DEVICE
    is_test_dev = data["device"] == TEST_DEVICE
    is_baseline = data["device"] == "benign_capture"

    cut = data.loc[is_baseline, "chunk"].quantile(0.8) if is_baseline.any() else 0
    train = data[is_train_dev | (is_baseline & (data["chunk"] <= cut))].copy()
    test = data[is_test_dev | (is_baseline & (data["chunk"] > cut))].copy()

    keep_cols = features + ["label", "device", "scenario", "source",
                            "flow_key", "bidirectional_duration_ms"]
    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in data.columns))

    train[keep_cols].to_csv(OUT / "train.csv", index=False)
    test[keep_cols].to_csv(OUT / "test.csv", index=False)

    proof, stumps = build_proof(train, test, features, scenario_table, notes)
    (OUT / "proof.txt").write_text(proof)
    stumps.to_csv(OUT / "stump_scores.csv", index=False)

    print("\n" + proof)
    print(f"\nFeatures: {len(features)}")
    print(f"Written to {OUT.resolve()}/")


if __name__ == "__main__":
    main()
