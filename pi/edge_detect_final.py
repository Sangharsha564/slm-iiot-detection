#!/usr/bin/env python3
"""
The edge tier: detector on the Pi. Scores every flow, escalates only alerts.

    demo_flows.csv -> BERT-mini -> prob >= threshold ? -> MQTT -> laptop

Flows below the threshold are handled entirely on the device and never leave
it. That is the architecture, not an optimisation: the detector is 11M
parameters and milliseconds per flow, the explainer is 1.5B and seconds per
alert, and only a minority of flows are alerts. Each stage sits where its
cost belongs.

WHAT THE MESSAGE CARRIES, AND WHY
    The serialised flow text is included in full. Integrated Gradients on the
    laptop must run on the EXACT string the detector scored -- attributing a
    reconstructed or re-serialised input would explain something the detector
    never saw. Everything else (probability, threshold, model id) is there so
    the laptop can reproduce and verify the decision rather than trust it.

MEASURING TRANSFER TIME WITHOUT SYNCHRONISED CLOCKS
    The Pi's clock and the laptop's clock disagree by more than the quantity
    being measured, so subtracting one from the other is meaningless. Instead
    the laptop echoes the flow key back the instant it receives an alert,
    before doing any work, and the Pi times the round trip on its OWN clock.
    One clock, two readings, no synchronisation needed. One-way transfer is
    reported as half the round trip.

Requires:  pip install paho-mqtt
Broker:    sudo apt install mosquitto mosquitto-clients   (runs on this Pi)

Usage:
    python3 edge_detect.py --ckpt . --csv demo_flows.csv
"""

import argparse
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load(ckpt, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    mdl = AutoModelForSequenceClassification.from_pretrained(ckpt)
    mdl.to(device).eval()
    return tok, mdl


@torch.no_grad()
def score(tok, mdl, text, max_length, device):
    enc = tok(text, padding="max_length", truncation=True,
              max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = mdl(**enc).logits.float()
    return float(torch.softmax(logits, -1)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=".")
    ap.add_argument("--csv", default="demo_flows.csv")
    ap.add_argument("--broker", default="127.0.0.1",
                    help="the broker runs on this Pi, so localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="iiot/alerts")
    ap.add_argument("--ack-topic", default="iiot/ack")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: tuned_threshold from metrics.json")
    ap.add_argument("--ack-timeout", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--model-id", default="bert-mini_s42/readable192")
    ap.add_argument("--out", default="edge_log.csv")
    ap.add_argument("--no-ack", action="store_true",
                    help="skip the round-trip measurement and publish "
                         "fire-and-forget")
    args = ap.parse_args()

    import paho.mqtt.client as mqtt

    ckpt = Path(args.ckpt).resolve()
    thr = args.threshold
    if thr is None:
        for cand in (ckpt / "metrics.json", ckpt.parent / "metrics.json"):
            if cand.exists():
                thr = json.loads(cand.read_text()).get("tuned_threshold")
                if thr is not None:
                    print(f"threshold  {thr:.4f}  (from {cand})")
                    break
    if thr is None:
        raise SystemExit("no tuned_threshold found; pass --threshold")

    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        raise SystemExit(f"{args.csv} has no 'text' column: {list(df.columns)}")
    if args.limit:
        df = df.head(args.limit)

    print(f"host       {platform.node()}  {platform.machine()}")
    print(f"torch      {torch.__version__}  threads {torch.get_num_threads()}")
    tok, mdl = load(str(ckpt), "cpu")
    n_par = sum(p.numel() for p in mdl.parameters())
    print(f"model      {n_par/1e6:.1f}M parameters, float32, cpu, "
          f"max_length {args.max_length}")
    print(f"flows      {len(df)} from {args.csv}\n")

    # ---- MQTT -----------------------------------------------------------
    acked = {}                       # flow_key -> Event, set when the ack lands

    def on_message(client, userdata, msg):
        ev = acked.get(msg.payload.decode(errors="ignore").strip())
        if ev is not None:
            ev.set()

    try:
        cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)   # paho >= 2.0
    except (AttributeError, TypeError):
        cli = mqtt.Client()                                   # paho 1.x
    cli.on_message = on_message
    try:
        cli.connect(args.broker, args.port, keepalive=60)
    except Exception as e:                                    # noqa: BLE001
        raise SystemExit(
            f"cannot reach the broker at {args.broker}:{args.port} ({e}).\n"
            f"Is mosquitto running?   sudo systemctl status mosquitto")
    if not args.no_ack:
        cli.subscribe(args.ack_topic, qos=1)
    cli.loop_start()
    print(f"broker     {args.broker}:{args.port}  publishing to "
          f"'{args.topic}'" + ("" if args.no_ack else
                               f", acks on '{args.ack_topic}'"))
    print(f"           waiting for the laptop subscriber to be running\n")

    # warm-up: the first passes pay one-off allocation costs and would
    # inflate the per-flow detect time without describing anything real
    for t in df.text.astype(str).head(args.warmup):
        score(tok, mdl, t, args.max_length, "cpu")

    rows = []
    t_run = time.perf_counter()
    for i, r in df.iterrows():
        text = str(r.text)

        t0 = time.perf_counter()
        prob = score(tok, mdl, text, args.max_length, "cpu")
        detect_ms = (time.perf_counter() - t0) * 1000

        rec = {"flow_key": str(r.get("flow_key", i)),
               "scenario": r.get("scenario", ""),
               "label": r.get("label", ""),
               "prob": prob, "alerted": prob >= thr,
               "detect_ms": detect_ms, "payload_bytes": 0,
               "rtt_ms": None, "acked": None}

        if prob >= thr:
            payload = json.dumps({
                "v": 1,
                "flow_key": rec["flow_key"],
                "ts": time.time(),
                "scenario": rec["scenario"],
                "device": r.get("device", ""),
                "prob": round(prob, 6),
                "threshold": float(thr),
                "model": args.model_id,
                "max_length": args.max_length,
                # the exact string that was scored -- IG must attribute this
                "text": text,
            }, separators=(",", ":"))
            rec["payload_bytes"] = len(payload.encode())

            ev = threading.Event()
            if not args.no_ack:
                acked[rec["flow_key"]] = ev
            t1 = time.perf_counter()
            cli.publish(args.topic, payload, qos=1)
            if not args.no_ack:
                ok = ev.wait(args.ack_timeout)
                rec["rtt_ms"] = (time.perf_counter() - t1) * 1000
                rec["acked"] = ok
                acked.pop(rec["flow_key"], None)
                if not ok:
                    print(f"  WARNING no ack for {rec['flow_key'][:12]} "
                          f"within {args.ack_timeout}s -- is the laptop "
                          f"subscriber running?")

            print(f"  {i+1:4d}/{len(df)}  ALERT  {rec['flow_key'][:12]}  "
                  f"p={prob:.4f}  detect {detect_ms:5.1f} ms  "
                  f"{rec['payload_bytes']:4d} B" +
                  ("" if rec["rtt_ms"] is None
                   else f"  rtt {rec['rtt_ms']:6.1f} ms"), flush=True)
        rows.append(rec)

    run_s = time.perf_counter() - t_run
    cli.loop_stop()
    cli.disconnect()

    log = pd.DataFrame(rows)
    log.to_csv(args.out, index=False)

    n_alert = int(log.alerted.sum())
    d = log.detect_ms.to_numpy()
    print(f"\n  flows            {len(log)} in {run_s:.1f}s")
    print(f"  escalated        {n_alert}  ({100*n_alert/len(log):.1f}%)  "
          f"-- {len(log)-n_alert} handled at the edge and never transmitted")
    print(f"  detect latency   median {np.median(d):.1f} ms, "
          f"p95 {np.percentile(d,95):.1f} ms, max {d.max():.1f} ms (batch 1)")
    if n_alert:
        pb = log.loc[log.alerted, "payload_bytes"].to_numpy()
        print(f"  payload          mean {pb.mean():.0f} B, max {pb.max()} B")
        rt = log.loc[log.alerted, "rtt_ms"].dropna().to_numpy()
        if len(rt):
            print(f"  round trip       median {np.median(rt):.1f} ms, "
                  f"p95 {np.percentile(rt,95):.1f} ms")
            print(f"  one-way transfer median {np.median(rt)/2:.1f} ms "
                  f"(half the round trip, measured on this clock only)")
        n_bad = int((log.acked == False).sum())          # noqa: E712
        if n_bad:
            print(f"  UNACKED          {n_bad} alerts were never acknowledged")
    print(f"\n  written {args.out}")


if __name__ == "__main__":
    main()
