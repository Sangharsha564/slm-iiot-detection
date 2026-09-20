#!/usr/bin/env python3
"""
Requires:  pip install paho-mqtt

Usage:
    python src/analyst_explain.py --broker 192.168.1.42
    python src/analyst_explain.py --broker 192.168.1.42 --expect 50
"""

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_search import Retriever, advise_query, explain_query, knowledge_block
from stage2_generate import (FEWSHOT, MODELS, SYSTEM, fields, generate,
                             load_model, parse_json, unit_of)
from xai_methods import integrated_gradients, split_segments


def build_evidence(scores, text, baseline, prob, scenario, threshold,
                   top_k=3):
    labels = list(scores)
    a = np.array([scores[l] for l in labels], dtype=float)
    order = np.argsort(-np.abs(a))
    top, rest = order[:top_k], order[top_k:]

    total = float(np.abs(a).sum()) or 1.0
    value_str = dict(split_segments(text))

    lines = ["VERDICT: Attack (Slowloris)",
             f"Scenario: {scenario}",
             f"Confidence: {prob:.4f} (alert threshold {threshold:.2f})",
             "", "EVIDENCE"]
    for n, j in enumerate(top, 1):
        lab = labels[j]
        lines.append(
            f"{n}. {lab} = {value_str.get(lab, '?')}{unit_of(lab)} | "
            f"normal: {baseline.get(lab, 'unknown')} | "
            f"{'RAISES' if a[j] > 0 else 'LOWERS'} suspicion | "
            f"{100*abs(a[j])/total:.0f}% of the decision")
    lines.append(
        f"Remaining {len(rest)} features: "
        f"{100*sum(abs(a[j]) for j in rest)/total:.0f}% combined, "
        f"none individually significant.")
    return "\n".join(lines), [labels[j] for j in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--ckpt",
                    default="experiments/stage1_full_13_readable192/"
                            "bert-mini_s42/checkpoint")
    ap.add_argument("--xai-dir", default="results/07_xai")
    ap.add_argument("--rag", default="rag")
    ap.add_argument("--broker", required=True, help="the Pi's IP address")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="iiot/alerts")
    ap.add_argument("--ack-topic", default="iiot/ack")
    ap.add_argument("--model", default="qwen15", choices=list(MODELS))
    ap.add_argument("--ig-steps", type=int, default=50)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--per-query", type=int, default=2)
    ap.add_argument("--per-source", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--gen-device", default="auto",
                    help="device for the 1.5B generator. IG always uses cpu "
                         "float32 to match the Pi")
    ap.add_argument("--expect", type=int, default=None,
                    help="stop after this many alerts. Default: run until "
                         "interrupted with Ctrl-C")
    ap.add_argument("--out", default="results/10_pipeline/pipeline_log.csv")
    args = ap.parse_args()

    import paho.mqtt.client as mqtt

    root = Path(args.root).resolve()
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    gen_dev = args.gen_device
    if gen_dev == "auto":
        gen_dev = ("mps" if torch.backends.mps.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")
    gen_dtype = torch.float16 if gen_dev in ("mps", "cuda") else torch.float32

    print("loading ...")
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    det_tok = AutoTokenizer.from_pretrained(str(root / args.ckpt))
    det = AutoModelForSequenceClassification.from_pretrained(
        str(root / args.ckpt)).to("cpu").eval()          # float32, matches Pi

    bl = json.loads((root / args.xai_dir / "ig_baseline.json").read_text())
    baseline, baseline_text = bl["baseline"], bl["baseline_text"]

    retriever = Retriever(root / args.rag, "cpu")
    model_id = MODELS[args.model]
    gen_tok, gen_mdl = load_model(model_id, gen_dev, gen_dtype)

    print(f"  detector   {sum(p.numel() for p in det.parameters())/1e6:.1f}M, "
          f"float32, cpu   (IG {args.ig_steps} steps)")
    print(f"  index      {retriever.V.shape[0]} chunks")
    print(f"  generator  {model_id} on {gen_dev} ({gen_dtype})")


    inbox = queue.Queue()

    def on_connect(client, userdata, flags, rc, properties=None):
        client.subscribe(args.topic, qos=1)
        print(f"\nsubscribed to '{args.topic}' at {args.broker}:{args.port}")
        print("waiting for alerts. Start edge_detect.py on the Pi.\n")

    def on_message(client, userdata, msg):
    
        try:
            alert = json.loads(msg.payload.decode())
            client.publish(args.ack_topic, alert.get("flow_key", ""), qos=1)
            inbox.put((time.time(), alert))
        except Exception as e:                            
            print(f"  malformed message ignored: {e}")

    try:
        cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except (AttributeError, TypeError):
        cli = mqtt.Client()
    cli.on_connect = on_connect
    cli.on_message = on_message
    try:
        cli.connect(args.broker, args.port, keepalive=60)
    except Exception as e:                                
        raise SystemExit(
            f"cannot reach the broker at {args.broker}:{args.port} ({e}).\n"
            f"On the Pi:  sudo systemctl status mosquitto\n"
            f"and check both machines are on the same Wi-Fi.")
    cli.loop_start()

    rows, n = [], 0
    try:
        while args.expect is None or n < args.expect:
            try:
                t_recv, alert = inbox.get(timeout=1.0)
            except queue.Empty:
                continue
            n += 1
            text = alert["text"]
            t_all = time.perf_counter()

            t0 = time.perf_counter()
            scores, diag = integrated_gradients(
                det, det_tok, text, baseline_text, steps=args.ig_steps,
                max_length=args.max_length, chunk=args.chunk)
            ig_ms = (time.perf_counter() - t0) * 1000

            evidence, top = build_evidence(
                scores, text, baseline, alert["prob"], alert["scenario"],
                alert["threshold"])

            t0 = time.perf_counter()
            hits = retriever.search(
                [explain_query(top), advise_query(alert["scenario"])],
                per_query=args.per_query, per_source=args.per_source)
            know = knowledge_block(hits)
            retrieve_ms = (time.perf_counter() - t0) * 1000

            user = f"{evidence}\n\n{know}"
            raw, gen_s, n_new, n_in = generate(
                gen_tok, gen_mdl, SYSTEM, user, gen_dev,
                args.max_new_tokens, FEWSHOT)
            obj, how = parse_json(raw)
            f = fields(obj)
            total_ms = (time.perf_counter() - t_all) * 1000

            rows.append({
                "flow_key": alert["flow_key"], "scenario": alert["scenario"],
                "prob": alert["prob"], "top_features": "; ".join(top),
                "ig_ms": round(ig_ms, 1),
                "retrieve_ms": round(retrieve_ms, 1),
                "generate_ms": round(gen_s * 1000, 1),
                "total_ms": round(total_ms, 1),
                "completeness_error": round(diag["completeness_error"], 4),
                "unassigned_frac": round(diag["unassigned_frac"], 4),
                "prompt_tokens": n_in, "gen_tokens": n_new,
                "valid_json": obj is not None, "json_status": how, **f,
                "knowledge_sources": "; ".join(
                    f"{h['source']} p{h['page']}" for h in hits),
                "evidence": evidence, "raw_output": raw,
                "received_at": t_recv,
            })
        
            pd.DataFrame(rows).to_csv(out, index=False)

            print(f"  {n:3d}  {alert['flow_key'][:12]}  p={alert['prob']:.4f}  "
                  f"IG {ig_ms:6.0f}  ret {retrieve_ms:5.0f}  "
                  f"gen {gen_s*1000:6.0f}  total {total_ms:6.0f} ms  "
                  f"json={'ok' if obj else how}", flush=True)
            print(f"       {f['explanation'][:150]}", flush=True)
    except KeyboardInterrupt:
        print(f"\n  stopped after {n} alerts")
    finally:
        cli.loop_stop()
        cli.disconnect()

    if not rows:
        print("\n  no alerts received. Was edge_detect.py running, and did it "
              "reach the broker?")
        return

    t = pd.DataFrame(rows)
    print(f"\n  {len(t)} alerts explained")
    print(f"  {'stage':12s}{'median ms':>12s}{'p95 ms':>10s}{'share':>8s}")
    tot = t.total_ms.median()
    for c, name in (("ig_ms", "IG"), ("retrieve_ms", "retrieve"),
                    ("generate_ms", "generate"), ("total_ms", "TOTAL")):
        print(f"  {name:12s}{t[c].median():12.0f}{t[c].quantile(.95):10.0f}"
              f"{100*t[c].median()/tot:7.0f}%")
    print(f"\n  valid JSON        {int(t.valid_json.sum())}/{len(t)}")
    print(f"  completeness err  median {t.completeness_error.median():.3f} "
          f"(IG at {args.ig_steps} steps)")
    print(f"\n  written {out}")


if __name__ == "__main__":
    main()
