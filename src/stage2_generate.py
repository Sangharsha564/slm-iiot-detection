#!/usr/bin/env python3
"""

Usage:
    python src/stage2_generate.py --condition rag --limit 2   
    python src/stage2_generate.py --condition rag           
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xai_methods import split_segments

MODELS = {
    "qwen05":  "Qwen/Qwen2.5-0.5B-Instruct",
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen15":  "Qwen/Qwen2.5-1.5B-Instruct",
}

SYSTEM = """You are a network-security analyst explaining an Industrial-IoT intrusion-detection decision. A detector has flagged a network flow. You are given its CONFIDENCE and the most influential features identified by Integrated Gradients (IG). Explain the decision using ONLY the given EVIDENCE (and, when a KNOWLEDGE section is present, that knowledge). Never invent features, values, or attack behaviour.

Feature names are composed as: <scope> [statistic] <quantity>.
  scope      client = the host that opened the connection
             server = the IIoT device being contacted
             total  = both directions combined
  statistic  max / min / mean / stddev of that quantity across the flow.
             Absent when the quantity is already a single total or count.
  quantity   packet size    size of one packet, in bytes
             inter-arrival  gap between two consecutive packets, in seconds
             bytes          total volume sent
             psh packets    count of TCP packets with the PSH flag set,
                            meaning application data was delivered
             rst packets    count of TCP packets with the RST flag set,
                            meaning a connection was reset
"normal" in EVIDENCE is the median value of that feature across benign traffic from the same devices.

Each EVIDENCE factor gives: the feature, its value in this flow, the typical value in normal traffic, whether it RAISES or LOWERS suspicion, and how much of the decision it accounts for.

Rules:
1. Every number you write must exactly match a value in EVIDENCE.
2. Mention only features that appear in EVIDENCE.
3. Lead with the strongest factor and weight the rest by their stated influence.
4. If a factor LOWERS suspicion, say so. Do not present it as incriminating.
5. State EVIDENCE values as observed fact; mark any interpretation as inference ("consistent with", "suggests"). Make no claim EVIDENCE does not support.
6. Say what the factors indicate about the traffic pattern: whether one factor alone characterises the flow, or several combine to do so, and how.

Output ONLY this JSON object. No markdown fences, no text before or after.
{"verdict":"Attack (Slowloris)",
 "confidence":"<qualitative + numeric, e.g. very high (0.9992)>",
 "explanation":"<= 80 words: why this flow was flagged. Cite the EVIDENCE values, and say what they indicate about the traffic pattern, singly or in combination>",
 "mitigation":["<action>", "<action>"]}"""




FEWSHOT = [
    ("""VERDICT: Attack (Slowloris)
Scenario: ddos_wisenet-camera_80
Confidence: 0.9992 (alert threshold 0.24)

EVIDENCE
1. client max packet size = 234 bytes | normal: 106 | RAISES suspicion | 86% of the decision
2. server stddev packet size = 3.27 bytes | normal: 0.25 | LOWERS suspicion | 6% of the decision
3. server min inter-arrival = 0.0s | normal: 0.0s | LOWERS suspicion | 2% of the decision
Remaining 10 features: 7% combined, none individually significant.""",
     '{"verdict":"Attack (Slowloris)",'
     '"confidence":"very high (0.9992)",'
     '"explanation":"client max packet size is 234 bytes against a normal 106 '
     'and carries 86% of the decision, so this one feature characterises the '
     'flow rather than a combination. server stddev packet size is 3.27 against '
     'a normal 0.25, contributes 6%, and argues against the attack label rather '
     'than for it. server min inter-arrival is 0.0s, the same as normal, and '
     'also argues against the attack label, adding 2%.",'
     '"mitigation":["Record the flow identifier and source address for follow-up",'
     '"Review other flows from this device in the same time window"]}'),

    ("""VERDICT: Attack (Slowloris)
Scenario: ddos_wisenet-camera_80
Confidence: 0.9986 (alert threshold 0.24)

EVIDENCE
1. client max packet size = 255 bytes | normal: 106 | RAISES suspicion | 43% of the decision
2. server bytes = 296 bytes | normal: 866 | LOWERS suspicion | 9% of the decision
3. server min inter-arrival = 1.0s | normal: 0.0s | RAISES suspicion | 8% of the decision
Remaining 10 features: 40% combined, none individually significant.""",
     '{"verdict":"Attack (Slowloris)",'
     '"confidence":"very high (0.9986)",'
     '"explanation":"No single factor dominates this flow. client max packet '
     'size is 255 bytes against a normal 106 and carries 43% of the decision. '
     'server min inter-arrival is 1.0s against a normal 0.0s and adds 8%, so '
     'these two combine rather than either being decisive alone. server bytes '
     'is 296 against a normal 866, contributes 9%, and argues against the '
     'attack label rather than for it.",'
     '"mitigation":["Check whether the same source appears in other recent alerts",'
     '"Escalate to the device owner if the pattern persists"]}'),
]


def unit_of(label):
    if "inter-arrival" in label:
        return ""
    if "packet size" in label or label.endswith("bytes"):
        return " bytes"
    return ""


def build_evidence(row, labels, baseline, top_k=3):
    a = row[[f"attr_{l}" for l in labels]].to_numpy(dtype=float)
    order = np.argsort(-np.abs(a))
    top, rest = order[:top_k], order[top_k:]

    value_str = dict(split_segments(row.text))
    shares = {l: float(row[f"share_{l}"]) for l in labels}

    lines = [
        "VERDICT: Attack (Slowloris)",
        f"Scenario: {row.scenario}",
        f"Confidence: {row.prob:.4f} (alert threshold 0.24)",
        "",
        "EVIDENCE",
    ]
    for n, j in enumerate(top, 1):
        lab = labels[j]
        val = value_str.get(lab, f"{row[f'val_{lab}']:g}")
        u = unit_of(lab)
        lines.append(
            f"{n}. {lab} = {val}{u} | normal: {baseline.get(lab, 'unknown')} | "
            f"{'RAISES' if a[j] > 0 else 'LOWERS'} suspicion | "
            f"{100*shares[lab]:.0f}% of the decision")
    lines.append(
        f"Remaining {len(rest)} features: "
        f"{100*sum(shares[labels[j]] for j in rest):.0f}% combined, "
        f"none individually significant.")
    return "\n".join(lines)

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def parse_json(text):
    raw = str(text).strip()
    for how, candidate in (("clean", raw), ("fenced", FENCE.sub("", raw).strip())):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, how
        except (json.JSONDecodeError, TypeError):
            pass

    t = FENCE.sub("", raw).strip()
    start = t.find("{")
    if start < 0:
        return None, "no_object"
    stack, in_str, esc = [], False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                try:
                    obj = json.loads(t[start:i + 1])
                    if isinstance(obj, dict):
                        return obj, "embedded"
                except json.JSONDecodeError:
                    break
    if stack or in_str:
        tail = t[start:]
        if in_str:
            tail += '"'
        tail = tail.rstrip().rstrip(",")
        closing = ("".join("}" if c == "{" else "]" for c in reversed(stack))
                   or "}")
        try:
            obj = json.loads(tail + closing)
            if isinstance(obj, dict):
                return obj, "repaired"
        except json.JSONDecodeError:
            pass

    obj = salvage(t)
    if obj:
        return obj, "salvaged"
    return None, "unparseable"


def salvage(t):
    out = {}
    for k in ("verdict", "confidence", "explanation"):
       
        m = re.search(rf'{k}"?\s*:\s*"((?:[^"\\]|\\.)*)"', t, re.I)
        if m:
            out[k] = m.group(1)
    
    m = re.search(r'mitigation"?\s*:\s*[\[{](.*?)[\]}]', t, re.I | re.S)
    if m:
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
        if items:
            out["mitigation"] = items
    return out or None


def fields(obj):
    if not obj:
        return {"verdict": "", "confidence": "", "explanation": "",
                "mitigation": ""}
    mit = obj.get("mitigation", "")
    if isinstance(mit, (list, tuple)):
        mit = "; ".join(str(m) for m in mit)
    return {"verdict": str(obj.get("verdict", "")),
            "confidence": str(obj.get("confidence", "")),
            "explanation": str(obj.get("explanation", "")),
            "mitigation": str(mit)}

def load_model(name, device, dtype, trust_remote_code=False, attn=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    kw = {"dtype": dtype, "trust_remote_code": trust_remote_code}
    if attn:
        kw["attn_implementation"] = attn
    mdl = AutoModelForCausalLM.from_pretrained(name, **kw).to(device)
    mdl.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, mdl


def context_limit(tok, mdl):
    cand = [getattr(mdl.config, "max_position_embeddings", None),
            getattr(tok, "model_max_length", None)]
    vals = [int(c) for c in cand if isinstance(c, int) and 0 < c < 10 ** 7]
    return min(vals) if vals else None


def check_headroom(tok, mdl, system, evidences, max_new_tokens, label,
                   shots=()):
    lens = []
    for ev in evidences:
        msgs = [{"role": "system", "content": system}]
        for ex_user, ex_reply in shots:
            msgs.append({"role": "user", "content": ex_user})
            msgs.append({"role": "assistant", "content": ex_reply})
        msgs.append({"role": "user", "content": ev})
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        except Exception:                                   # noqa: BLE001
            text = system + "".join(a + b for a, b in shots) + ev
        lens.append(len(tok(text)["input_ids"]))

    longest, limit = max(lens), context_limit(tok, mdl)
    need = longest + max_new_tokens
    if limit is None:
        print(f"  tokens   prompt {min(lens)}-{longest}, "
              f"+{max_new_tokens} output; context window unknown")
        return
    pct = 100 * need / limit
    print(f"  tokens   prompt {min(lens)}-{longest}, +{max_new_tokens} output "
          f"= {need} of {limit} ({pct:.1f}% of the window)")
    if need > limit:
        raise SystemExit(
            f"  {label}: prompt + output ({need}) exceeds the {limit}-token "
            f"window. Reduce --max-new-tokens, or shorten the added context.")
    if pct > 75:
        print(f"  WARNING  under 25% headroom. Adding examples or retrieved "
              f"chunks will overflow this model.")


@torch.no_grad()
def generate(tok, mdl, system, user, device, max_new_tokens, shots=()):
    msgs = [{"role": "system", "content": system}]
    for ex_user, ex_reply in shots:
        msgs.append({"role": "user", "content": ex_user})
        msgs.append({"role": "assistant", "content": ex_reply})
    msgs.append({"role": "user", "content": user})
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:                                       # no system role
        flat = system + "\n\n"
        for ex_user, ex_reply in shots:
            flat += f"{ex_user}\n{ex_reply}\n\n"
        text = tok.apply_chat_template(
            [{"role": "user", "content": flat + user}],
            tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    dt = time.perf_counter() - t0
    gen = out[0][enc["input_ids"].shape[1]:]
    return (tok.decode(gen, skip_special_tokens=True).strip(), dt,
            int(gen.shape[0]), int(enc["input_ids"].shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dir", default="results/08_stage2")
    ap.add_argument("--xai-dir", default="results/07_xai")
    ap.add_argument("--condition", choices=["zero_shot", "few_shot", "rag"],
                    default="zero_shot",
                    help="'rag' is few-shot PLUS retrieved knowledge: the "
                         "system prompt and both worked examples are byte-for-"
                         "byte the same, so the only variable against the "
                         "few-shot condition is the KNOWLEDGE block")
    ap.add_argument("--rag", default="rag", help="folder holding index.npz")
    ap.add_argument("--per-query", type=int, default=2)
    ap.add_argument("--per-source", type=int, default=2,
                    help="cap on chunks from any one document. Without it the "
                         "Cloudflare Slowloris page takes every explain slot")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=400,
                    help="generous on purpose. A compliant answer is about "
                         "200 tokens (80-word explanation, confidence, two "
                         "mitigations, JSON syntax), so 400 leaves 2x "
                         "headroom. Output that still truncates at 400 is "
                         "the model ignoring the word limit, which is a "
                         "finding; truncation caused by our own cap is not.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--attn", default=None,
                    choices=["eager", "sdpa"],
                    help="attention kernel. Phi-3.5 on MPS is often far "
                         "faster with 'eager'; leave unset to let "
                         "transformers choose.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    d = root / args.dir
    d.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

    sample = pd.read_csv(d / "explainer_sample.csv")
    baseline = json.loads(
        (root / args.xai_dir / "ig_baseline.json").read_text())["baseline"]
    labels = [c[5:] for c in sample.columns if c.startswith("attr_")]
    if args.limit:
        sample = sample.head(args.limit)

    shots = FEWSHOT if args.condition in ("few_shot", "rag") else ()
    print(f"condition  {args.condition}"
          + (f"   {len(shots)} worked examples" if shots else "")
          + ("   + retrieved knowledge" if args.condition == "rag" else ""))
    print(f"device     {device} ({dtype})")
    print(f"alerts     {len(sample)}   factors per alert: {args.top_k}")

    retriever = None
    if args.condition == "rag":
        from rag_search import Retriever, advise_query, explain_query, \
            knowledge_block
        retriever = Retriever(root / args.rag, "cpu")
        print(f"index      {retriever.V.shape[0]} chunks from "
              f"{len({c['source'] for c in retriever.chunks})} documents")

    prompts, retrieve_times = [], []
    for _, r in sample.iterrows():
        ev = build_evidence(r, labels, baseline, args.top_k)
        rec = {"flow_key": r.flow_key, "scenario": r.scenario,
               "prob": r.prob, "evidence": ev, "user": ev,
               "knowledge": "", "knowledge_sources": "", "retrieve_seconds": 0.0}
        if retriever is not None:
            a = r[[f"attr_{l}" for l in labels]].to_numpy(dtype=float)
            top = [labels[j] for j in np.argsort(-np.abs(a))[:args.top_k]]
            t0 = time.perf_counter()
            hits = retriever.search(
                [explain_query(top), advise_query(r.scenario)],
                per_query=args.per_query, per_source=args.per_source)
            rec["retrieve_seconds"] = time.perf_counter() - t0
            know = knowledge_block(hits)
            rec["knowledge"] = know
            rec["knowledge_sources"] = "; ".join(
                f"{h['source']} p{h['page']} ({h['score']:.2f})" for h in hits)
            rec["user"] = f"{ev}\n\n{know}"
            retrieve_times.append(rec["retrieve_seconds"])
        prompts.append(rec)
    pd.DataFrame(prompts).to_csv(d / f"{args.condition}_prompts.csv", index=False)

    if retrieve_times:
        n_chunks = len(prompts[0]["knowledge_sources"].split(";"))
        print(f"retrieval  {n_chunks} chunks per alert, "
              f"{1000*np.mean(retrieve_times):.0f} ms per alert "
              f"({1000*np.max(retrieve_times):.0f} ms worst case)")
        srcs = {}
        for p in prompts:
            for s in p["knowledge_sources"].split("; "):
                name = s.split(" p")[0]
                srcs[name] = srcs.get(name, 0) + 1
        tot = sum(srcs.values())
        for s, n in sorted(srcs.items(), key=lambda kv: -kv[1]):
            print(f"             {s[:44]:46s} {n:4d}  ({100*n/tot:4.1f}%)")
        n_distinct = len({p["knowledge"] for p in prompts})
        print(f"           {n_distinct} distinct knowledge blocks across "
              f"{len(prompts)} alerts")

    if shots:
        overlap = {p["flow_key"] for p in prompts
                   if p["evidence"].strip() in {e.strip() for e, _ in shots}}
        if overlap:
            raise SystemExit(
                f"LEAKAGE: a worked example is identical to a test alert "
                f"({overlap}). The examples must come from outside the 20.")
        print(f"examples   {len(shots)}, none matching a test alert")

    n_uniq = len({p["evidence"] for p in prompts})
    n_mitig = sum("LOWERS" in p["evidence"] for p in prompts)
    print(f"prompts    {len(prompts)} built, {n_uniq} distinct")
    print(f"           {n_mitig} contain a factor that LOWERS suspicion")

    print(f"\n{'='*72}\nSYSTEM\n{'='*72}\n{SYSTEM}")
    print(f"\n{'='*72}\nUSER MESSAGE (alert 1)\n{'='*72}\n{prompts[0]['user']}")
    print("=" * 72)

    rows = []
    out_path = d / f"{args.condition}_explanations.csv"
    for key in args.models:
        model_id = MODELS.get(key, key)
        print(f"\nloading {model_id} ...")
        try:
            tok, mdl = load_model(model_id, device, dtype,
                                  args.trust_remote_code, args.attn)
        except Exception as e:                             
            print(f"  SKIPPED: {type(e).__name__}: {e}")
            if "llama" in model_id.lower():
                print("  Llama-3.2 is gated: accept the licence and run "
                      "`huggingface-cli login`, or substitute "
                      "Qwen/Qwen2.5-1.5B-Instruct")
            continue

        check_headroom(tok, mdl, SYSTEM, [p["user"] for p in prompts],
                       args.max_new_tokens, key, shots)

        t0 = time.perf_counter()
        done = 0
        try:
          for i, p in enumerate(prompts):
            raw, dt, n_new, n_in = generate(tok, mdl, SYSTEM, p["user"],
                                            device, args.max_new_tokens, shots)
            obj, how = parse_json(raw)
            f = fields(obj)
            rows.append({
                "flow_key": p["flow_key"], "scenario": p["scenario"],
                "prob": p["prob"], "condition": args.condition,
                "model": key, "model_id": model_id,
                "raw_output": raw, "json_status": how,
                "valid_json": obj is not None, **f,
                "knowledge_sources": p["knowledge_sources"],
                "retrieve_seconds": p["retrieve_seconds"],
                "text_for_eval": f["explanation"] or raw,
                "gen_seconds": dt, "prompt_tokens": n_in, "gen_tokens": n_new,
                "truncated": n_new >= args.max_new_tokens,
                "tokens_per_second": n_new / dt if dt else 0.0})
            done = i + 1
            
            el_so_far = time.perf_counter() - t0
            eta = el_so_far / done * (len(prompts) - done)
            print(f"  {done:2d}/{len(prompts)}  {dt:6.1f}s  "
                  f"{n_new:3d} tok  {n_new/dt if dt else 0:5.1f} tok/s  "
                  f"eta {eta/60:4.1f}m", flush=True)
        except KeyboardInterrupt:
            print(f"\n  interrupted after {done}/{len(prompts)} alerts")
        el = time.perf_counter() - t0
        g = pd.DataFrame(rows[-len(prompts):])
        print(f"  {key}: {el:.1f}s total, {el/len(prompts):.2f}s per alert, "
              f"valid JSON {int(g.valid_json.sum())}/{len(g)}")
        print(f"\n  --- {key}, alert 1 (raw) ---\n  {rows[-len(prompts)]['raw_output'][:600]}")

        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  saved {len(rows)} rows so far -> {out_path.name}")

        del mdl
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    if not rows:
        raise SystemExit("no model produced output")

    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False)

    print(f"\n  {'model':8s} {'n':>4s} {'valid':>7s} {'trunc':>6s} "
          f"{'s/alert':>8s} {'tok/s':>7s} {'expl words':>11s}")
    print("  " + "-" * 56)
    for m, g in res.groupby("model"):
        w = g.explanation.fillna("").str.split().str.len()
        print(f"  {m:8s} {len(g):4d} {int(g.valid_json.sum()):3d}/{len(g):<3d} "
              f"{int(g.truncated.sum()):6d} {g.gen_seconds.mean():8.2f} "
              f"{g.tokens_per_second.mean():7.1f} {w.mean():11.1f}")

    bad = res[~res.valid_json]
    if len(bad):
        print(f"\n  JSON failures by cause")
        for (m, how), gg in bad.groupby(["model", "json_status"]):
            print(f"    {m:8s} {how:14s} {len(gg)}")
    fixed = res[res.json_status.isin(["fenced", "embedded", "repaired"])]
    if len(fixed):
        print(f"\n  {len(fixed)} outputs needed recovery "
              f"({dict(fixed.json_status.value_counts())}) -- valid content, "
              f"imperfect formatting")

    print(f"\nwritten to {d}")


if __name__ == "__main__":
    main()