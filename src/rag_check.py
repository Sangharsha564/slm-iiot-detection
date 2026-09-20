#!/usr/bin/env python3
"""
Usage:
    python src/rag_check.py
    python src/rag_check.py --k 5 --show-text
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_build import EMBED_MODEL, embed, load_embedder
from rag_search import advise_query, explain_query

QUERIES = (
    [("explain", explain_query(f)) for f in [
        ["client max packet size", "server stddev packet size",
         "server min inter-arrival"],
        ["client max packet size", "server bytes", "server min inter-arrival"],
        ["client mean inter-arrival", "total rst packets",
         "server max packet size"],
    ]]
    + [("advise", advise_query(s)) for s in
       ["ddos_wisenet-camera_80", "ddos_dlink-camera_8080"]]
    + [("advise", "how to mitigate a slow-rate denial of service that holds "
                  "connections open"),
       ("advise", "reverse proxy buffering to protect a weak backend")]
    + [("control", "client max packet size, large packets from the client"),
       ("control", "server stddev packet size, variation in server packet sizes"),
       ("control", "inter-arrival time, long gaps between packets"),
       ("control", "TCP PSH and RST flags, what they mean")]
)
FOCUSED = ("cloudflare", "imperva", "attck", "ha.cker", "slowloris")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag", default="rag")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--show-text", action="store_true",
                    help="print the full chunk, not just the opening")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    r = Path(args.rag)
    V = np.load(r / "index.npz")["vectors"]
    chunks = json.loads((r / "chunks.json").read_text())
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    tok, mdl = load_embedder(device)
    Q = embed([q for _, q in QUERIES], tok, mdl, device)

    print(f"index     {V.shape[0]} chunks x {V.shape[1]} dims from "
          f"{len({c['source'] for c in chunks})} documents\n")

    by_kind = {}                       
    scores = {}                        
    hit_texts = []                     
    for (kind, q), qv in zip(QUERIES, Q):
        sims = V @ qv                  
        top = np.argsort(-sims)[:args.k]
        hit_texts.append([chunks[int(j)]["text"] for j in top])
        print(f"  [{kind}] {q}")
        for j in top:
            c = chunks[j]
            body = c["text"] if args.show_text else c["text"][:96] + "..."
            print(f"    {sims[j]:.3f}  {c['source'][:34]:36s} p{c['page']:<4d}")
            print(f"           {body}")
            d = by_kind.setdefault(kind, {})
            d[c["source"]] = d.get(c["source"], 0) + 1
        scores.setdefault(kind, []).append(float(sims[top[0]]))
        print()

    try:
        from stage2_eval import SLOWRATE_ADVICE
    except Exception:                                       
        SLOWRATE_ADVICE = None
    if SLOWRATE_ADVICE is not None:
        adv = [c for (kind, _), c in zip(QUERIES, hit_texts) if kind == "advise"]
        hit = [t for chunk_list in adv for t in chunk_list
               if SLOWRATE_ADVICE.search(t)]
        n_tot = sum(len(x) for x in adv)
        print(f"  ACTIONABLE ADVICE IN RETRIEVED TEXT")
        print(f"    {len(hit)}/{n_tot} advise chunks match the evaluator's "
              f"slow-rate advice pattern")
        if hit:
            m = SLOWRATE_ADVICE.search(hit[0])
            print(f"    e.g. \"...{hit[0][max(0, m.start()-40):m.end()+40]}...\"")
        else:
            print(f"    WARNING: none. advice_grounded cannot exceed what the "
                  f"model already knew,")
            print(f"    so a low score would not be a retrieval-corrected "
                  f"result. Fix retrieval first.")
        print()

    def focused_share(counts):
        good = sum(n for s, n in counts.items()
                   if any(f in s.lower() for f in FOCUSED))
        return good, sum(counts.values()) or 1

    live = {k: v for k, v in by_kind.items() if k != "control"}
    total = sum(sum(v.values()) for v in live.values())
    merged = {}
    for v in live.values():
        for s, n in v.items():
            merged[s] = merged.get(s, 0) + n
    print(f"  SOURCE MIX over the {total} slots that reach a prompt "
          f"(control queries excluded)")
    for s, n in sorted(merged.items(), key=lambda kv: -kv[1]):
        share = 100 * n / total
        flag = "   <-- dominating" if share > 50 else ""
        print(f"    {s[:44]:46s} {n:4d}  ({share:4.1f}%){flag}")

    print(f"\n  SLOTS FROM A SLOWLORIS-SPECIFIC DOCUMENT")
    for kind in ("explain", "advise", "control"):
        if kind not in by_kind:
            continue
        good, tot = focused_share(by_kind[kind])
        med = float(np.median(scores[kind]))
        note = "   <-- the phrasing we replaced" if kind == "control" else ""
        print(f"    {kind:9s} {good:3d}/{tot:<3d} ({100*good/tot:3.0f}%)   "
              f"median top-1 similarity {med:.3f}{note}")

    for kind, why in (("advise", "advice_grounded would fail for a retrieval "
                                 "reason, not a model one"),
                      ("explain", "the model gets no attack knowledge and RAG "
                                  "cannot beat few-shot")):
        if kind not in by_kind:
            continue
        good, tot = focused_share(by_kind[kind])
        if good / tot < 0.5:
            print(f"\n    WARNING [{kind}]: under half. {why}.")
            print(f"    Fix retrieval before generating: add keyword matching, "
                  f"raise k, or restrict the corpus.")

    # readability: chunks quoted to a model should read as prose
    bad = [c for c in chunks if c["text"][:1].islower()
           or c["text"].count("  ") > 3]
    print(f"\n  READABILITY  {len(bad)} of {len(chunks)} chunks start "
          f"mid-sentence or contain table debris "
          f"({100*len(bad)/len(chunks):.1f}%)")
    for c in bad[:3]:
        print(f"    {c['source'][:30]:32s} p{c['page']:<4d} {c['text'][:80]}...")


if __name__ == "__main__":
    main()