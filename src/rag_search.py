#!/usr/bin/env python3
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_build import embed, load_embedder

EXPLAIN_FRAME = ("Slowloris slow HTTP denial of service, an attack that holds "
                 "connections open without completing requests. ")


def explain_query(feature_labels):
    labs = ", ".join(feature_labels)
    return (EXPLAIN_FRAME + f"The flagged flow is unusual in {labs} compared "
            f"with normal traffic. What the attack does and why it looks "
            f"like this.")


def scenario_phrase(scenario):
    m = re.match(r"^\w+?_(.+?)_(\d+)$", str(scenario))
    if not m:
        return str(scenario).replace("_", " ")
    return f"an IIoT {m.group(1).replace('-', ' ')} on port {m.group(2)}"


def advise_query(scenario):
    return (f"How to mitigate a Slowloris slow-rate denial of service that "
            f"holds connections open against {scenario_phrase(scenario)}. "
            f"Server timeouts and connection limits that stop it.")


class Retriever:

    def __init__(self, rag_dir, device="cpu"):
        import json
        r = Path(rag_dir)
        if not (r / "index.npz").exists():
            raise SystemExit(
                f"no index at {r}. Build it first:\n"
                f"    python src/rag_build.py --docs docs/")
        self.V = np.load(r / "index.npz")["vectors"]
        self.chunks = json.loads((r / "chunks.json").read_text())
        self.tok, self.mdl = load_embedder(device)
        self.device = device

    def search(self, queries, per_query=2, per_source=2):
        Q = embed(list(queries), self.tok, self.mdl, self.device)
        S = self.V @ Q.T                          # (chunks, queries)
        taken, by_src, out = set(), {}, []
        for qi in range(len(queries)):
            got = 0
            for j in np.argsort(-S[:, qi]):
                if got >= per_query:
                    break
                j = int(j)
                src = self.chunks[j]["source"]
                if j in taken or by_src.get(src, 0) >= per_source:
                    continue
                taken.add(j)
                by_src[src] = by_src.get(src, 0) + 1
                out.append({**self.chunks[j], "score": float(S[j, qi]),
                            "query": qi})
                got += 1
        return out


def knowledge_block(hits):
    lines = ["KNOWLEDGE (background from security documentation. It explains "
             "the attack in general; EVIDENCE alone describes this flow. Do "
             "not take any number from here.)"]
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h['source']} p{h['page']}")
        lines.append(f"    {h['text']}")
    return "\n".join(lines)
