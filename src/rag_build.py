#!/usr/bin/env python3
"""

Usage:
    python src/rag_build.py --docs docs/
    python src/rag_build.py --docs docs/ --chunk-tokens 150
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HARD_LIMIT = 256                      # the encoder's input limit

# A back-matter heading on its own line: everything after it is citations or
# credits, which retrieve well on keyword overlap and answer nothing. The
# section number and the Normative/Informative qualifier are optional because
# RFCs head these sections "6. References" and "10.1. Normative References" --
# a bare-word pattern never fires on the document that needs it most.
REFS = re.compile(
    r"^[ \t]*(?:appendix[ \t]+[A-Z0-9]+[ \t]*[.:–—-]?[ \t]*)?"
    r"(?:\d+(?:\.\d+)*\.?[ \t]+)?"
    r"(?:normative[ \t]+|informative[ \t]+)?"
    r"(?:references|bibliography|works[ \t]+cited|acknowledge?ments?)"
    r"[ \t]*$", re.I | re.M)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
RFC_PAGE = re.compile(r"^\s*(\[Page \d+\]|RFC \d+.*\d{4})\s*$", re.M)


def read_doc(path: Path) -> list[tuple[int, str]]:
    """-> [(page_number, raw_text)]. Page 1 for plain text files."""
    if path.suffix.lower() == ".txt":
        return [(1, path.read_text(errors="ignore"))]
    try:
        import pymupdf as fitz          # the `fitz` alias is deprecated
    except ImportError:
        try:
            import fitz
        except ImportError:
            raise SystemExit("PDF support needs PyMuPDF:  pip install pymupdf")
    doc = fitz.open(path)
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


def drop_back_matter(pages):
    total = sum(len(t) for _, t in pages) or 1
    seen, out = 0, []
    for pageno, text in pages:
        hit = next((m for m in REFS.finditer(text)
                    if seen + m.start() > 0.5 * total), None)
        if hit is not None:
            head = text[:hit.start()]
            if head.strip():
                out.append((pageno, head))
            return out                      
        out.append((pageno, text))
        seen += len(text)
    return out


def clean(text: str) -> str:
    text = RFC_PAGE.sub("", text)
    text = text.replace("\x0c", "\n")                  
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      
    keep = [ln for ln in text.split("\n") if len(ln.split()) >= 3]
    return re.sub(r"\s+", " ", " ".join(keep)).strip()


def chunk_text(text, tok, target, overlap=1):
    sents = [s.strip() for s in SENT.split(text) if s.strip()]
    out, cur, cur_n = [], [], 0
    for s in sents:
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        if n > HARD_LIMIT:                              
            ids = tok(s, add_special_tokens=False)["input_ids"]
            for i in range(0, len(ids), target):
                out.append(tok.decode(ids[i:i + target]))
            continue
        if cur and cur_n + n > target:
            out.append(" ".join(cur))
            carry = cur[-overlap:] if overlap else []
            carry_n = sum(len(tok(x, add_special_tokens=False)["input_ids"])
                          for x in carry)
            if carry_n + n > HARD_LIMIT or carry_n > target // 2:
                carry, carry_n = [], 0
            cur, cur_n = carry, carry_n
        cur.append(s)
        cur_n += n
    if cur:
        out.append(" ".join(cur))
    return [c for c in out if len(c.split()) >= 12]     


def load_embedder(device):
    from transformers import AutoModel, AutoTokenizer, logging
    logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    mdl = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()
    return tok, mdl


@torch.no_grad()
def embed(texts, tok, mdl, device, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True,
                  max_length=HARD_LIMIT, return_tensors="pt").to(device)
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs", help="folder of PDFs and .txt")
    ap.add_argument("--out", default="rag")
    ap.add_argument("--chunk-tokens", type=int, default=150)
    ap.add_argument("--overlap", type=int, default=1, help="sentences")
    ap.add_argument("--dedupe", type=float, default=0.95)
    ap.add_argument("--exclude", nargs="*", default=["rfc793", "800-82r3"],
                    help="filename substrings to skip. rfc793 is obsoleted by "
                         "rfc9293 and its text is near-identical. 800-82r3 was "
                         "half the index and won none of the 33 slots in "
                         "rag_check, so it is pruned by measured contribution; "
                         "pass --exclude rfc793 to put it back")
    ap.add_argument("--drop-phrases", nargs="*",
                    default=["exact article content"],
                    help="skip chunks containing any of these (case-"
                         "insensitive). The default catches the provenance "
                         "header added when the web articles were saved as "
                         "PDFs: it is about the document rather than the "
                         "attack, and it was taking top-3 retrieval slots")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    docs = Path(args.docs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    files = sorted(p for p in docs.iterdir()
                   if p.suffix.lower() in (".pdf", ".txt"))
    skipped = [p for p in files if any(x in p.name.lower() for x in args.exclude)]
    files = [p for p in files if p not in skipped]
    if not files:
        raise SystemExit(f"no .pdf or .txt files in {docs}")
    for p in skipped:
        print(f"  skipping {p.name} (excluded)")

    tok, mdl = load_embedder(device)
    print(f"embedder  {EMBED_MODEL} on {device}\n")

    chunks = []
    drop = [x.lower() for x in args.drop_phrases]
    n_dropped = 0
    for p in files:
        pages = read_doc(p)
        n_pages, n_chars = len(pages), sum(len(t) for _, t in pages)
        pages = drop_back_matter(pages)
        if not pages:
            print(f"  {p.name:44s} skipped: back-matter cut removed everything")
            continue
        kept = sum(len(t) for _, t in pages)
        n_before = len(chunks)
        for pageno, ptext in pages:
            t = clean(ptext)
            if len(t.split()) < 12:
                continue
            for c in chunk_text(t, tok, args.chunk_tokens, args.overlap):
                if any(x in c.lower() for x in drop):
                    n_dropped += 1
                    continue
                chunks.append({"source": p.name, "page": pageno, "text": c})
        cut = (f"  (back matter cut: -{100*(1-kept/max(n_chars,1)):.0f}% of text)"
               if kept < n_chars else "")
        print(f"  {p.name:44s} {n_pages:4d} pages -> "
              f"{len(chunks)-n_before:5d} chunks{cut}")

    if not chunks:
        raise SystemExit("no chunks produced -- check extraction")
    if n_dropped:
        print(f"\nfiltered  {n_dropped} chunks matching {args.drop_phrases}")

    lens = [len(tok(c["text"], add_special_tokens=False)["input_ids"])
            for c in chunks]
    over = sum(1 for n in lens if n > HARD_LIMIT)
    print(f"\nchunks    {len(chunks)}   tokens: median {int(np.median(lens))}, "
          f"max {max(lens)}")
    assert over == 0, f"{over} chunks exceed the {HARD_LIMIT}-token encoder limit"

    print("embedding ...")
    V = embed([c["text"] for c in chunks], tok, mdl, device)

    keep, kept_vecs = [], []
    for i in range(len(chunks)):
        if kept_vecs:
            sim = float(np.max(np.asarray(kept_vecs) @ V[i]))
            if sim > args.dedupe:
                continue
        keep.append(i)
        kept_vecs.append(V[i])
    dropped = len(chunks) - len(keep)
    chunks = [chunks[i] for i in keep]
    V = V[keep]
    print(f"dedupe    dropped {dropped} chunks above cosine {args.dedupe}")

    np.savez_compressed(out / "index.npz", vectors=V)
    (out / "chunks.json").write_text(json.dumps(chunks, indent=1))

    print(f"\nindex     {V.shape[0]} chunks x {V.shape[1]} dims  "
          f"({V.nbytes/1e6:.1f} MB in memory)")
    print(f"  by source:")
    for s in sorted({c['source'] for c in chunks}):
        n = sum(1 for c in chunks if c["source"] == s)
        print(f"    {s:44s} {n:5d}  ({100*n/len(chunks):4.1f}%)")
    print(f"\nwritten   {out/'index.npz'}\n          {out/'chunks.json'}")


if __name__ == "__main__":
    main()