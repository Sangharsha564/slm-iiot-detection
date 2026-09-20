#!/usr/bin/env python3
"""
Usage:
    python src/stage2_report.py             # the summary table
    python src/stage2_report.py --detail    # plus every individual check
"""

import argparse
from pathlib import Path

import pandas as pd

GROUPS = {
    "form":     ["strict_json", "schema_ok", "length_ok"],
    "faithful": ["names_top_feature", "direction_correct", "numbers_grounded",
                 "lowers_preserved", "no_false_comparison"],
    "domain":   ["no_flood_language", "mitigation_safe", "advice_grounded"],
}
PRETTY = {"strict_json": "strict JSON",
          "schema_ok": "required fields and two actions",
          "length_ok": "within 80 words",
          "names_top_feature": "names top factor",
          "direction_correct": "direction correct",
          "numbers_grounded": "numbers grounded",
          "lowers_preserved": "mitigating factor kept",
          "no_false_comparison": "no false comparison",
          "no_flood_language": "no flood language",
          "mitigation_safe": "mitigation safe",
          "advice_grounded": "slow-rate advice"}
LABEL = {"zero_shot": "zero-shot", "few_shot": "few-shot", "rag": "RAG"}
SIZE = {"qwen05": "0.5B", "llama1b": "1B", "qwen15": "1.5B"}


def rate(df, checks):
    """Share of applicable checks passed. None if none applied."""
    cols = [c for c in checks if c in df.columns]
    vals = [v for v in df[cols].values.ravel() if pd.notna(v)]
    return (100 * sum(bool(v) for v in vals) / len(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dir", default="results/08_stage2")
    ap.add_argument("--conditions", nargs="+",
                    default=["zero_shot", "few_shot", "rag"])
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    d = Path(args.root).resolve() / args.dir
    frames = []
    for c in args.conditions:
        f = d / f"{c}_faithfulness.csv"
        if f.exists():
            t = pd.read_csv(f)
            t["condition"] = c
            frames.append(t)
    if not frames:
        raise SystemExit("no condition results found")
    ev = pd.concat(frames, ignore_index=True)

    conds = [c for c in args.conditions if c in set(ev.condition)]
    models = [m for m in ["qwen05", "llama1b", "qwen15"] if m in set(ev.model)] \
        or sorted(ev.model.unique())

    print(f"\n{len(ev)} explanations   {ev.flow_key.nunique()} alerts   "
          f"{len(models)} models   {len(conds)} conditions\n")

    print(f"  {'model':9s}{'size':7s}{'condition':12s}"
          f"{'form':>7s}{'faithful':>10s}{'domain':>8s}{'copying':>9s}")
    print("  " + "-" * 62)
    rows = []
    for m in models:
        for c in conds:
            sub = ev[(ev.model == m) & (ev.condition == c)]
            if sub.empty:
                continue
            r = {g: rate(sub, ch) for g, ch in GROUPS.items()}
            cp = (sub.template_similarity.dropna()
                  if "template_similarity" in sub.columns
                  else pd.Series(dtype=float))
            cps = f"{cp.median():.2f}" if len(cp) else "-"
            print(f"  {m:9s}{SIZE.get(m, ''):7s}{LABEL.get(c, c):12s}"
                  f"{r['form']:6.0f}%{r['faithful']:9.0f}%"
                  f"{r['domain']:7.0f}%{cps:>9s}")
            rows.append({"model": m, "size": SIZE.get(m, ""), "condition": c,
                         **{k: round(v, 1) for k, v in r.items()},
                         "copying": cps})
        print()


    if "advice_grounded" in ev.columns:
        a = ev.advice_grounded.dropna()
        print(f"\n  SLOW-RATE ADVICE   {int(a.sum())} of {len(a)} explanations "
              f"recommended something that actually")
        print(f"  addresses a slow-rate attack: a request timeout, a per-source "
              f"connection cap,")
        print(f"  or a buffering proxy. Everything else was generic security "
              f"advice.")
        for c in conds:
            s = ev[ev.condition == c].advice_grounded.dropna()
            print(f"    {LABEL.get(c, c):10s} {int(s.sum())}/{len(s)}")

    pd.DataFrame(rows).to_csv(d / "summary.csv", index=False)

    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Explanation quality by model and prompting condition. "
           r"Each score is the percentage of applicable checks passed. "
           r"\emph{Copying} is the median similarity to the worked example "
           r"with numbers removed; 1.00 means its wording was reused "
           r"verbatim.}",
           r"\label{tab:stage2-summary}",
           r"\begin{tabular}{lllrrrr}", r"\toprule",
           r"Model & Size & Condition & Form & Faithful & Domain & Copying \\",
           r"\midrule"]
    for m in models:
        for c in conds:
            sub = ev[(ev.model == m) & (ev.condition == c)]
            if sub.empty:
                continue
            r = {g: rate(sub, ch) for g, ch in GROUPS.items()}
            cp = (sub.template_similarity.dropna()
                  if "template_similarity" in sub.columns
                  else pd.Series(dtype=float))
            cps = f"{cp.median():.2f}" if len(cp) else "--"
            tex.append(f"{m} & {SIZE.get(m, '')} & {LABEL.get(c, c)} & "
                       f"{r['form']:.0f}\\% & {r['faithful']:.0f}\\% & "
                       f"{r['domain']:.0f}\\% & {cps} \\\\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex += [r"\end{tabular}", r"\end{table}"]
    (d / "summary.tex").write_text("\n".join(tex) + "\n")

    if args.detail:
        w = 12
        print(f"\n\n  EVERY CHECK   passed / scored\n")
        print(f"    {'':24s}" + "".join(
            "".join(f"{m:>{w}s}" for m in models) for c in conds))
        print(f"    {'':24s}" + "".join(
            f"{LABEL.get(c, c):>{w*len(models)}s}" for c in conds))
        print("    " + "-" * (24 + w * len(models) * len(conds)))
        for grp, checks in GROUPS.items():
            for chk in checks:
                if chk not in ev.columns:
                    continue
                line = f"    {PRETTY.get(chk, chk):24s}"
                for c in conds:
                    for m in models:
                        s = ev[(ev.model == m) & (ev.condition == c)][chk].dropna()
                        line += (f"{int(s.sum())}/{len(s)}".rjust(w)
                                 if len(s) else "-".rjust(w))
                print(line)
            print()

    print(f"\n  written  {d/'summary.csv'}\n           {d/'summary.tex'}")


if __name__ == "__main__":
    main()
