#!/usr/bin/env python3
"""

Usage:
    python src/reparse_log.py --log results/10_pipeline/pipeline_log.csv
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_generate import fields, parse_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/10_pipeline/pipeline_log.csv")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the log (a .bak copy is kept). Default: "
                         "write alongside it with a _reparsed suffix")
    args = ap.parse_args()

    p = Path(args.log)
    t = pd.read_csv(p)
    if "raw_output" not in t.columns:
        raise SystemExit(f"{p} has no raw_output column: {list(t.columns)}")

    before = dict(t.json_status.value_counts()) if "json_status" in t else {}

    rows = []
    for _, r in t.iterrows():
        obj, how = parse_json(str(r.raw_output))
        f = fields(obj)
        rows.append({"json_status": how, "valid_json": obj is not None,
                
                     "json_strict": how in ("clean", "fenced"),
                     **f,
                     "text_for_eval": f["explanation"] or str(r.raw_output)})
    new = pd.DataFrame(rows, index=t.index)
    for c in new.columns:
        t[c] = new[c]

    out = p if args.in_place else p.with_name(p.stem + "_reparsed" + p.suffix)
    if args.in_place:
        shutil.copy(p, p.with_suffix(p.suffix + ".bak"))
    t.to_csv(out, index=False)

    print(f"  {len(t)} outputs re-parsed\n")
    if before:
        print(f"  before  {before}")
    print(f"  after   {dict(t.json_status.value_counts())}\n")
    print(f"  parsed as JSON by the model   "
          f"{int(t.json_strict.sum())}/{len(t)}  "
          f"({100*t.json_strict.mean():.0f}%)  <- instruction-following")
    print(f"  content recovered             "
          f"{int(t.valid_json.sum())}/{len(t)}  "
          f"({100*t.valid_json.mean():.0f}%)  <- pipeline robustness")
    miss = t[~t.valid_json]
    if len(miss):
        print(f"\n  still unrecoverable: {len(miss)}")
        for s in miss.raw_output.head(2):
            print(f"    {str(s)[:160]}")
    n_mit = int((t.mitigation.fillna("").astype(str).str.len() > 0).sum())
    print(f"\n  mitigation present            {n_mit}/{len(t)}")
    print(f"\n  written {out}")


if __name__ == "__main__":
    main()
