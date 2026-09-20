#!/usr/bin/env python3
"""
Usage:
    python src/stage2_eval.py
    python src/stage2_eval.py --condition zero_shot --show 3
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_generate import SYSTEM, FEWSHOT

NUM = re.compile(r"\d+(?:\.\d+)?")

FACTOR = re.compile(
    r"^\s*(\d+)\.\s+(.+?)\s+=\s+([^\s|]+)(?:\s+bytes)?\s*\|\s*"
    r"normal:\s*([^\s|]+)\s*\|\s*(RAISES|LOWERS)\s+suspicion\s*\|\s*"
    r"(\d+)%", re.M)

UP = re.compile(r"\b(high|higher|larger|large|greater|above|exceed\w*|"
                r"increas\w*|longer|elevated|more than)\b", re.I)
DOWN = re.compile(r"\b(low|lower|smaller|small|less|below|fewer|shorter|"
                  r"reduc\w*|decreas\w*|absent|zero)\b", re.I)
MITIGATING = re.compile(
    r"\b(lower\w*\s+(the\s+)?suspicion|reduc\w*\s+(the\s+)?suspicion|"
    r"argues?\s+against|against\s+the|less\s+suspicious|mitigat\w*|"
    r"counter\w*|normal\s+variation|not\s+suspicious|does\s+not\s+support|"
    r"weigh\w*\s+against|contrary)\b", re.I)
INCRIMINATING = re.compile(
    r"\b(suspicious|malicious|support\w*|indicat\w*|rais\w*|confirm\w*|"
    r"consistent\s+with\s+(an?\s+)?attack|point\w*\s+to|evidence\s+of|"
    r"anomal\w*|unusual\w*|abnormal)\b", re.I)
COMPARATIVE = re.compile(
    r"\b(higher|larger|greater|lower|smaller|shorter|longer|elevated|"
    r"exceed\w*|above|below|differ\w*|more than|less than|unlike)\b", re.I)
FLOOD = re.compile(
    r"\b(flood\w*|volumetric|high[- ]volume|large[- ]volume|"
    r"traffic\s+volume|traffic\s+spike\w*|spike\w*\s+in\s+traffic|"
    r"(high|excessive|large|massive|saturat\w*|exhaust\w*|consum\w*)"
    r"\s+(\w+\s+){0,2}(bandwidth|throughput)|"
    r"bandwidth\s+(saturation|exhaustion|consumption)|"
    r"overwhelm\w*\s+.{0,20}(traffic|bandwidth|volume)|"
    r"massive\s+(traffic|number)|surge)\b", re.I)

SLOWRATE_ADVICE = re.compile(
    r"\b(timeout\w*|mod_reqtimeout|client_header_timeout|client_body_timeout|"
    r"connection\s+limit\w*|limit\s+.{0,25}connection\w*|limit_conn|"
    r"concurrent\s+connection\w*|simultaneous\s+connection\w*|"
    r"open\s+connection\w*|half[- ]open|"
    r"reverse\s+proxy|buffer\w*\s+.{0,20}request|buffering\s+proxy|"
    r"max\s*clients|worker\s+pool|connection\s+pool|"
    r"close\s+.{0,25}(idle|incomplete|stalled)\s+connection\w*|"
    r"inspect\w*\s+.{0,25}connection\w*|"
    r"how\s+many\s+connection\w*)\b", re.I)

GENERIC_ADVICE = re.compile(
    r"\b(rate[- ]limit\w*|block\w*\s+.{0,20}(ip|address|source)|"
    r"firewall\w*|intrusion\s+detection|monitor\w*|alert\s+the\s+\w+\s+team|"
    r"escalat\w*|report\s+.{0,20}(admin|team|owner)|"
    r"review\w*|investigat\w*|record\w*|log\s+the)\b", re.I)

GOOD_ADVICE = SLOWRATE_ADVICE          
NO_ACTION = re.compile(
    r"\b(no\s+action|not\s+required|no\s+mitigation|none\s+required|"
    r"no\s+further\s+action|nothing\s+to\s+do)\b", re.I)

SCOPE = {"client": ["client", "source"],
         "server": ["server", "target", "destination"],
         "total":  ["total", "overall", "combined", "either direction"]}
STAT = {"max": ["max", "maximum", "largest", "longest", "biggest", "peak"],
        "min": ["min", "minimum", "smallest", "shortest"],
        "mean": ["mean", "average"],
        "stddev": ["stddev", "standard deviation", "variation", "variability",
                   "variance", "spread", "irregular", "vary", "varied",
                   "varies", "varying", "fluctuat", "erratic"]}
QUANTITY = {"packet size": ["packet size", "packet sizes", "packet length"],
            "inter-arrival": ["inter-arrival", "interarrival", "inter arrival",
                              "gap", "delay", "interval", "spacing", "timing",
                              "between packets"],
            "bytes": ["bytes", "byte count", "volume sent", "data sent"],
            "rst packets": ["rst", "reset"],
            "psh packets": ["psh", "push"]}


def parse_feature(name):
    f = name.lower()
    return (next((s for s in SCOPE if f.startswith(s)), None),
            next((s for s in STAT if f" {s} " in f" {f} "), None),
            next((q for q in QUANTITY if q in f), None))


def _any(t, words):
    return any(w in t for w in words)


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", str(text))
            if s.strip()]


def match_factor(sentence, factors):
   
    t = sentence.lower()
    scored = []
    for f in factors:
        scope, stat, qty = parse_feature(f["feature"])
        if not qty or not _any(t, QUANTITY[qty]):
            continue                       
        sc = (2 if scope and _any(t, SCOPE[scope]) else 0) + \
             (2 if stat and _any(t, STAT[stat]) else 0)
        for other in factors:
            ostat = parse_feature(other["feature"])[1]
            if other is not f and ostat and ostat != stat and _any(t, STAT[ostat]):
                sc -= 1
        scored.append((sc, f))
    if not scored:
        return []
    best = max(sc for sc, _ in scored)
    return [f for sc, f in scored if sc == best]


def focus(text, feature, factors):
    out = []
    for s in sentences(text):
        m = match_factor(s, factors)
        if len(m) == 1 and m[0]["feature"] == feature:
            out.append(s)
    return out


def mentions(text, feature, factors):
    return any(feature in [x["feature"] for x in match_factor(s, factors)]
               for s in sentences(text))


def norm_nums(text):
    out = set()
    for m in NUM.findall(str(text)):
        try:
            out.add(f"{float(m):g}")
        except ValueError:
            pass
    return out


def skeleton(text):
    t = re.sub(r"[\d.,%]+", " ", str(text).lower())
    return " ".join(t.split())


def template_similarity(text, references):
    if not references:
        return None
    sk = skeleton(text)
    return max(difflib.SequenceMatcher(None, sk, skeleton(r)).ratio()
               for r in references)


def longest_quote(text, source):
    a = re.findall(r"[a-z0-9]+", str(text).lower())
    b = re.findall(r"[a-z0-9]+", str(source).lower())
    if not a or not b:
        return None
    m = difflib.SequenceMatcher(None, a, b, autojunk=False) \
        .find_longest_match(0, len(a), 0, len(b))
    return int(m.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dir", default="results/08_stage2")
    ap.add_argument("--condition", default="zero_shot")
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--word-limit", type=int, default=80)
    ap.add_argument("--template-max", type=float, default=0.75,
                    help="similarity to a few-shot reference above which the "
                         "output is treated as copied rather than composed")
    ap.add_argument("--quote-max", type=int, default=8,
                    help="longest run of consecutive words shared with the "
                         "retrieved text that still counts as the model's own "
                         "words. Eight is roughly a clause: shorter runs are "
                         "unavoidable when writing about the same subject")
    args = ap.parse_args()

    d = Path(args.root).resolve() / args.dir
    res = pd.read_csv(d / f"{args.condition}_explanations.csv")
    prompts = pd.read_csv(d / f"{args.condition}_prompts.csv").set_index("flow_key")

    sys_nums = norm_nums(SYSTEM)
    refs = []
    for _, reply in FEWSHOT:
        try:
            refs.append(json.loads(reply).get("explanation", ""))
        except json.JSONDecodeError:
            pass
    rows = []

    has_know = "knowledge" in prompts.columns

    for _, r in res.iterrows():
        ev = str(prompts.loc[r.flow_key, "evidence"])
        know = str(prompts.loc[r.flow_key, "knowledge"]) if has_know else ""
        know = "" if know == "nan" else know
        factors = [{"rank": int(m[0]), "feature": m[1].strip(), "value": m[2],
                    "normal": m[3], "dir": m[4], "share": int(m[5])}
                   for m in FACTOR.findall(ev)]
        names = [f["feature"] for f in factors]
        top = factors[0] if factors else None

        text = str(r.text_for_eval) if pd.notna(r.text_for_eval) else str(r.raw_output)
        mit = str(r.mitigation) if pd.notna(r.mitigation) and str(r.mitigation) else ""
        whole = f"{text} {mit}"

        try:
            strict_obj = json.loads(str(r.raw_output).strip())
            strict_json = isinstance(strict_obj, dict)
        except (json.JSONDecodeError, TypeError):
            strict_obj = None
            strict_json = False

        required_fields = {"verdict", "confidence", "explanation", "mitigation"}
        schema_ok = bool(
            strict_json
            and required_fields.issubset(strict_obj)
            and isinstance(strict_obj["verdict"], str)
            and isinstance(strict_obj["confidence"], str)
            and isinstance(strict_obj["explanation"], str)
            and isinstance(strict_obj["mitigation"], list)
            and len(strict_obj["mitigation"]) == 2
            and all(isinstance(action, str) and action.strip()
                    for action in strict_obj["mitigation"])
        )

        names_top = mentions(text, top["feature"], factors) if top else None

        dir_ok, dir_wrong = None, []
        verdicts = []
        for f in factors:
            try:
                v = float(str(f["value"]).rstrip("s"))
                b = float(str(f["normal"]).rstrip("s"))
            except ValueError:
                continue
            want = "up" if v > b else "down" if v < b else None
            if want is None:
                continue                       
            seg = " ".join(focus(text, f["feature"], factors))
            if not seg:
                continue                       
            up, dn = bool(UP.search(seg)), bool(DOWN.search(seg))
            if up == dn:
                continue                       
            ok = (want == "up") == up
            verdicts.append(ok)
            if not ok:
                dir_wrong.append(f["feature"])
        if verdicts:
            dir_ok = all(verdicts)

        allowed = norm_nums(ev) | sys_nums | {"0", "1", "2", "3", "4", "5",
                                              "100", "80"}
        ungrounded = sorted(norm_nums(text) - allowed)

        lowers = [f for f in factors if f["dir"] == "LOWERS"]
        low_ok, low_detail = None, []
        if lowers:
            verdicts = []
            for f in lowers:
                seg = " ".join(focus(text, f["feature"], factors))
                if not seg:
                    verdicts.append(None)              
                    continue
                m, i = bool(MITIGATING.search(seg)), bool(INCRIMINATING.search(seg))
                verdicts.append(True if m else (False if i else None))
                if m is False and i:
                    low_detail.append(f["feature"])
            got = [v for v in verdicts if v is not None]
            low_ok = all(got) if got else None
        same = [f for f in factors if str(f["value"]).rstrip("s") ==
                str(f["normal"]).rstrip("s")]
        false_cmp, fc_detail = None, []
        if same:
            bad = []
            for f in same:
                seg = " ".join(focus(text, f["feature"], factors))
                if seg and COMPARATIVE.search(seg):
                    bad.append(f["feature"])
            false_cmp = not bad
            fc_detail = bad
        flood = FLOOD.search(whole)
        rows.append({
            "model": r.model, "condition": r.condition, "flow_key": r.flow_key,
            "scenario": r.scenario,
            "strict_json": strict_json,
            "recoverable_output": bool(r.valid_json),
            "schema_ok": schema_ok,
            "names_top_feature": names_top,
            "direction_correct": dir_ok,
            "direction_wrong_on": ",".join(dir_wrong),
            "numbers_grounded": len(ungrounded) == 0,
            "ungrounded": ",".join(ungrounded),
            "lowers_preserved": low_ok,
            "lowers_flattened": ",".join(low_detail),
            "no_false_comparison": false_cmp,
            "false_comparison_on": ",".join(fc_detail),
            "no_flood_language": not bool(flood),
            "flood_phrase": flood.group(0) if flood else "",
            "mitigation_present": bool(mit.strip()),
            "mitigation_safe": not bool(NO_ACTION.search(mit)) if mit else None,
            "advice_grounded": bool(SLOWRATE_ADVICE.search(mit)) if mit else False,
            "advice_generic_only": (bool(GENERIC_ADVICE.search(mit))
                                    and not bool(SLOWRATE_ADVICE.search(mit)))
                                   if mit else False,
            "template_similarity": template_similarity(text, refs),
            "not_templated": (template_similarity(text, refs) or 0) < args.template_max,

           
            "retrieved_has_mitigation": (bool(SLOWRATE_ADVICE.search(know))
                                         if know else None),
            "quoted_run_words": longest_quote(whole, know) if know else None,
            "not_quoting": ((longest_quote(whole, know) or 0) < args.quote_max
                            if know else None),
            "n_words": len(text.split()),
            "length_ok": len(text.split()) <= args.word_limit,
            "gen_seconds": r.gen_seconds,
        })

    ev = pd.DataFrame(rows)
    ev.to_csv(d / f"{args.condition}_faithfulness.csv", index=False)

    CHECKS = ["strict_json", "schema_ok", "length_ok",
              "names_top_feature", "direction_correct",
              "numbers_grounded", "lowers_preserved",
              "no_false_comparison", "no_flood_language",
              "mitigation_safe", "advice_grounded"]
    if args.condition != "zero_shot":
        CHECKS.append("not_templated")
    if has_know and ev.not_quoting.notna().any():
        CHECKS.append("not_quoting")

    print(f"condition {args.condition}   {len(ev)} explanations, "
          f"{ev.model.nunique()} models, {ev.flow_key.nunique()} alerts\n")
    print(f"  {'check':22s} " +
          "".join(f"{m:>10s}" for m in sorted(ev.model.unique())) + "    n scored")
    print("  " + "-" * (23 + 10 * ev.model.nunique() + 12))
    for c in CHECKS:
        cells, scored = [], 0
        for m in sorted(ev.model.unique()):
            s = ev[ev.model == m][c].dropna()
            scored = max(scored, len(s))
            cells.append(f"{int(s.sum()):>6d}/{len(s):<3d}" if len(s) else "      -   ")
        print(f"  {c:22s} " + "".join(cells) + f"    {scored}")

    print(f"\n  counts are out of the alerts where the check applies:")
    print(f"    strict_json          raw output only; recovered JSON does not pass")
    print(f"    schema_ok            four required fields and exactly two actions")
    print(f"    length_ok            explanation contains at most {args.word_limit} words")
    print(f"    lowers_preserved     only alerts with a LOWERS factor")
    print(f"    no_false_comparison  only alerts where a value equals normal")
    print(f"    direction_correct    only where the text states a direction")
    print(f"    mitigation_safe      only where a mitigation was produced")
    gen = ev[ev.advice_generic_only]
    if len(gen):
        print(f"\n  GENERIC ADVICE ONLY  ({len(gen)} of {len(ev)}) -- security "
              f"boilerplate that would fit any alert,\n  with no slow-rate "
              f"specific measure (timeouts, connection caps, buffering proxy)")
        for m, g_ in gen.groupby("model"):
            print(f"    {m:9s} {len(g_):3d}")

    for col, label in (("direction_wrong_on", "DIRECTION STATED BACKWARDS"),
                       ("lowers_flattened", "MITIGATING FACTOR PRESENTED AS INCRIMINATING"),
                       ("false_comparison_on", "CLAIMED A DIFFERENCE WHERE VALUE == NORMAL"),
                       ("flood_phrase", "FLOOD / VOLUME LANGUAGE (wrong for Slowloris)"),
                       ("ungrounded", "NUMBERS NOT IN THE PROMPT")):
        bad = ev[ev[col].astype(str).str.len() > 0]
        if len(bad):
            print(f"\n  {label}  ({len(bad)} of {len(ev)})")
            for m, g in bad.groupby("model"):
                top = pd.Series(",".join(g[col].astype(str)).split(",")).value_counts()
                print(f"    {m:9s} {len(g):3d}   " +
                      ", ".join(f"{k}({v})" for k, v in top.head(4).items() if k))

    if args.condition != "zero_shot" and ev.template_similarity.notna().any():
        print(f"\n  TEMPLATE COPYING  similarity to the nearest worked example,")
        print(f"  numbers ignored. 1.00 = the reference sentence frame reused "
              f"verbatim.")
        for m, g_ in ev.groupby("model"):
            sim = g_.template_similarity
            print(f"    {m:9s} median {sim.median():.2f}   max {sim.max():.2f}   "
                  f"above {args.template_max:.2f}: {int((sim >= args.template_max).sum())}/{len(g_)}")
        print(f"    a high value means the check scores above reflect copying "
              f"the example,\n    not composing an explanation from the evidence.")

    if has_know and ev.retrieved_has_mitigation.notna().any():
        rh = ev.retrieved_has_mitigation.dropna()
        n_alert = ev[ev.retrieved_has_mitigation.notna()].flow_key.nunique()
        got = ev[ev.retrieved_has_mitigation.astype("boolean").fillna(False)] \
            .flow_key.nunique()
        print(f"\n  RETRIEVAL, NOT THE MODEL")
        print(f"    {got}/{n_alert} alerts retrieved text that actually "
              f"contains slow-rate advice.")
        if got < n_alert:
            miss = ev[~ev.retrieved_has_mitigation.astype("boolean").fillna(True)]
            ag = miss.advice_grounded.dropna()
            print(f"    On the {n_alert-got} where it did not, "
                  f"advice_grounded is {int(ag.sum())}/{len(ag)} -- those are "
                  f"retrieval failures,")
            print(f"    not model failures, and should not be read as RAG "
                  f"having nothing to add.")
        else:
            print(f"    Every alert had usable advice available, so "
                  f"advice_grounded measures the model.")

        q = ev.quoted_run_words.dropna()
        if len(q):
            print(f"\n  QUOTING FROM RETRIEVED TEXT  longest run of consecutive "
                  f"words taken\n  verbatim from the KNOWLEDGE block.")
            for m, g_ in ev.groupby("model"):
                s = g_.quoted_run_words.dropna()
                if not len(s):
                    continue
                print(f"    {m:9s} median {s.median():4.0f}   max {s.max():4.0f}"
                      f"   at or above {args.quote_max}: "
                      f"{int((s >= args.quote_max).sum())}/{len(s)}")
            print(f"    Using retrieved knowledge and copying it score the same "
                  f"on every other\n    check. This is the column that tells "
                  f"them apart.")

    print(f"\n  overall: all applicable checks passed")
    for m, g in ev.groupby("model"):
        ok = g[CHECKS].apply(lambda r: all(v for v in r if pd.notna(v)), axis=1)
        print(f"    {m:9s} {int(ok.sum()):2d}/{len(g)}   "
              f"{g.gen_seconds.mean():.2f}s per alert")

    if args.show:
        for m, g in res.groupby("model"):
            print(f"\n{'='*74}\n{m}\n{'='*74}")
            for _, r in g.head(args.show).iterrows():
                e = ev[(ev.model == m) & (ev.flow_key == r.flow_key)].iloc[0]
                failed = [c for c in CHECKS if e[c] is False or e[c] == 0]
                print(f"\n  {str(r.flow_key)[:14]}  {r.scenario}")
                print(f"  {str(r.text_for_eval)[:400]}")
                if str(r.mitigation).strip() and str(r.mitigation) != "nan":
                    print(f"  MITIGATION: {r.mitigation}")
                print(f"  FAILED: {', '.join(failed) if failed else 'nothing'}")

    print(f"\nwritten to {d / (args.condition + '_faithfulness.csv')}")


if __name__ == "__main__":
    main()
