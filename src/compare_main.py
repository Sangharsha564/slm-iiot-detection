#!/usr/bin/env python3
"""
Usage:
    python src/compare_main.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

VARIANTS = {
    "stage1_full_13_readable192": "readable",
    "stage1_full_13_rawnames": "raw",
}
MODEL_ORDER = ["bert-tiny", "bert-mini", "electra-small",
               "tinybert-4l", "minilm-l6"]
PALETTE = ["#2E5EAA", "#C4462E", "#3F8F5C", "#B8860B", "#6A4C93"]


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for vdir, label in VARIANTS.items():
        base = root / "experiments" / vdir
        if not base.exists():
            print(f"  warning: {vdir} not found")
            continue
        for run in sorted(base.iterdir()):
            mf, cf = run / "metrics.json", run / "run_config.json"
            if not (mf.exists() and cf.exists()):
                continue
            m = json.loads(mf.read_text())
            c = json.loads(cf.read_text())

            row = {
                "model": c["model"],
                "serialisation": label,
                "seed": c["seed"],
                "n_params": c["n_params"],
                "best_epoch": m["best_epoch"],
                "epochs_run": m["epochs_run"],
                "train_s": m["train_seconds"],
                "threshold": m["tuned_threshold"],
                "run_dir": str(run.relative_to(root)),
            }
            for key, pre in [("val_at_tuned", "val"),
                             ("test_at_tuned", "test"),
                             ("test_at_0.5", "test05")]:
                for k, v in m.get(key, {}).items():
                    if isinstance(v, (int, float)):
                        row[f"{pre}_{k}"] = v

            pf = run / "profile.json"
            if pf.exists():
                p = json.loads(pf.read_text())
                row["disk_mb"] = p["disk_mb_fp32"]
                row["rss_mb"] = p["model_rss_mb"]
                b1 = [r for r in p["latency"] if r["batch_size"] == 1]
                if b1:
                    fast = min(b1, key=lambda r: r["p50_ms"])
                    row["lat_p50"] = fast["p50_ms"]
                    row["lat_p95"] = fast["p95_ms"]
                    row["lat_p99"] = fast["p99_ms"]
                    row["lat_threads"] = fast["threads"]
                    row["fps"] = fast["flows_per_second"]

            sc = run / "per_scenario.csv"
            if sc.exists():
                for _, r in pd.read_csv(sc).iterrows():
                    if pd.notna(r.get("recall")):
                        row[f"rec_{r['scenario']}"] = r["recall"]
            rows.append(row)

    df = pd.DataFrame(rows)
    df["order"] = df.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    return df.sort_values(["serialisation", "order", "seed"])


def tex(df, path, caption, label, fmt="%.4f", note=None):
    body = df.to_latex(index=False, escape=False, float_format=fmt,
                       column_format="l" + "r" * (len(df.columns) - 1))
    tail = f"\\vspace{{2pt}}\n\\footnotesize {note}\n" if note else ""
    path.write_text("\\begin{table}[htbp]\n\\centering\n"
                    f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
                    f"{body}{tail}\\end{{table}}\n")


def pm(m, s):
    return f"{m:.4f} $\\pm$ {s:.4f}" if pd.notna(s) else f"{m:.4f}"



def t1_per_serialisation(df, D):
    g = (df.groupby(["serialisation", "model"])
         .agg(n=("seed", "size"),
              mcc_m=("test_mcc", "mean"), mcc_s=("test_mcc", "std"),
              f1_m=("test_f1", "mean"),
              prec_m=("test_precision", "mean"),
              rec_m=("test_recall", "mean"), rec_s=("test_recall", "std"),
              fn_m=("test_fn", "mean"))
         .reset_index())
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values(["serialisation", "order"]).drop(columns="order")
    g.to_csv(D["cmp"] / "per_serialisation.csv", index=False)

    wide = g.pivot(index="model", columns="serialisation")
    out = pd.DataFrame({
        "Model": MODEL_ORDER,
        "Readable": [pm(wide.loc[m, ("mcc_m", "readable")],
                        wide.loc[m, ("mcc_s", "readable")])
                     for m in MODEL_ORDER],
        "Raw names": [pm(wide.loc[m, ("mcc_m", "raw")],
                         wide.loc[m, ("mcc_s", "raw")])
                      for m in MODEL_ORDER],
    })
    tex(out, D["cmp"] / "per_serialisation.tex",
        "Test MCC on the held-out device, by model and input serialisation. "
        "Mean $\\pm$ standard deviation over three seeds.",
        "tab:per-serialisation",
        note="Both arms use the same training configuration "
             "(192 tokens, patience 4).")
    return g


def t2_combined(df, D):
    g = (df.groupby("model")
         .agg(n=("seed", "size"),
              params=("n_params", "first"),
              mcc_m=("test_mcc", "mean"), mcc_s=("test_mcc", "std"),
              mcc_min=("test_mcc", "min"), mcc_max=("test_mcc", "max"),
              f1_m=("test_f1", "mean"),
              prec_m=("test_precision", "mean"),
              rec_m=("test_recall", "mean"), rec_s=("test_recall", "std"),
              fn_m=("test_fn", "mean"),
              pr_auc_m=("test_pr_auc", "mean"))
         .reset_index())
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values("order").drop(columns="order")
    g.to_csv(D["cmp"] / "combined_ranking.csv", index=False)

    out = pd.DataFrame({
        "Model": g.model,
        "Params": (g.params / 1e6).round(1).astype(str) + "M",
        "MCC": [pm(a, b) for a, b in zip(g.mcc_m, g.mcc_s)],
        "$F_1$": g.f1_m.round(4),
        "Precision": g.prec_m.round(4),
        "Recall": [pm(a, b) for a, b in zip(g.rec_m, g.rec_s)],
        "Missed": g.fn_m.round(0).astype(int),
    })
    tex(out, D["cmp"] / "combined_ranking.tex",
        "Model comparison pooled over both serialisations (six runs per "
        "model). Missed is the mean count of undetected attack flows out of "
        "923.", "tab:combined-ranking",
        note="Precision is 1.0000 for three models; the remaining two produce "
             "one or two false positives out of 3\\,720 benign flows.")
    return g


def t3_resources(df, D, combined):
    sub = df[df.disk_mb.notna()]
    if sub.empty:
        return None
    g = (sub.groupby("model")
         .agg(params=("n_params", "first"), disk=("disk_mb", "first"),
              rss=("rss_mb", "first"), p50=("lat_p50", "first"),
              p95=("lat_p95", "first"), p99=("lat_p99", "first"),
              threads=("lat_threads", "first"), fps=("fps", "first"))
         .reset_index())
    g["mcc"] = g.model.map(dict(zip(combined.model, combined.mcc_m)))
    g["mcc_per_mparam"] = (g.mcc / (g.params / 1e6)).round(4)
    g["mcc_per_mb"] = (g.mcc / g.disk).round(4)
    g["mcc_per_ms"] = (g.mcc / g.p50).round(4)
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values("order").drop(columns="order")
    g.to_csv(D["res"] / "resource_profile.csv", index=False)

    tex(pd.DataFrame({
        "Model": g.model,
        "Disk (MB)": g.disk.round(1),
        "RSS (MB)": g.rss.round(1),
        "p50 (ms)": g.p50.round(2),
        "p95 (ms)": g.p95.round(2),
        "Flows/s": g.fps.round(0).astype(int),
    }), D["res"] / "resource_profile.tex",
        "Deployment cost measured on CPU at batch size~1, the condition under "
        "which an inline detector operates.", "tab:resources", fmt="%.2f",
        note="Measured on the rawnames seed-42 runs. These quantities depend "
             "on architecture, not on seed or serialisation. Single-threaded "
             "execution was fastest for four of five models.")

    tex(pd.DataFrame({
        "Model": g.model,
        "MCC": g.mcc.round(4),
        "MCC/Mparam": g.mcc_per_mparam,
        "MCC/MB": g.mcc_per_mb,
        "MCC/ms": g.mcc_per_ms,
    }), D["res"] / "efficiency.tex",
        "Detection quality per unit of resource. Higher is better throughout.",
        "tab:efficiency")
    return g


def t4_serialisation(df, D):
    piv = df.pivot_table(index=["model", "seed"], columns="serialisation",
                         values="test_mcc").reset_index().dropna()
    if piv.empty or not {"readable", "raw"} <= set(piv.columns):
        return None
    piv["delta"] = piv["raw"] - piv["readable"]
    piv.to_csv(D["ser"] / "paired.csv", index=False)

    note = ""
    try:
        from scipy import stats
        t, p = stats.ttest_rel(piv["raw"], piv["readable"])
        w = stats.wilcoxon(piv["raw"], piv["readable"])
        note = (f"Paired $t$-test over {len(piv)} matched runs: "
                f"$t = {t:.3f}$, $p = {p:.3f}$. "
                f"Wilcoxon signed-rank: $p = {w.pvalue:.3f}$. ")
        if p > 0.05:
            note += "No significant difference between serialisations."
    except ImportError:
        pass

    g = (piv.groupby("model")
         .agg(read_m=("readable", "mean"), read_s=("readable", "std"),
              raw_m=("raw", "mean"), raw_s=("raw", "std"),
              delta=("delta", "mean"))
         .reset_index())
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values("order").drop(columns="order")

    tex(pd.DataFrame({
        "Model": g.model,
        "Readable": [pm(a, b) for a, b in zip(g.read_m, g.read_s)],
        "Raw names": [pm(a, b) for a, b in zip(g.raw_m, g.raw_s)],
        "$\\Delta$": g.delta.round(4),
    }), D["ser"] / "serialisation.tex",
        "Serialisation ablation, paired by model and seed. $\\Delta$ is raw "
        "minus readable.", "tab:serialisation", note=note)
    return piv, g, note


def t5_scenarios(df, D):
    cols = [c for c in df.columns if c.startswith("rec_")]
    if not cols:
        return None
    g = df.groupby("model")[cols].mean().reset_index()
    g.columns = [c.replace("rec_", "") for c in g.columns]
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values("order").drop(columns="order")
    g.to_csv(D["scen"] / "per_scenario.csv", index=False)

    ren = {"model": "Model",
           "dos_wisenet-camera_554": "RTSP 554 (unseen)",
           "dos_wisenet-camera_80": "HTTP 80 DoS",
           "ddos_wisenet-camera_80": "HTTP 80 DDoS"}
    out = g.rename(columns=ren)
    order = ["Model", "HTTP 80 DoS", "HTTP 80 DDoS", "RTSP 554 (unseen)"]
    out = out[[c for c in order if c in out.columns]]
    tex(out, D["scen"] / "per_scenario.tex",
        "Recall by attack scenario on the held-out device, averaged over all "
        "six runs per model. Port 554 (RTSP) does not appear in training.",
        "tab:per-scenario",
        note="Every model detects the unseen RTSP scenario more reliably than "
             "the HTTP scenarios represented in training.")
    return g


def t6_variance(df, D):
    g = (df.groupby("model")
         .agg(n=("seed", "size"),
              val_sd=("val_mcc", "std"),
              val_rng=("val_mcc", lambda x: x.max() - x.min()),
              test_sd=("test_mcc", "std"),
              test_rng=("test_mcc", lambda x: x.max() - x.min()),
              test_min=("test_mcc", "min"), test_max=("test_mcc", "max"))
         .reset_index())
    g["ratio"] = (g.test_rng / g.val_rng.replace(0, np.nan)).round(1)
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values("order").drop(columns="order")
    g.to_csv(D["var"] / "variance_by_model.csv", index=False)

    tex(pd.DataFrame({
        "Model": g.model,
        "Val.\\ range": g.val_rng.round(4),
        "Test range": g.test_rng.round(4),
        "Test min": g.test_min.round(4),
        "Test max": g.test_max.round(4),
        "Ratio": g.ratio,
    }), D["var"] / "variance_by_model.tex",
        "Seed sensitivity by architecture over six runs each. Ratio is the "
        "test MCC range divided by the validation MCC range.",
        "tab:variance", fmt="%.4f")

    sat = df[df.val_mcc >= 0.999]
    overall = pd.DataFrame([
        {"Partition": "Validation", "Min": df.val_mcc.min(),
         "Max": df.val_mcc.max(), "Range": df.val_mcc.max() - df.val_mcc.min(),
         "SD": df.val_mcc.std()},
        {"Partition": "Test (held-out device)", "Min": df.test_mcc.min(),
         "Max": df.test_mcc.max(),
         "Range": df.test_mcc.max() - df.test_mcc.min(),
         "SD": df.test_mcc.std()},
    ])
    overall.to_csv(D["var"] / "val_vs_test.csv", index=False)
    ratio = ((df.test_mcc.max() - df.test_mcc.min())
             / (df.val_mcc.max() - df.val_mcc.min()))
    tex(overall.round(4), D["var"] / "val_vs_test.tex",
        f"Validation and test MCC across all {len(df)} runs of the two main "
        f"configurations. Test performance varies {ratio:.0f} times more "
        f"widely than validation performance.", "tab:val-vs-test",
        note=(f"{len(sat)} runs reached validation MCC $\\geq 0.999$; their "
              f"test MCC ranges {sat.test_mcc.min():.4f} to "
              f"{sat.test_mcc.max():.4f}." if len(sat) else None))
    return g, overall, sat


def style(ax, xl, yl, title=None):
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    if title:
        ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(df, piv, scen, res, fd):
    models = [m for m in MODEL_ORDER if m in set(df.model)]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    for i, ser in enumerate(["readable", "raw"]):
        sub = df[df.serialisation == ser].groupby("model").test_mcc
        means = [sub.mean().get(m, np.nan) for m in models]
        sds = [sub.std().get(m, np.nan) for m in models]
        ax.bar(x + (i - 0.5) * 0.36, means, 0.34, yerr=sds, capsize=3,
               label=ser, color=PALETTE[i], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=12, fontsize=8)
    ax.set_ylim(0.85, 0.97)
    style(ax, "", "Test MCC", "Detection quality by model and serialisation")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(fd / "F1_model_serialisation.png", dpi=200)
    plt.close(fig)

    if res is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(res.p50, res.mcc, s=res.disk * 4, alpha=0.6,
                   color=PALETTE[1], edgecolor="white", zorder=3)
        for _, r in res.iterrows():
            ax.annotate(r.model, (r.p50, r.mcc), textcoords="offset points",
                        xytext=(9, 5), fontsize=8)
        style(ax, "Median latency per flow, batch 1 (ms)", "Test MCC",
              "Marker area proportional to on-disk size")
        fig.tight_layout(); fig.savefig(fd / "F2_accuracy_vs_latency.png",
                                        dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, ser in enumerate(["readable", "raw"]):
        s = df[df.serialisation == ser]
        ax.scatter(s.val_mcc, s.test_mcc, s=52, alpha=0.8, label=ser,
                   color=PALETTE[i], edgecolor="white")
    ax.set_xlim(df.val_mcc.min() - 0.001, 1.0015)
    style(ax, "Validation MCC", "Test MCC (held-out device)",
          "Validation performance does not predict transfer")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(fd / "F3_val_vs_test.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, ser in enumerate(["readable", "raw"]):
        s = df[df.serialisation == ser]
        xs = [models.index(m) + (i - 0.5) * 0.18 for m in s.model]
        ax.scatter(xs, s.test_mcc, s=46, alpha=0.8, label=ser,
                   color=PALETTE[i], edgecolor="white")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=12, fontsize=8)
    style(ax, "", "Test MCC", "All 30 runs")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(fd / "F4_all_runs.png", dpi=200)
    plt.close(fig)

    if piv is not None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for _, r in piv.iterrows():
            ax.plot([0, 1], [r["readable"], r["raw"]], "-", color="#ccc",
                    linewidth=0.9, zorder=1)
        ax.scatter([0] * len(piv), piv["readable"], s=46, color=PALETTE[0],
                   zorder=3, edgecolor="white", label="readable")
        ax.scatter([1] * len(piv), piv["raw"], s=46, color=PALETTE[1],
                   zorder=3, edgecolor="white", label="raw names")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Readable", "Raw names"])
        ax.set_xlim(-0.4, 1.4)
        style(ax, "", "Test MCC", "Paired by model and seed")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout(); fig.savefig(fd / "F5_serialisation_paired.png",
                                        dpi=200)
        plt.close(fig)

    if scen is not None:
        cols = [c for c in scen.columns if c != "model"]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        x = np.arange(len(scen)); w = 0.8 / len(cols)
        for i, c in enumerate(cols):
            ax.bar(x + i * w - 0.4 + w / 2, scen[c], w * 0.9,
                   label=c.replace("_", " "), color=PALETTE[i],
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(scen.model, rotation=12,
                                             fontsize=8)
        ax.set_ylim(0, 1.05)
        style(ax, "", "Recall", "Recall by attack scenario")
        ax.legend(fontsize=7, frameon=False, ncol=3)
        fig.tight_layout(); fig.savefig(fd / "F6_per_scenario.png", dpi=200)
        plt.close(fig)


def main():
    root = Path(".").resolve()
    res_dir = root / "results_main"
    D = {k: res_dir / v for k, v in {
        "cmp": "01_comparison", "res": "02_resources",
        "ser": "03_serialisation", "scen": "04_scenarios",
        "var": "05_variance", "app": "06_appendix", "fig": "figures",
    }.items()}
    for d in D.values():
        d.mkdir(parents=True, exist_ok=True)

    df = collect(root)
    if df.empty:
        sys.exit("no runs found in the two main variants")

    print(f"collected {len(df)} runs")
    print(df.groupby(["serialisation", "model"]).size().to_string())
    print()

    df.to_csv(D["app"] / "all_runs.csv", index=False)
    keep = ["model", "serialisation", "seed", "best_epoch", "epochs_run",
            "threshold", "val_mcc", "test_mcc", "test_f1", "test_precision",
            "test_recall", "test_tp", "test_fp", "test_tn", "test_fn",
            "train_s"]
    df[[c for c in keep if c in df.columns]].to_csv(
        D["app"] / "runs_summary.csv", index=False)

    print("=" * 74)
    print("1  MODEL x SERIALISATION  (test MCC, mean over 3 seeds)")
    print("=" * 74)
    t1 = t1_per_serialisation(df, D)
    wide = t1.pivot(index="model", columns="serialisation",
                    values=["mcc_m", "mcc_s"])
    show = pd.DataFrame({
        "model": MODEL_ORDER,
        "readable": [wide.loc[m, ("mcc_m", "readable")] for m in MODEL_ORDER],
        "read_sd": [wide.loc[m, ("mcc_s", "readable")] for m in MODEL_ORDER],
        "raw": [wide.loc[m, ("mcc_m", "raw")] for m in MODEL_ORDER],
        "raw_sd": [wide.loc[m, ("mcc_s", "raw")] for m in MODEL_ORDER],
    })
    show["delta"] = show["raw"] - show["readable"]
    print(show.round(4).to_string(index=False))

    print("\n" + "=" * 74)
    print("2  COMBINED RANKING  (6 runs per model)")
    print("=" * 74)
    t2 = t2_combined(df, D)
    print(t2[["model", "n", "params", "mcc_m", "mcc_s", "mcc_min",
              "mcc_max", "rec_m", "fn_m"]].round(4).to_string(index=False))

    print("\n" + "=" * 74)
    print("3  RESOURCES  (batch 1, best thread count)")
    print("=" * 74)
    t3 = t3_resources(df, D, t2)
    if t3 is not None:
        print(t3[["model", "disk", "rss", "p50", "p95", "threads", "fps",
                  "mcc", "mcc_per_mparam", "mcc_per_mb", "mcc_per_ms"]]
              .round(3).to_string(index=False))

    print("\n" + "=" * 74)
    print("4  SERIALISATION ABLATION")
    print("=" * 74)
    ser = t4_serialisation(df, D)
    piv = None
    if ser:
        piv, g4, note = ser
        print(g4.round(4).to_string(index=False))
        print(f"\n  mean delta = {piv.delta.mean():+.4f}  "
              f"(sd {piv.delta.std():.4f}, n = {len(piv)})")
        print("  " + note.replace("$", "").replace("\\", ""))

    print("\n" + "=" * 74)
    print("5  PER-SCENARIO RECALL")
    print("=" * 74)
    scen = t5_scenarios(df, D)
    if scen is not None:
        print(scen.round(4).to_string(index=False))

    print("\n" + "=" * 74)
    print("6  SEED SENSITIVITY AND VALIDATION SELECTION")
    print("=" * 74)
    g6, overall, sat = t6_variance(df, D)
    print(g6.round(4).to_string(index=False))
    print()
    print(overall.round(4).to_string(index=False))
    if len(sat):
        print(f"\n  {len(sat)} of {len(df)} runs reached validation MCC "
              f">= 0.999")
        print(f"  their test MCC ranges {sat.test_mcc.min():.4f} to "
              f"{sat.test_mcc.max():.4f} "
              f"(spread {sat.test_mcc.max() - sat.test_mcc.min():.4f})")

    if HAVE_PLT:
        figures(df, piv, scen, t3, D["fig"])
        print(f"\nfigures -> {D['fig']}")
    print(f"tables  -> {res_dir}")


if __name__ == "__main__":
    main()