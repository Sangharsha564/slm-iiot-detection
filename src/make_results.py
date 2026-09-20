#!/usr/bin/env python3
"""
Usage:
    python src/make_results.py
    python src/make_results.py --headline full_13_rawnames
"""

import argparse
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

PALETTE = ["#2E5EAA", "#C4462E", "#3F8F5C", "#B8860B", "#6A4C93"]
MODEL_ORDER = ["bert-tiny", "bert-mini", "electra-small",
               "tinybert-4l", "minilm-l6"]


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for mfile in sorted(root.glob("experiments/*/*/metrics.json")):
        d = mfile.parent
        try:
            m = json.loads(mfile.read_text())
            cfg = json.loads((d / "run_config.json").read_text())
        except Exception:                                     
            continue

        row = {
            "model": cfg["model"],
            "variant": d.parent.name.replace("stage1_", ""),
            "seed": cfg["seed"],
            "n_features": cfg["n_features"],
            "n_params": cfg["n_params"],
            "best_epoch": m["best_epoch"],
            "epochs_run": m["epochs_run"],
            "train_seconds": m["train_seconds"],
            "threshold": m["tuned_threshold"],
            "run_dir": str(d.relative_to(root)),
        }
        for split, prefix in [("val_at_tuned", "val"),
                              ("test_at_tuned", "test"),
                              ("test_at_0.5", "test05")]:
            for k, v in m.get(split, {}).items():
                if isinstance(v, (int, float)):
                    row[f"{prefix}_{k}"] = v

        pf = d / "profile.json"
        if pf.exists():
            p = json.loads(pf.read_text())
            row["disk_mb"] = p.get("disk_mb_fp32")
            row["model_rss_mb"] = p.get("model_rss_mb")
            row["cold_start_s"] = p.get("cold_start_total_s")
            b1 = [r for r in p.get("latency", [])
                  if r["batch_size"] == 1 and r.get("precision", "fp32") == "fp32"]
            if b1:
                fastest = min(b1, key=lambda r: r["p50_ms"])
                row["latency_p50_ms"] = fastest["p50_ms"]
                row["latency_p95_ms"] = fastest["p95_ms"]
                row["latency_threads"] = fastest["threads"]
                row["flows_per_sec"] = fastest["flows_per_second"]

        ps = d / "per_scenario.csv"
        if ps.exists():
            sc = pd.read_csv(ps)
            for _, r in sc.iterrows():
                if pd.notna(r.get("recall")):
                    row[f"recall_{r['scenario']}"] = r["recall"]

        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(df, path, caption, label, fmt="%.4f", note=None):
    body = df.to_latex(index=False, escape=True, float_format=fmt,
                       column_format="l" + "r" * (len(df.columns) - 1))
    notes = f"\n\\vspace{{2pt}}\n\\footnotesize {note}\n" if note else ""
    path.write_text(
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
        f"{body}{notes}\\end{{table}}\n")


def msd(mean, sd):
    if pd.isna(sd):
        return f"{mean:.4f}"
    return f"{mean:.4f} $\\pm$ {sd:.4f}"

def table_model_comparison(runs, D, headline):
    sub = runs[runs.variant == headline]
    if sub.empty:
        return None
    g = (sub.groupby("model")
         .agg(n_params=("n_params", "first"),
              n_seeds=("seed", "size"),
              mcc_mean=("test_mcc", "mean"), mcc_sd=("test_mcc", "std"),
              f1_mean=("test_f1", "mean"), f1_sd=("test_f1", "std"),
              prec_mean=("test_precision", "mean"),
              rec_mean=("test_recall", "mean"), rec_sd=("test_recall", "std"),
              pr_auc_mean=("test_pr_auc", "mean"),
              fp_mean=("test_fp", "mean"))
         .reset_index())
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values("order").drop(columns="order")
    g.to_csv(D["models"] / "model_comparison.csv", index=False)
    tex = pd.DataFrame({
        "Model": g.model,
        "MCC": [msd(a, b) for a, b in zip(g.mcc_mean, g.mcc_sd)],
        "$F_1$": [msd(a, b) for a, b in zip(g.f1_mean, g.f1_sd)],
        "Precision": g.prec_mean.round(4),
        "Recall": [msd(a, b) for a, b in zip(g.rec_mean, g.rec_sd)],
        "PR-AUC": g.pr_auc_mean.round(4),
    })
    to_latex(tex, D["models"] / "model_comparison.tex",
             f"Stage~1 model comparison on the held-out device "
             f"({headline.replace('_', ' ')}). Mean $\\pm$ standard deviation "
             f"over three seeds.", "tab:model-comparison", fmt="%.4f",
             note="Precision was 1.0000 in all but two runs; false positives "
                  "were zero or near-zero throughout.")
    return g


def table_all_runs(runs, D):
    cols = ["model", "variant", "seed", "n_params", "best_epoch", "epochs_run",
            "threshold", "val_mcc", "test_mcc", "test_f1", "test_precision",
            "test_recall", "test_pr_auc", "test_tp", "test_fp", "test_tn",
            "test_fn", "train_seconds"]
    cols = [c for c in cols if c in runs.columns]
    out = runs[cols].sort_values(["variant", "model", "seed"])
    out.to_csv(D["appendix"] / "all_runs.csv", index=False)
    for v, grp in out.groupby("variant"):
        grp.drop(columns="variant").to_csv(
            D["appendix"] / f"runs_{v}.csv", index=False)
    return out


def table_resources(runs, D, headline):
    sub = runs[(runs.variant == headline) & runs.disk_mb.notna()]
    if sub.empty:
        sub = runs[runs.disk_mb.notna()]
    if sub.empty:
        return None
    g = (sub.groupby("model")
         .agg(n_params=("n_params", "first"),
              disk_mb=("disk_mb", "first"),
              rss_mb=("model_rss_mb", "first"),
              cold_s=("cold_start_s", "first"),
              p50=("latency_p50_ms", "first"),
              p95=("latency_p95_ms", "first"),
              threads=("latency_threads", "first"),
              fps=("flows_per_sec", "first"))
         .reset_index())
    acc = runs[runs.variant == headline].groupby("model")["test_mcc"].mean()
    g["mcc"] = g.model.map(acc)
    g["mcc_per_mb"] = (g.mcc / g.disk_mb * 100).round(3)
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values("order").drop(columns="order")
    g.to_csv(D["resource"] / "resource_profile.csv", index=False)

    tex = pd.DataFrame({
        "Model": g.model,
        "Params": (g.n_params / 1e6).round(1).astype(str) + "M",
        "Disk (MB)": g.disk_mb.round(1),
        "RSS (MB)": g.rss_mb.round(0).astype(int),
        "Cold start (s)": g.cold_s.round(2),
        "p50 (ms)": g.p50.round(2),
        "p95 (ms)": g.p95.round(2),
        "Flows/s": g.fps.round(0).astype(int),
        "MCC": g.mcc.round(4),
    })
    to_latex(tex, D["resource"] / "resource_profile.tex",
             "Resource profile measured on CPU (development hardware). "
             "Latency is per flow at batch size~1, the condition under which "
             "an inline detector operates.", "tab:resource", fmt="%.2f",
             note="Thread count selected per model as the fastest of 1 and 4; "
                  "single-threaded execution was fastest for four of five "
                  "models at batch size~1.")
    return g


def table_serialisation(runs, D):
    a, b = "full_13_readable192", "full_13_rawnames"
    if not {a, b} <= set(runs.variant):
        return None
    piv = runs[runs.variant.isin([a, b])].pivot_table(
        index=["model", "seed"], columns="variant",
        values="test_mcc").reset_index().dropna()
    if piv.empty:
        return None
    piv["delta"] = piv[b] - piv[a]
    piv.to_csv(D["ablation"] / "serialisation_paired.csv", index=False)

    summ = (piv.groupby("model")
            .agg(readable_mean=(a, "mean"), readable_sd=(a, "std"),
                 raw_mean=(b, "mean"), raw_sd=(b, "std"),
                 delta_mean=("delta", "mean"))
            .reset_index())
    summ["order"] = summ.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    summ = summ.sort_values("order").drop(columns="order")

    stat_note = ""
    try:
        from scipy import stats
        t, p = stats.ttest_rel(piv[b], piv[a])
        stat_note = (f"Paired $t$-test over {len(piv)} matched runs: "
                     f"$t = {t:.3f}$, $p = {p:.3f}$. ")
        if p > 0.05:
            stat_note += "The difference is not statistically significant."
    except ImportError:
        pass

    tex = pd.DataFrame({
        "Model": summ.model,
        "Readable": [msd(x, y) for x, y in
                     zip(summ.readable_mean, summ.readable_sd)],
        "Raw names": [msd(x, y) for x, y in
                      zip(summ.raw_mean, summ.raw_sd)],
        "$\\Delta$": summ.delta_mean.round(4),
    })
    to_latex(tex, D["ablation"] / "serialisation.tex",
             "Serialisation ablation: test MCC under readable feature labels "
             "versus raw column names, paired by model and seed.",
             "tab:serialisation", note=stat_note)
    return piv, summ, stat_note


def table_feature_ablation(runs, D):
    variants = ["full_13", "reduced_9", "no_fingerprint",
                "no_fingerprint_reduced"]
    sub = runs[runs.variant.isin(variants)]
    if sub.empty:
        return None
    g = (sub.groupby(["variant", "model"])
         .agg(n_features=("n_features", "first"),
              n_seeds=("seed", "size"),
              mcc=("test_mcc", "mean"),
              recall=("test_recall", "mean"),
              fn=("test_fn", "mean"))
         .reset_index()
         .sort_values(["model", "variant"]))
    g.to_csv(D["ablation"] / "feature_sets.csv", index=False)

    tex = g.rename(columns={"variant": "Feature set", "model": "Model",
                            "n_features": "$n$", "mcc": "MCC",
                            "recall": "Recall", "fn": "Missed"})
    tex["Feature set"] = tex["Feature set"].str.replace("_", " ")
    to_latex(tex.drop(columns="n_seeds"),
             D["ablation"] / "feature_sets.tex",
             "Feature-set ablation. Removing either the four lowest-mutual-"
             "information features or the highest-scoring feature degraded "
             "transfer for most models.", "tab:feature-ablation")
    return g


def table_scenarios(runs, D, headline):
    cols = [c for c in runs.columns if c.startswith("recall_")]
    if not cols:
        return None
    sub = runs[runs.variant == headline]
    if sub.empty:
        sub = runs
    g = sub.groupby("model")[cols].mean().reset_index()
    g.columns = [c.replace("recall_", "") for c in g.columns]
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values("order").drop(columns="order")
    g.to_csv(D["scenario"] / "per_scenario_recall.csv", index=False)

    tex = g.rename(columns={"model": "Model"})
    tex.columns = [c.replace("_", " ").replace("wisenet-camera", "wisenet")
                   for c in tex.columns]
    to_latex(tex, D["scenario"] / "per_scenario.tex",
             "Per-scenario recall on the held-out device. The RTSP port (554) "
             "was absent from training, yet is detected more reliably than the "
             "HTTP ports present in training.", "tab:per-scenario")
    return g


def table_val_vs_test(runs, D):
    g = (runs.groupby("variant")
         .agg(n=("seed", "size"),
              val_min=("val_mcc", "min"), val_max=("val_mcc", "max"),
              test_min=("test_mcc", "min"), test_max=("test_mcc", "max"))
         .reset_index())
    g["val_range"] = (g.val_max - g.val_min).round(4)
    g["test_range"] = (g.test_max - g.test_min).round(4)
    g["ratio"] = (g.test_range / g.val_range.replace(0, np.nan)).round(1)
    g.to_csv(D["variance"] / "val_vs_test_by_variant.csv", index=False)

    overall = pd.DataFrame([{
        "Partition": "Validation",
        "Min": runs.val_mcc.min(), "Max": runs.val_mcc.max(),
        "Range": runs.val_mcc.max() - runs.val_mcc.min(),
        "SD": runs.val_mcc.std(),
    }, {
        "Partition": "Test (held-out device)",
        "Min": runs.test_mcc.min(), "Max": runs.test_mcc.max(),
        "Range": runs.test_mcc.max() - runs.test_mcc.min(),
        "SD": runs.test_mcc.std(),
    }])
    ratio = ((runs.test_mcc.max() - runs.test_mcc.min())
             / (runs.val_mcc.max() - runs.val_mcc.min()))
    to_latex(overall.round(4), D["variance"] / "val_vs_test.tex",
             f"Validation and test MCC across all {len(runs)} runs. Test "
             f"performance varies {ratio:.0f} times more widely than "
             f"validation performance.", "tab:val-vs-test",
             note="Runs achieving validation MCC $\\geq 0.999$ produced test "
                  "MCC between "
                  f"{runs[runs.val_mcc >= 0.999].test_mcc.min():.4f} and "
                  f"{runs[runs.val_mcc >= 0.999].test_mcc.max():.4f}.")
    return g, overall


def table_variance_by_model(runs, D, headline):
    sub = runs[runs.variant.isin(["full_13_rawnames", "full_13_readable192"])]
    if sub.empty:
        return None
    g = (sub.groupby("model")
         .agg(n=("seed", "size"),
              val_sd=("val_mcc", "std"),
              val_range=("val_mcc", lambda x: x.max() - x.min()),
              test_sd=("test_mcc", "std"),
              test_range=("test_mcc", lambda x: x.max() - x.min()),
              test_min=("test_mcc", "min"),
              test_max=("test_mcc", "max"))
         .reset_index())
    g["ratio"] = (g.test_range / g.val_range.replace(0, np.nan)).round(1)
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values("order").drop(columns="order")
    g.to_csv(D["variance"] / "variance_by_model.csv", index=False)

    tex = pd.DataFrame({
        "Model": g.model,
        "Runs": g.n,
        "Val.\\ SD": g.val_sd.round(4),
        "Test SD": g.test_sd.round(4),
        "Test min": g.test_min.round(4),
        "Test max": g.test_max.round(4),
        "Range ratio": g.ratio,
    })
    to_latex(tex, D["variance"] / "variance_by_model.tex",
             "Seed sensitivity by architecture, across both serialisations "
             "(six runs each). The range ratio is the test MCC range divided "
             "by the validation MCC range.", "tab:variance-by-model",
             fmt="%.4f")
    return g


def table_efficiency(runs, D, headline):
    sub = runs[(runs.variant == headline) & runs.latency_p50_ms.notna()]
    if sub.empty:
        return None
    acc = (runs[runs.variant == headline]
           .groupby("model")["test_mcc"].mean())
    g = (sub.groupby("model")
         .agg(params=("n_params", "first"),
              disk=("disk_mb", "first"),
              rss=("model_rss_mb", "first"),
              lat=("latency_p50_ms", "first"))
         .reset_index())
    g["mcc"] = g.model.map(acc)
    g["mcc_per_mparam"] = (g.mcc / (g.params / 1e6)).round(4)
    g["mcc_per_mb_disk"] = (g.mcc / g.disk).round(4)
    g["mcc_per_ms"] = (g.mcc / g.lat).round(4)
    g["mcc_per_mb_rss"] = (g.mcc / g.rss).round(4)
    g["order"] = g.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    g = g.sort_values("order").drop(columns="order")
    g.to_csv(D["resource"] / "efficiency.csv", index=False)
    tex = pd.DataFrame({
        "Model": g.model,
        "MCC": g.mcc.round(4),
        "MCC / Mparam": g.mcc_per_mparam,
        "MCC / MB disk": g.mcc_per_mb_disk,
        "MCC / MB RSS": g.mcc_per_mb_rss,
        "MCC / ms": g.mcc_per_ms,
    })
    to_latex(tex, D["resource"] / "efficiency.tex",
             "Accuracy per unit of resource. Higher is better in every column.",
             "tab:efficiency")
    return g

def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_mcc_vs_params(runs, fig_dir, headline):
    sub = runs[runs.variant == headline]
    if sub.empty:
        return
    g = sub.groupby("model").agg(p=("n_params", "first"),
                                 m=("test_mcc", "mean"),
                                 s=("test_mcc", "std")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(g.p / 1e6, g.m, yerr=g.s, fmt="o", capsize=4,
                color=PALETTE[0], markersize=7, linewidth=1.2)
    for _, r in g.iterrows():
        ax.annotate(r.model, (r.p / 1e6, r.m), textcoords="offset points",
                    xytext=(8, 4), fontsize=8)
    style(ax, "Parameters (millions)", "Test MCC",
          "Accuracy does not increase with model size")
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_mcc_vs_params.png", dpi=200)
    plt.close(fig)


def fig_mcc_vs_latency(runs, fig_dir, headline):
    sub = runs[(runs.variant == headline) & runs.latency_p50_ms.notna()]
    if sub.empty:
        return
    g = sub.groupby("model").agg(lat=("latency_p50_ms", "first"),
                                 m=("test_mcc", "mean"),
                                 s=("test_mcc", "std"),
                                 disk=("disk_mb", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(g.lat, g.m, yerr=g.s, fmt="none", ecolor="#999", capsize=3)
    ax.scatter(g.lat, g.m, s=g.disk * 3, alpha=0.65, color=PALETTE[1],
               edgecolor="white", zorder=3)
    for _, r in g.iterrows():
        ax.annotate(r.model, (r.lat, r.m), textcoords="offset points",
                    xytext=(9, 5), fontsize=8)
    style(ax, "Median latency per flow, batch size 1 (ms)", "Test MCC",
          "Marker area is proportional to on-disk size")
    fig.tight_layout()
    fig.savefig(fig_dir / "F2_mcc_vs_latency.png", dpi=200)
    plt.close(fig)


def fig_val_vs_test(runs, fig_dir):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (v, grp) in enumerate(runs.groupby("variant")):
        ax.scatter(grp.val_mcc, grp.test_mcc, s=48, alpha=0.8,
                   label=v.replace("_", " "),
                   color=PALETTE[i % len(PALETTE)], edgecolor="white")
    lo = min(runs.val_mcc.min(), runs.test_mcc.min()) - 0.01
    ax.plot([lo, 1.001], [lo, 1.001], "--", color="#bbb", linewidth=1,
            label="parity")
    ax.set_xlim(runs.val_mcc.min() - 0.002, 1.002)
    style(ax, "Validation MCC", "Test MCC (held-out device)",
          "Validation performance does not predict transfer")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "F3_val_vs_test.png", dpi=200)
    plt.close(fig)


def fig_seed_strip(runs, fig_dir):
    variants = sorted(runs.variant.unique())
    models = [m for m in MODEL_ORDER if m in set(runs.model)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for vi, v in enumerate(variants):
        sub = runs[runs.variant == v]
        xs, ys = [], []
        for mi, m in enumerate(models):
            vals = sub[sub.model == m].test_mcc.values
            offset = (vi - (len(variants) - 1) / 2) * 0.16
            xs.extend([mi + offset] * len(vals))
            ys.extend(vals)
        ax.scatter(xs, ys, s=42, alpha=0.75, label=v.replace("_", " "),
                   color=PALETTE[vi % len(PALETTE)], edgecolor="white")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, fontsize=8)
    style(ax, "", "Test MCC", "Every run, by model and configuration")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "F4_seed_strip.png", dpi=200)
    plt.close(fig)


def fig_serialisation(piv, fig_dir):
    a, b = "full_13_readable192", "full_13_rawnames"
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for _, r in piv.iterrows():
        ax.plot([0, 1], [r[a], r[b]], "-", color="#ccc", linewidth=0.9,
                zorder=1)
    ax.scatter([0] * len(piv), piv[a], s=44, color=PALETTE[0],
               label="readable", zorder=3, edgecolor="white")
    ax.scatter([1] * len(piv), piv[b], s=44, color=PALETTE[1],
               label="raw names", zorder=3, edgecolor="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Readable labels", "Raw column names"], fontsize=9)
    ax.set_xlim(-0.4, 1.4)
    style(ax, "", "Test MCC", "Paired by model and seed")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "F5_serialisation.png", dpi=200)
    plt.close(fig)


def fig_scenarios(scen, fig_dir):
    cols = [c for c in scen.columns if c != "model"]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.8 / len(cols)
    x = np.arange(len(scen))
    for i, c in enumerate(cols):
        ax.bar(x + i * w - 0.4 + w / 2, scen[c], width=w * 0.92,
               label=c.replace("_", " "), color=PALETTE[i % len(PALETTE)],
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scen.model, rotation=15, fontsize=8)
    ax.set_ylim(0, 1.05)
    style(ax, "", "Recall", "Per-scenario recall on the held-out device")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / "F6_per_scenario.png", dpi=200)
    plt.close(fig)


def fig_training_curves(root, runs, fig_dir, headline):
    sub = runs[runs.variant == headline]
    if sub.empty:
        return
    best = sub.loc[sub.groupby("model").test_mcc.idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (_, r) in enumerate(best.iterrows()):
        h = root / r.run_dir / "history.csv"
        if not h.exists():
            continue
        df = pd.read_csv(h)
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(df.epoch, df.val_loss, marker="o", ms=3.5,
                     label=r.model, color=c, linewidth=1.4)
        axes[1].plot(df.epoch, df.val_mcc, marker="o", ms=3.5,
                     label=r.model, color=c, linewidth=1.4)
    style(axes[0], "Epoch", "Validation loss")
    style(axes[1], "Epoch", "Validation MCC")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "F7_training_curves.png", dpi=200)
    plt.close(fig)


def fig_confusion(runs, fig_dir, headline):
    sub = runs[runs.variant == headline]
    if sub.empty:
        return
    r = sub.loc[sub.test_mcc.idxmax()]
    cm = np.array([[r.test_tn, r.test_fp], [r.test_fn, r.test_tp]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues", alpha=0.85)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center",
                    fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "#222")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Attack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{r.model}, seed {int(r.seed)}  "
                 f"(MCC {r.test_mcc:.4f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "F8_confusion_matrix.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--headline", default=None,
                    help="variant used for the headline tables and figures")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    res = root / "results"
    for sub in ["01_model_comparison", "02_resources", "03_ablations",
                "04_scenarios", "05_variance", "06_appendix", "figures"]:
        (res / sub).mkdir(parents=True, exist_ok=True)
    D = {
        "models":   res / "01_model_comparison",
        "resource": res / "02_resources",
        "ablation": res / "03_ablations",
        "scenario": res / "04_scenarios",
        "variance": res / "05_variance",
        "appendix": res / "06_appendix",
        "figures":  res / "figures",
    }

    runs = collect(root)
    if runs.empty:
        sys.exit("no completed runs found under experiments/")

    variants = sorted(runs.variant.unique())
    headline = args.headline or ("full_13_rawnames"
                                 if "full_13_rawnames" in variants
                                 else variants[0])

    print(f"collected {len(runs)} runs across {len(variants)} variants")
    print(f"variants: {', '.join(variants)}")
    print(f"headline variant: {headline}\n")

    runs.to_csv(D["appendix"] / "all_runs_raw.csv", index=False)

    print("=" * 72)
    print("T1  MODEL COMPARISON")
    print("=" * 72)
    t1 = table_model_comparison(runs, D, headline)
    if t1 is not None:
        show = t1[["model", "n_params", "n_seeds", "mcc_mean", "mcc_sd",
                   "rec_mean", "prec_mean"]].round(4)
        print(show.to_string(index=False))

    table_all_runs(runs, D)

    print("\n" + "=" * 72)
    print("T3  RESOURCE PROFILE")
    print("=" * 72)
    t3 = table_resources(runs, D, headline)
    if t3 is not None:
        print(t3.drop(columns=["mcc_per_mb"]).round(3).to_string(index=False))
    else:
        print("  no profile.json found - run profile_model.py first")

    print("\n" + "=" * 72)
    print("T4  SERIALISATION ABLATION")
    print("=" * 72)
    ser = table_serialisation(runs, D)
    if ser:
        piv, summ, note = ser
        print(summ.round(4).to_string(index=False))
        print(f"\n  mean delta = {piv.delta.mean():+.4f} "
              f"(sd {piv.delta.std():.4f}, n = {len(piv)})")
        if note:
            print("  " + note.replace("$", "").replace("\\", ""))
    else:
        print("  both serialisation variants not available")

    print("\n" + "=" * 72)
    print("T5  FEATURE-SET ABLATION")
    print("=" * 72)
    t5 = table_feature_ablation(runs, D)
    if t5 is not None:
        print(t5.round(4).to_string(index=False))

    print("\n" + "=" * 72)
    print("T6  PER-SCENARIO RECALL")
    print("=" * 72)
    scen = table_scenarios(runs, D, headline)
    if scen is not None:
        print(scen.round(4).to_string(index=False))

    print("\n" + "=" * 72)
    print("T7  VALIDATION VERSUS TEST")
    print("=" * 72)
    t7, overall = table_val_vs_test(runs, D)
    print(overall.round(4).to_string(index=False))
    sat = runs[runs.val_mcc >= 0.999]
    if len(sat):
        print(f"\n  {len(sat)} runs reached validation MCC >= 0.999;")
        print(f"  their test MCC ranges {sat.test_mcc.min():.4f} to "
              f"{sat.test_mcc.max():.4f}")

    print("\n" + "=" * 72)
    print("T7b SEED SENSITIVITY BY MODEL")
    print("=" * 72)
    t7b = table_variance_by_model(runs, D, headline)
    if t7b is not None:
        print(t7b.round(4).to_string(index=False))

    print("\n" + "=" * 72)
    print("T8  EFFICIENCY")
    print("=" * 72)
    t8 = table_efficiency(runs, D, headline)
    if t8 is not None:
        print(t8[["model", "mcc", "mcc_per_mparam", "mcc_per_mb_disk",
                  "mcc_per_ms"]].to_string(index=False))

    if HAVE_PLT:
        fd = D["figures"]
        fig_mcc_vs_params(runs, fd, headline)
        fig_mcc_vs_latency(runs, fd, headline)
        fig_val_vs_test(runs, fd)
        fig_seed_strip(runs, fd)
        if ser:
            fig_serialisation(ser[0], fd)
        if scen is not None:
            fig_scenarios(scen, fd)
        fig_training_curves(root, runs, fd, headline)
        fig_confusion(runs, fd, headline)
        print(f"\nfigures written to {fd}")
    else:
        print("\nmatplotlib not available - figures skipped")

    print("\ntables written to results/, grouped by topic:")
    for k, v in D.items():
        if k == "figures":
            continue
        files = sorted(f.name for f in v.glob("*"))
        if files:
            print(f"  {v.name}/")
            for f in files:
                print(f"      {f}")


if __name__ == "__main__":
    main()