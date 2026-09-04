#!/usr/bin/env python
"""
Reproduce Yuan et al. 2024 Fig. 2 (biotech-wise boxplots by method) with our
sheaf method added and highlighted, for EVERY metric present in their source
data (figure2.feather). Method labels use the paper's published names.

Matches the authors' notebook: horizontal boxplot, one box per method over all
(slice x replicate) points, ordered so the best method is at the top (direction
per metric). Our box is built from the seed-sensitivity per-run values; it is
shown on every metric for which those values exist (NMI, and — after an
all-metric seed run — HOM/COM/CHAOS/PAS/ASW). Metrics with no "ours" value
(Moran's I, Geary's C, time, memory) are drawn baselines-only.

Usage
-----
    python benchmark_fig2d.py --feather "figure2b&2d/figure2.feather" \
        --our-per-run output_seed_sensitivity/seed_sensitivity_per_run.csv \
        --biotech MERFISH --metrics all --output-dir output_benchmark
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# feather internal key -> published display name (Yuan et al. 2024, Fig. 2)
METHOD_RENAME = {
    "louvain": "Louvain", "leiden": "Leiden",
    "SpaGCN_without": "SpaGCN", "SpaGCN_with": "SpaGCN(HE)",
    "conST_nopre": "conST", "DeepST": "GraphST",
}

# metric -> ('asc' = higher is better, 'des' = lower is better)  [from their notebook]
METRIC_DIR = {
    "NMI": "asc", "HOM": "asc", "COM": "asc",
    "CHAOS": "des", "PAS": "des", "ASW": "asc",
    "MoranI": "asc", "GearyC": "des", "time": "des", "memory": "des",
}
PRETTY = {"MoranI": "Moran's I", "GearyC": "Geary's C",
          "time": "Time (s)", "memory": "Memory (MiB)"}
# metrics that are not cross-comparable (hardware-dependent): show for
# reference, but no direction hint and no ranking.
REFERENCE_ONLY = {"time", "memory"}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feather", default="figure2b&2d/figure2.feather")
    p.add_argument("--our-per-run",
                   default="output_seed_sensitivity/seed_sensitivity_per_run.csv")
    p.add_argument("--biotech", default="MERFISH")
    p.add_argument("--metrics", default="all",
                   help="'all' or comma list, e.g. NMI,HOM,COM")
    p.add_argument("--our-name", default="Sheaf (ours)")
    p.add_argument("--gt-calibration", default=None,
                   help="gt_calibration.csv to overlay a ground-truth "
                        "reference line on the Moran's I / Geary's C panels.")
    p.add_argument("--output-dir", default="output_benchmark")
    return p


def one_panel(feat, ours, metric, our_name, out_dir, gt_value=None):
    direction = METRIC_DIR.get(metric, "asc")
    df = feat[["Method", metric]].dropna().copy()
    df["Method"] = df["Method"].replace(METHOD_RENAME)

    have_ours = ours is not None and metric.lower() in ours.columns
    if have_ours:
        od = pd.DataFrame({"Method": our_name,
                           metric: ours[metric.lower()].values})
        df = pd.concat([df, od], ignore_index=True)

    med = df.groupby("Method")[metric].median()
    # order so best is at the TOP (last in the list -> highest y)
    methods = med.sort_values(ascending=(direction == "asc")).index.tolist()
    data = [df[df.Method == m][metric].values for m in methods]

    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(methods) + 1.2))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black"))
    for i, m in enumerate(methods):
        bp["boxes"][i].set(facecolor="crimson" if m == our_name else "0.8", alpha=0.9)
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        y = rng.normal(i + 1, 0.05, len(data[i]))
        ax.scatter(data[i], y, s=6,
                   color="darkred" if m == our_name else "0.35", alpha=0.5, zorder=3)
    ax.set_yticks(range(1, len(methods) + 1))
    ax.set_yticklabels([m + ("  ★" if m == our_name else "") for m in methods])
    for lbl in ax.get_yticklabels():
        if our_name in lbl.get_text():
            lbl.set_color("crimson"); lbl.set_fontweight("bold")
    not_comparable = metric in REFERENCE_ONLY
    if gt_value is not None and np.isfinite(gt_value):
        ax.axvline(gt_value, color="green", ls="--", lw=1.5,
                   label=f"ground truth = {gt_value:.3f}")
        ax.legend(loc="lower right", fontsize=8)
        # if the GT itself falls outside the baseline range, the metric is not
        # comparable across preprocessing pipelines
        base_med = df[df.Method != our_name].groupby("Method")[metric].median()
        if direction == "asc" and gt_value < base_med.min():
            not_comparable = True
        if direction == "des" and gt_value > base_med.max():
            not_comparable = True
    # drop the "higher/lower better" direction hint on non-comparable panels:
    # for these the absolute value is a data/preprocessing property, not a
    # quality ranking, so implying a direction would be misleading.
    if not_comparable:
        ax.set_xlabel(PRETTY.get(metric, metric))
    else:
        better = "higher better" if direction == "asc" else "lower better"
        ax.set_xlabel(f"{PRETTY.get(metric, metric)}  ({better})")
    sub = "" if have_ours else "  [baselines only]"
    ax.set_title(f"{feat.name}: {PRETTY.get(metric, metric)} by method{sub}\n"
                 "(Yuan et al. 2024 + ours)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, f"benchmark_{feat.name}_{metric}.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)

    rank_asc = (direction == "des")   # best-first ordering for ranking
    r = df.groupby("Method")[metric].median().sort_values(ascending=rank_asc)
    ranking = r.reset_index(); ranking["rank"] = range(1, len(ranking) + 1)
    ranking.to_csv(os.path.join(out_dir, f"benchmark_{feat.name}_{metric}_ranking.csv"),
                   index=False)
    our_rank = (int(ranking[ranking.Method == our_name]["rank"].iloc[0])
                if have_ours and metric not in REFERENCE_ONLY else None)
    return dict(metric=metric, n_methods=len(methods), our_shown=bool(have_ours),
                reference_only=metric in REFERENCE_ONLY,
                our_rank=our_rank, our_median=(float(r[our_name]) if have_ours else None))


def main():
    a = build_parser().parse_args()
    feat = pd.read_feather(a.feather)
    feat = feat[feat["Biotech"] == a.biotech].copy()
    feat.name = a.biotech
    ours = None
    if os.path.exists(a.our_per_run):
        ours = pd.read_csv(a.our_per_run)

    gt = {}
    if a.gt_calibration and os.path.exists(a.gt_calibration):
        g = pd.read_csv(a.gt_calibration)
        gt = {"MoranI": float(g["gt_morani"].median()),
              "GearyC": float(g["gt_gearyc"].median())}

    present = [m for m in METRIC_DIR if m in feat.columns]
    metrics = present if a.metrics == "all" else \
        [m for m in a.metrics.split(",") if m in present]
    os.makedirs(a.output_dir, exist_ok=True)

    summary = []
    for m in metrics:
        summary.append(one_panel(feat, ours, m, a.our_name, a.output_dir,
                                 gt_value=gt.get(m)))
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(a.output_dir, f"benchmark_{a.biotech}_all_metrics_ranks.csv"),
               index=False)
    print(f"{a.biotech}: generated {len(metrics)} panels -> {a.output_dir}/")
    for r in summary:
        if r["our_rank"]:
            rk = f"rank {r['our_rank']}/{r['n_methods']}"
        elif r["our_shown"] and r["reference_only"]:
            rk = "ours shown (reference only — different hardware, not ranked)"
        elif r["our_shown"]:
            rk = "ours shown (not ranked)"
        else:
            rk = "baselines only"
        print(f"  {r['metric']:8s}: {rk}")


if __name__ == "__main__":
    main()
