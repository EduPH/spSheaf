#!/usr/bin/env python
"""
Comparison figure: our method vs the methods in Yuan et al. 2024 (SDMBench),
using their Fig. 2d source data.

Because the exact column layout of the source file is confirmed on first sight,
run ``--inspect`` first, then set the column names. The figure ranks methods by
the chosen metric (ARI/NMI) with our method appended and highlighted, drawn as
mean +/- std from the seed-sensitivity run.

Usage
-----
    python benchmark_comparison.py --benchmark-file <fig2d source>.csv --inspect

    python benchmark_comparison.py --benchmark-file <fig2d source>.csv \
        --method-col method --metric-col metric --value-col value \
        --dataset-col dataset --dataset MERFISH --metric NMI \
        --our-mean 0.631 --our-std 0.01 --output-dir output_benchmark
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark-file", required=True,
                   help="Yuan et al. Fig 2d source data (.csv/.tsv/.xlsx).")
    p.add_argument("--inspect", action="store_true")
    # long-format column mapping (set after --inspect)
    p.add_argument("--method-col", default=None)
    p.add_argument("--metric-col", default=None)
    p.add_argument("--value-col", default=None)
    p.add_argument("--dataset-col", default=None)
    p.add_argument("--dataset", default=None, help="Which dataset value to filter to.")
    p.add_argument("--metric", default="NMI")
    # our method (from seed_sensitivity_summary.csv)
    p.add_argument("--our-name", default="Sheaf (ours)")
    p.add_argument("--our-mean", type=float, default=None)
    p.add_argument("--our-std", type=float, default=0.0)
    p.add_argument("--our-summary", default=None,
                   help="seed_sensitivity_summary.csv to read our mean/std from.")
    p.add_argument("--output-dir", default="output_benchmark")
    return p


def load_any(path):
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def inspect(df):
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("\nhead:\n", df.head(10).to_string())
    for c in df.columns:
        u = df[c].nunique()
        if u <= 30:
            print(f"\n{c} ({u} unique): {sorted(map(str, df[c].dropna().unique()))[:30]}")


def our_scores(a):
    if a.our_summary and os.path.exists(a.our_summary):
        s = pd.read_csv(a.our_summary)
        row = s[s.metric.str.upper() == a.metric.upper()]
        if len(row):
            return float(row["mean"].iloc[0]), float(row["std"].iloc[0])
    if a.our_mean is not None:
        return a.our_mean, a.our_std
    raise SystemExit("Provide --our-mean/--our-std or --our-summary.")


def main():
    a = build_parser().parse_args()
    df = load_any(a.benchmark_file)
    if a.inspect:
        inspect(df)
        print("\nNext: set --method-col/--metric-col/--value-col "
              "(+ --dataset-col --dataset) and rerun.")
        return

    if not (a.method_col and a.value_col):
        raise SystemExit("Set --method-col and --value-col (see --inspect).")

    d = df.copy()
    if a.metric_col and a.metric:
        d = d[d[a.metric_col].astype(str).str.upper() == a.metric.upper()]
    if a.dataset_col and a.dataset:
        d = d[d[a.dataset_col].astype(str).str.contains(a.dataset, case=False)]

    # per-method distribution (boxplot) or single value
    grouped = d.groupby(a.method_col)[a.value_col]
    methods = grouped.mean().sort_values()
    our_mean, our_std = our_scores(a)

    os.makedirs(a.output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * (len(methods) + 1)), 5.5))
    names = list(methods.index) + [a.our_name]
    means = list(methods.values) + [our_mean]
    colors = ["0.7"] * len(methods) + ["crimson"]
    errs = [d[d[a.method_col] == m][a.value_col].std() for m in methods.index] + [our_std]
    ax.bar(range(len(names)), means, yerr=errs, color=colors, capsize=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylabel(a.metric)
    ttl = f"{a.metric} by method"
    if a.dataset:
        ttl += f" — {a.dataset}"
    ax.set_title(ttl + " (Yuan et al. 2024 + ours)")
    ax.axhline(our_mean, color="crimson", ls="--", lw=0.8)
    fig.tight_layout()
    out = os.path.join(a.output_dir, f"benchmark_{a.metric}.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)

    comp = pd.DataFrame({"method": names, a.metric: means, "std": errs})
    comp.to_csv(os.path.join(a.output_dir, f"benchmark_{a.metric}.csv"), index=False)
    rank = names[::-1].index(a.our_name) + 1
    print(f"Our {a.metric} = {our_mean:.3f} ± {our_std:.3f} "
          f"-> rank {rank}/{len(names)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
