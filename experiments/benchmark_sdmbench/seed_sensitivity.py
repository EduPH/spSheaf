#!/usr/bin/env python
"""
Seed-sensitivity of the sheaf pipeline on the MERFISH benchmark.

The only stochastic step is the k-means initialisation (``random_state``). This
runs every slice under N seeds and reports how much ARI/NMI move, so the
benchmark figure can show our method as mean +/- std rather than a single point.

Molecules are loaded once and reused across all (seed, slice) runs.

Usage
-----
    python seed_sensitivity.py \
        --mol-path ../MERFISH/moffitt2018_export_full/molecules_full_add.csv.gz \
        --seeds 0-19 --output-dir output_seed_sensitivity
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spsheaf import io, pipeline


def parse_seeds(spec):
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mol-path",
                   default="../MERFISH/moffitt2018_export_full/molecules_full_add.csv.gz")
    p.add_argument("--seeds", default="0-19", help="'0-19' or '1,2,3'")
    p.add_argument("--slices", default=None,
                   help="Comma-separated subset of slice names (default: all).")
    p.add_argument("--output-dir", default="output_seed_sensitivity")
    # MERFISH reference hyperparameters
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--grid-spacing", type=float, default=15)
    p.add_argument("--bandwidth", type=float, default=25)
    p.add_argument("--k-graph", type=int, default=6)
    p.add_argument("--k-nbrs", type=int, default=50)
    p.add_argument("--uniform-dim", type=int, default=6)
    p.add_argument("--smooth-radius", type=float, default=60.0)
    p.add_argument("--spatial-weight", type=float, default=0.5)
    p.add_argument("--boundary-alpha", type=float, default=0.5)
    p.add_argument("--no-autocorr", action="store_true",
                   help="Skip Moran's I / Geary's C marker autocorrelation.")
    return p


def main():
    a = build_parser().parse_args()
    seeds = parse_seeds(a.seeds)
    os.makedirs(a.output_dir, exist_ok=True)

    mol = io.load_data(a.mol_path)
    slice_names = sorted(io.identify_slices(mol, verbose=False)["slice_name"].tolist())
    if a.slices:
        wanted = set(a.slices.split(","))
        slice_names = [s for s in slice_names if s in wanted]
    print(f"{len(seeds)} seeds x {len(slice_names)} slices = "
          f"{len(seeds) * len(slice_names)} runs")

    rows = []
    t0 = time.time()
    for seed in seeds:
        for s in slice_names:
            _t = time.time()
            r = pipeline.run_single_slice(
                mol, s, n_clusters=a.n_clusters,
                grid_spacing=a.grid_spacing, bandwidth=a.bandwidth,
                k_graph=a.k_graph, k_nbrs=a.k_nbrs, uniform_dim=a.uniform_dim,
                smooth_radius=a.smooth_radius, spatial_weight=a.spatial_weight,
                boundary_alpha=a.boundary_alpha,
                random_state=seed, compute_autocorr=not a.no_autocorr,
                verbose=False)
            elapsed = time.time() - _t
            mets = r.get("metrics", {}) or {}
            rec = dict(seed=seed, slice=s, time=elapsed)
            for k, v in mets.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rec[k] = float(v)
            rec.setdefault("ari", float(r["ari"]))
            rec.setdefault("nmi", float(r["nmi"]))
            rows.append(rec)
            print(f"  seed {seed:>2} {s}: ARI={rec.get('ari', float('nan')):.3f} "
                  f"NMI={rec.get('nmi', float('nan')):.3f} "
                  f"(metrics: {', '.join(k for k in rec if k not in ('seed','slice'))})",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.output_dir, "seed_sensitivity_per_run.csv"),
              index=False)

    metric_cols = [c for c in df.columns if c not in ("seed", "slice")]

    # per-slice: mean +/- std across seeds (all metrics)
    per_slice = df.groupby("slice")[metric_cols].agg(["mean", "std"])
    per_slice.columns = ["_".join(c) for c in per_slice.columns]
    per_slice = per_slice.reset_index()
    per_slice.to_csv(os.path.join(a.output_dir, "seed_sensitivity_per_slice.csv"),
                     index=False)

    # dataset-level: per-seed mean across slices, then spread across seeds
    per_seed = df.groupby("seed")[metric_cols].mean()
    summary = pd.DataFrame({
        "metric": [m.upper() for m in metric_cols],
        "mean": [per_seed[m].mean() for m in metric_cols],
        "std": [per_seed[m].std() for m in metric_cols],
        "min": [per_seed[m].min() for m in metric_cols],
        "max": [per_seed[m].max() for m in metric_cols],
        "n_seeds": [len(seeds)] * len(metric_cols),
    })
    summary.to_csv(os.path.join(a.output_dir, "seed_sensitivity_summary.csv"),
                   index=False)

    _plot(df, per_seed, summary, a.output_dir)
    print(f"\n=== dataset-level (mean over slices, across {len(seeds)} seeds) ===")
    for _, r in summary.iterrows():
        print(f"  {r.metric}: {r['mean']:.3f} +/- {r['std']:.3f} "
              f"[{r['min']:.3f}, {r['max']:.3f}]")
    print(f"\nTotal time {time.time() - t0:.0f}s -> {a.output_dir}/")


def _plot(df, per_seed, summary, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    slices = sorted(df["slice"].unique())
    for k, metric in enumerate(["ari", "nmi"]):
        data = [df[df["slice"] == s][metric].values for s in slices]
        bp = ax[k].boxplot(data, labels=[s.replace("MERFISH_", "").replace(".h5ad", "")
                                          for s in slices], showmeans=True)
        m = summary[summary.metric == metric.upper()]["mean"].values[0]
        sd = summary[summary.metric == metric.upper()]["std"].values[0]
        ax[k].axhline(m, color="crimson", ls="--",
                      label=f"dataset {metric.upper()} = {m:.3f}±{sd:.3f}")
        ax[k].set_title(f"{metric.upper()} across seeds, per slice")
        ax[k].set_xlabel("slice (z)"); ax[k].set_ylabel(metric.upper())
        ax[k].legend(fontsize=8)
    fig.suptitle(f"Sheaf pipeline seed sensitivity — MERFISH "
                 f"({per_seed.shape[0]} seeds)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "seed_sensitivity.png"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
