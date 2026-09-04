#!/usr/bin/env python
"""
Run sheaf boundary detection on all MERFISH slices.

Usage:
    python scripts/run_all_slices.py [path_to_csv]

Produces per-slice diagnostic plots, summary bar charts, and a
CSV with all results.
"""

import argparse
import sys
import os


from spsheaf import pipeline
from spsheaf.viz.boundaries import plot_single_slice_results
from spsheaf.viz.benchmark import (
    plot_all_slices_summary,
    plot_per_slice_comparison,
)


def main():
    parser = argparse.ArgumentParser(
        description="Sheaf boundary detection — all slices benchmark",
    )
    parser.add_argument(
        "mol_path",
        nargs="?",
        default="moffitt2018_export_full/molecules_full_add.csv.gz",
        help="Path to molecule CSV",
    )
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--grid-spacing", type=float, default=15)
    parser.add_argument("--bandwidth", type=float, default=25)
    parser.add_argument("--k-graph", type=int, default=6)
    parser.add_argument("--k-nbrs", type=int, default=50)
    parser.add_argument("--uniform-dim", type=int, default=6)
    parser.add_argument("--smooth-radius", type=float, default=60.0)
    parser.add_argument("--spatial-weight", type=float, default=0.5)
    parser.add_argument("--boundary-alpha", type=float, default=0.5)
    parser.add_argument("--overcluster-factor", type=float, default=1.0,
                        help="Over-clustering factor (1.0 to disable)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip spatial connectivity cleanup")
    parser.add_argument("--output-dir", type=str, default="output_merfish")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    summary, all_results = pipeline.run_all_slices(
        mol_path=args.mol_path,
        n_clusters=args.n_clusters,
        grid_spacing=args.grid_spacing,
        bandwidth=args.bandwidth,
        k_graph=args.k_graph,
        k_nbrs=args.k_nbrs,
        uniform_dim=args.uniform_dim,
        smooth_radius=args.smooth_radius,
        spatial_weight=args.spatial_weight,
        boundary_alpha=args.boundary_alpha,
        overcluster_factor=args.overcluster_factor,
        cleanup=not args.no_cleanup,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Per-slice plots
    if not args.no_plot:
        for result in all_results:
            if result is not None and result["gt_int"] is not None:
                plot_single_slice_results(
                    result["grid_coords"],
                    result["labels"],
                    result["gt_int"], result["gt_names"], result["gt_valid"],
                    result["boundary_raw"], result["boundary_smooth"],
                    result["ari"], result["nmi"],
                    slice_name=result["slice_name"],
                    output_dir=args.output_dir,
                )

        # Summary plots
        plot_all_slices_summary(summary, output_dir=args.output_dir)
        plot_per_slice_comparison(summary, output_dir=args.output_dir)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "all_slices_results.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n  → Results saved to: {csv_path}")
    print(f"  Outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()