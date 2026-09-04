#!/usr/bin/env python
"""
Run sheaf boundary detection on a single MERFISH slice.

Usage:
    python scripts/run_single_slice.py [path_to_csv] [--slice SLICE_NAME]

If no slice is specified, the largest slice is selected automatically.
"""

import argparse
import sys
import os


from spsheaf import io, pipeline
from spsheaf.viz.boundaries import plot_single_slice_results


def main():
    parser = argparse.ArgumentParser(
        description="Sheaf boundary detection on a single MERFISH slice",
    )
    parser.add_argument(
        "mol_path",
        nargs="?",
        default="moffitt2018_export_full/molecules_full_add.csv.gz",
        help="Path to molecule CSV",
    )
    parser.add_argument("--slice", type=str, default=None,
                        help="Slice name (matched_h5ad_file value)")
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
    parser.add_argument("--output-dir", type=str, default="output_sheaf_boundary")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    mol = io.load_data(args.mol_path)
    slice_info = io.identify_slices(mol)

    slice_name = args.slice
    if slice_name is None:
        best_idx = slice_info["n_molecules"].idxmax()
        slice_name = slice_info.loc[best_idx, "slice_name"]
        print(f"\n  Auto-selected largest slice: {slice_name}")

    result = pipeline.run_single_slice(
        mol, slice_name,
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

    if not args.no_plot and result["gt_int"] is not None:
        plot_single_slice_results(
            result["grid_coords"],
            result["labels"],
            result["gt_int"], result["gt_names"], result["gt_valid"],
            result["boundary_raw"], result["boundary_smooth"],
            result["ari"], result["nmi"],
            slice_name=slice_name,
            output_dir=args.output_dir,
        )

    print(f"\n{'=' * 60}")
    print(f"RESULT: ARI = {result['ari']:.3f}, NMI = {result['nmi']:.3f}")
    if result["n_reassigned"] > 0:
        print(f"  Cleanup: {result['n_reassigned']} fragments reassigned")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()