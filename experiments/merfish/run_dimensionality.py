#!/usr/bin/env python
"""
Run dimensionality analysis on a MERFISH slice.

Usage:
    python scripts/run_dimensionality.py [path_to_csv] [--slice SLICE_NAME]

Estimates local and per-patch intrinsic dimensionality via PCA and
Isomap, selects manifold-relevant genes, and generates diagnostic plots.
"""

import argparse
import sys
import os
import numpy as np


from spsheaf import io, fibers, dimensionality
from spsheaf.viz.dimensionality import (
    plot_dim_per_patch,
    plot_local_dim_fields,
    plot_dim_histograms,
    plot_dim_all_patches,
    plot_gene_selection,
)


def main():
    parser = argparse.ArgumentParser(
        description="Dimensionality analysis of MERFISH expression manifold",
    )
    parser.add_argument(
        "mol_path",
        nargs="?",
        default="moffitt2018_export_full/molecules_full_add.csv.gz",
        help="Path to molecule CSV",
    )
    parser.add_argument("--slice", type=str, default=None,
                        help="Slice name (auto-selects largest if omitted)")
    parser.add_argument("--grid-spacing", type=float, default=15)
    parser.add_argument("--bandwidth", type=float, default=25)
    parser.add_argument("--patch-size", type=float, default=400)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--k-local", type=int, default=50)
    parser.add_argument("--n-patches-dim", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="output_dimensionality")
    args = parser.parse_args()

    print("╔" + "═" * 58 + "╗")
    print("║  MERFISH — DIMENSIONALITY ANALYSIS                       ║")
    print("╚" + "═" * 58 + "╝")

    # Load
    mol = io.load_data(args.mol_path)
    slice_info = io.identify_slices(mol)

    slice_name = args.slice
    if slice_name is None:
        best_idx = slice_info["n_molecules"].idxmax()
        slice_name = slice_info.loc[best_idx, "slice_name"]
        print(f"\n  Auto-selected largest slice: {slice_name}")

    mol_single = io.select_slice(mol, slice_name)

    # Build fibers
    grid_coords, raw_fibers, gene_names = fibers.build_kernel_fibers(
        mol_single, grid_spacing=args.grid_spacing, bandwidth=args.bandwidth,
    )
    fibers_norm = fibers.normalize_fibers(raw_fibers)

    # Atlas cover
    subset_assignments, patch_centers = dimensionality.build_atlas_cover(
        grid_coords, patch_size=args.patch_size, overlap=args.overlap,
    )

    # Per-patch dimensionality (detailed, first n)
    patch_dim = dimensionality.estimate_dim_per_patch(
        fibers_norm, subset_assignments,
        max_dim=min(50, len(gene_names) - 1),
        n_patches=args.n_patches_dim,
    )
    plot_dim_per_patch(patch_dim, output_dir=args.output_dir)

    # All-patches dimensionality
    pca_all, iso_all = dimensionality.estimate_dim_all_patches(
        fibers_norm, subset_assignments, patch_centers,
        max_dim=min(50, len(gene_names) - 1),
    )
    plot_dim_all_patches(
        patch_centers, pca_all, iso_all, grid_coords,
        output_dir=args.output_dir,
    )

    # Local dimension fields
    dim_pca = dimensionality.local_dim_field_pca(
        fibers_norm, grid_coords, k=args.k_local,
    )
    dim_iso = dimensionality.local_dim_field_isomap(
        fibers_norm, grid_coords, k=args.k_local,
    )
    plot_local_dim_fields(grid_coords, dim_pca, dim_iso,
                          output_dir=args.output_dir)
    plot_dim_histograms(dim_pca, dim_iso, output_dir=args.output_dir)

    # Gene selection
    gene_sel = dimensionality.select_manifold_genes(
        fibers_norm, grid_coords, gene_names, latent_dim=3, k=args.k_local,
    )
    plot_gene_selection(gene_sel, gene_names, output_dir=args.output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Slice          : {slice_name}")
    print(f"  Grid points    : {len(grid_coords):,}")
    print(f"  Genes          : {len(gene_names)}")
    print(f"  Patches        : {len(subset_assignments)}")
    print(f"  Local dim PCA  : mean={dim_pca.mean():.1f}, "
          f"median={np.median(dim_pca):.0f}")
    print(f"  Local dim Iso  : mean={dim_iso.mean():.1f}, "
          f"median={np.median(dim_iso):.0f}")
    print(f"  Genes for 90%  : {gene_sel['n_90']} / {len(gene_names)}")
    print(f"\n  Outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
