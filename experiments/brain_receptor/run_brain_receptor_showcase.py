#!/usr/bin/env python
"""
Run sheaf boundary detection on a Vizgen MERSCOPE detected_transcripts CSV
(designed for the Brain Receptor Showcase, Slice 1 / Replicate 1, but works
for any single-slice Vizgen MERSCOPE export).

Adaptation vs. the MERFISH benchmark (`spsheaf.pipeline.run_single_slice`)
-------------------------------------------------------------------------------
Imoto's Moffitt 2018 export has ``matched_h5ad_file`` (slice ID) and
``domain`` (ground truth). The Vizgen MERSCOPE format has neither:

    * one tissue section per file       -> no slice selection
    * no ground-truth annotations       -> no ARI / NMI (unsupervised metrics
                                           are reported instead, see below)
    * coordinates in ``global_x``,
      ``global_y`` (um)                 -> renamed to x, y
    * gene name in ``gene``             -> already matches
    * several z focal planes            -> keep ``global_z == 0`` by default

Every algorithmic step (kernel fibers -> spatial graph -> local PCA stalks ->
restriction maps -> boundary scores -> boundary-aware clustering -> spatial
cleanup) is identical to ``run_single_slice``.

Scale handling (follows the Stereo-seq adaptation in run_seqscope.py)
--------------------------------------------------------------------
A whole mouse-brain coronal section (~9.7 x 6.6 mm, ~55M molecules across 7 z
planes and 649 genes) is far larger than the Moffitt hypothalamic ROI, and does
not fit the naive pipeline in a small-RAM environment. Three preprocessing
steps make it tractable, exactly as for Stereo-seq:

    1. z-plane filter      keep a single focal plane (~7.5M molecules).
    2. gene filtering      keep the top ``--top-genes`` genes by molecule count
                           (reduces the fiber dimension G, and thus stalk memory
                           which scales as N x G x d).
    3. subsampling         keep a random ``--subsample`` fraction of molecules.
                           At a coarse grid each grid point still aggregates
                           thousands of molecules, so this is statistically
                           indistinguishable from the full set.

In addition, the Gaussian-kernel fiber field is built in *blocks* of grid
points (``build_kernel_fibers_batched``) so the neighbour lists from the
radius query never materialise all at once -- the only change from the library
``fibers.build_kernel_fibers``; the math is identical.

Metrics (no ground truth)
-------------------------
This dataset ships no manually curated spatial-domain annotation, so ARI/NMI
against a reference cannot be computed. Instead we report *unsupervised*
quality metrics on the inferred partition (saved to ``metrics.csv``):

    * silhouette            expression-space cluster separation  (higher better)
    * calinski_harabasz     between/within cluster variance ratio (higher better)
    * davies_bouldin        average cluster overlap               (lower  better)
    * spatial_coherence     fraction of spatial-graph edges whose endpoints
                            share a label -> domain contiguity     (higher better)
    * boundary_contrast     mean smoothed boundary score on cross-label edges
                            divided by that on same-label edges. >1 means the
                            sheaf boundary field concentrates on domain
                            transitions, i.e. the boundary signal agrees with
                            the clustering.

Usage
-----
    python run_brain_receptor_showcase.py \\
        --input datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_detected_transcripts_S1R1.csv \\
        --n-clusters 15 \\
        --output-dir output_brain_receptor

Outputs (in --output-dir)
-------------------------
    domain_labels.csv    : per-grid-point (x, y, label)
    boundary_field.csv   : per-grid-point (x, y, boundary_raw, boundary_smooth)
    metrics.csv          : unsupervised quality metrics
    domain_map.png       : 1x2 diagnostic (labels + smoothed boundary field)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
)

from spsheaf import fibers, graph, sheaf, clustering


# ============================================================================
# Argument parser
# ============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    p.add_argument(
        "--input", required=True,
        help="Path to the Vizgen MERSCOPE detected_transcripts CSV.",
    )
    p.add_argument(
        "--output-dir", default="output_brain_receptor",
        help="Directory for output files (default: output_brain_receptor).",
    )

    # Column-name overrides (default to Vizgen MERSCOPE conventions)
    p.add_argument("--x-col", default="global_x")
    p.add_argument("--y-col", default="global_y")
    p.add_argument("--z-col", default="global_z")
    p.add_argument("--gene-col", default="gene")
    p.add_argument(
        "--z-plane", type=float, default=0.0,
        help="Keep only molecules at this z value (default: 0.0). "
             "Pass a negative value to keep all planes.",
    )

    # Scale handling (Stereo-seq-style)
    p.add_argument("--top-genes", type=int, default=300,
                   help="Keep the top-N genes by molecule count (0 = all). "
                        "Default 300.")
    p.add_argument("--subsample", type=float, default=0.3,
                   help="Random fraction of molecules to keep after z/gene "
                        "filtering (default: 0.3).")
    p.add_argument("--chunksize", type=int, default=4_000_000,
                   help="CSV read chunk size in rows (default: 4,000,000).")
    p.add_argument("--seed", type=int, default=0)

    # Clustering target
    p.add_argument(
        "--n-clusters", type=int, default=15,
        help="Number of domains to recover (free parameter; no ground truth). "
             "Default 15.",
    )

    # Spatial / kernel parameters (um) -- coarse whole-section defaults
    p.add_argument("--grid-spacing", type=float, default=50.0,
                   help="Fiber grid spacing in um (default: 50).")
    p.add_argument("--bandwidth", type=float, default=100.0,
                   help="Gaussian kernel bandwidth in um (default: 100).")
    p.add_argument("--smooth-radius", type=float, default=200.0,
                   help="Boundary smoothing radius in um (default: 200).")
    p.add_argument("--min-molecules", type=float, default=15.0,
                   help="Minimum weighted count to keep a grid point.")
    p.add_argument("--block-size", type=int, default=1500,
                   help="Grid points processed per fiber-build block "
                        "(memory control; default: 1500).")

    # Graph & stalk construction
    p.add_argument("--k-graph", type=int, default=6)
    p.add_argument("--k-nbrs", type=int, default=50)
    p.add_argument("--uniform-dim", type=int, default=6)

    # Clustering / cleanup
    p.add_argument("--spatial-weight", type=float, default=0.5)
    p.add_argument("--boundary-alpha", type=float, default=0.5)
    p.add_argument("--overcluster-factor", type=float, default=1.0)
    p.add_argument("--no-cleanup", action="store_true")
    p.add_argument("--min-fragment-frac", type=float, default=0.01)

    return p


# ============================================================================
# Data loading  (memory-safe, chunked)
# ============================================================================

def load_vizgen_csv(path, x_col, y_col, z_col, gene_col, z_plane,
                    top_genes, subsample, chunksize, seed, verbose=True):
    """
    Stream a (multi-GB) Vizgen MERSCOPE detected_transcripts CSV in chunks and
    return the compact molecule table the pipeline expects:

        columns: ``gene``, ``x``, ``y``, ``x_local``, ``y_local``

    Only the coordinate + gene columns are read; the requested z-plane is
    selected inside each chunk so peak memory stays small. Top-gene filtering
    and subsampling are applied after concatenation (by then the table is
    already ~1/7 of the file or less).
    """
    if verbose:
        print(f"  Reading (chunked): {path}")

    usecols = [x_col, y_col, gene_col]
    dtypes = {x_col: "float32", y_col: "float32", gene_col: "category"}
    read_z = z_plane >= 0
    if read_z:
        usecols = [x_col, y_col, z_col, gene_col]
        dtypes[z_col] = "float32"

    parts = []
    n_raw = 0
    for ch in pd.read_csv(path, usecols=usecols, dtype=dtypes,
                          chunksize=chunksize):
        n_raw += len(ch)
        if read_z and z_col in ch.columns:
            ch = ch[ch[z_col] == z_plane]
        parts.append(ch[[x_col, y_col, gene_col]])
    mol = pd.concat(parts, ignore_index=True)
    del parts
    mol[gene_col] = mol[gene_col].astype("category")

    if verbose:
        print(f"  Molecules  : {n_raw:,} raw -> {len(mol):,} after z-filter "
              f"(z={z_plane})")
        print(f"  Genes      : {mol[gene_col].nunique()}")

    # Top-gene filtering
    if top_genes and top_genes > 0 and mol[gene_col].nunique() > top_genes:
        keep = mol[gene_col].value_counts().nlargest(top_genes).index
        before = len(mol)
        mol = mol[mol[gene_col].isin(keep)].copy()
        if verbose:
            frac = 100.0 * len(mol) / before
            print(f"  Gene filter: top {top_genes} genes kept "
                  f"({before:,} -> {len(mol):,} molecules, {frac:.1f}%)")

    # Subsampling
    if subsample and 0 < subsample < 1.0:
        before = len(mol)
        mol = mol.sample(frac=subsample, random_state=seed)
        if verbose:
            print(f"  Subsample  : {subsample:.0%} "
                  f"({before:,} -> {len(mol):,} molecules)")

    mol = mol.rename(columns={gene_col: "gene", x_col: "x", y_col: "y"})
    mol["gene"] = mol["gene"].astype(str)
    mol["x_local"] = mol["x"] - mol["x"].min()
    mol["y_local"] = mol["y"] - mol["y"].min()

    if verbose:
        print(f"  Extent     : {mol['x_local'].max():.0f} x "
              f"{mol['y_local'].max():.0f} um")
    return mol


# ============================================================================
# Memory-safe fiber build (block-processed; math identical to the library)
# ============================================================================

def build_kernel_fibers_batched(mol, grid_spacing, bandwidth, min_molecules,
                                block_size=1500, x_col="x_local",
                                y_col="y_local", verbose=True):
    """
    Same construction as ``spsheaf.fibers.build_kernel_fibers`` but the
    radius query is issued for one *block* of grid points at a time, so the
    neighbour lists never materialise for the whole grid at once. This is the
    only change required to run at whole-brain density in limited RAM.
    """
    gene_names = sorted(mol["gene"].unique())
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    mol_coords = mol[[x_col, y_col]].values.astype(np.float64)
    mol_gene_idx = mol["gene"].map(gene_to_idx).values

    x_min, x_max = mol_coords[:, 0].min(), mol_coords[:, 0].max()
    y_min, y_max = mol_coords[:, 1].min(), mol_coords[:, 1].max()
    x_grid = np.arange(x_min + grid_spacing / 2, x_max, grid_spacing)
    y_grid = np.arange(y_min + grid_spacing / 2, y_max, grid_spacing)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
    n_grid = len(grid_coords)

    tree = cKDTree(mol_coords)
    radius = 3.0 * bandwidth

    fibers_arr = np.zeros((n_grid, n_genes), dtype=np.float32)
    total_weight = np.zeros(n_grid, dtype=np.float32)

    t0 = time.time()
    for start in range(0, n_grid, block_size):
        end = min(start + block_size, n_grid)
        block = grid_coords[start:end]
        nbr_lists = tree.query_ball_point(block, r=radius)
        for local_i, nb in enumerate(nbr_lists):
            if not nb:
                continue
            gi = start + local_i
            nb = np.asarray(nb)
            d = np.linalg.norm(mol_coords[nb] - grid_coords[gi], axis=1)
            w = np.exp(-0.5 * (d / bandwidth) ** 2)
            np.add.at(fibers_arr[gi], mol_gene_idx[nb], w)
            total_weight[gi] = w.sum()
        del nbr_lists
        if verbose and (start // block_size) % 5 == 0:
            print(f"    fibers {end:,}/{n_grid:,} grid pts "
                  f"({time.time() - t0:.0f}s)", flush=True)

    keep = total_weight >= min_molecules
    grid_coords = grid_coords[keep]
    fibers_arr = fibers_arr[keep]
    if verbose:
        print(f"  Grid: {len(grid_coords):,} valid points, {n_genes} genes "
              f"({time.time() - t0:.0f}s)")
    return grid_coords, fibers_arr, gene_names


# ============================================================================
# Pipeline
# ============================================================================

def run_pipeline(mol, args, verbose=True):
    print("\n[1/7] Kernel fibers (batched)")
    grid_coords, raw_fibers, gene_names = build_kernel_fibers_batched(
        mol, grid_spacing=args.grid_spacing, bandwidth=args.bandwidth,
        min_molecules=args.min_molecules, block_size=args.block_size,
        verbose=verbose,
    )
    fibers_norm = fibers.normalize_fibers(raw_fibers)
    del raw_fibers

    print("\n[2/7] Spatial graph")
    edges = graph.build_spatial_graph(grid_coords, k=args.k_graph,
                                      verbose=verbose)

    print("\n[3/7] Stalks (local PCA)")
    stalks = sheaf.compute_stalks(
        fibers_norm, grid_coords,
        uniform_dim=args.uniform_dim, k_nbrs=args.k_nbrs, verbose=verbose,
    )

    print("\n[4/7] Restriction maps")
    restrictions = sheaf.compute_restriction_maps(
        stalks, edges, args.uniform_dim, verbose=verbose,
    )

    print("\n[5/7] Boundary scores")
    boundary_raw, boundary_smooth = sheaf.compute_boundary_scores(
        fibers_norm, grid_coords, stalks, edges, restrictions,
        smooth_radius=args.smooth_radius, verbose=verbose,
    )

    print("\n[6/7] Boundary-aware clustering")
    labels = clustering.boundary_aware_clustering(
        fibers_norm, grid_coords, boundary_smooth, edges,
        n_clusters=args.n_clusters,
        spatial_weight=args.spatial_weight,
        boundary_alpha=args.boundary_alpha,
        overcluster_factor=args.overcluster_factor,
        verbose=verbose,
    )

    n_reassigned = 0
    if not args.no_cleanup:
        print("\n[7/7] Spatial cleanup")
        labels, n_reassigned = clustering.spatial_cleanup(
            labels, edges, grid_coords=grid_coords,
            min_fragment_frac=args.min_fragment_frac, verbose=verbose,
        )

    return dict(
        grid_coords=grid_coords, fibers_norm=fibers_norm,
        boundary_raw=boundary_raw, boundary_smooth=boundary_smooth,
        labels=np.asarray(labels), edges=edges, n_reassigned=n_reassigned,
        n_points=len(grid_coords), n_genes=len(gene_names),
        gene_names=gene_names,
    )


# ============================================================================
# Unsupervised metrics (no ground truth available for this dataset)
# ============================================================================

def compute_metrics(result, seed=0, max_pts=6000, verbose=True):
    X = result["fibers_norm"]
    labels = result["labels"]
    edges = np.asarray(result["edges"])
    bsm = result["boundary_smooth"]
    uniq = np.unique(labels)
    metrics = {}
    metrics["n_domains"] = int(len(uniq))
    metrics["n_grid_points"] = int(len(labels))

    # Internal expression-space validation (subsample for tractable O(n^2))
    rng = np.random.default_rng(seed)
    if len(labels) > max_pts:
        idx = rng.choice(len(labels), size=max_pts, replace=False)
    else:
        idx = np.arange(len(labels))
    Xs, ls = X[idx], labels[idx]
    if len(np.unique(ls)) > 1:
        metrics["silhouette"] = float(silhouette_score(Xs, ls))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(Xs, ls))
        metrics["davies_bouldin"] = float(davies_bouldin_score(Xs, ls))
    else:
        metrics["silhouette"] = float("nan")
        metrics["calinski_harabasz"] = float("nan")
        metrics["davies_bouldin"] = float("nan")

    # Spatial coherence: fraction of graph edges within a single domain
    same = labels[edges[:, 0]] == labels[edges[:, 1]]
    metrics["spatial_coherence"] = float(same.mean())

    # Boundary contrast: boundary field on cross-label vs within-label edges
    edge_b = 0.5 * (bsm[edges[:, 0]] + bsm[edges[:, 1]])
    cross = edge_b[~same].mean() if (~same).any() else float("nan")
    within = edge_b[same].mean() if same.any() else float("nan")
    metrics["boundary_mean_cross_edge"] = float(cross)
    metrics["boundary_mean_within_edge"] = float(within)
    metrics["boundary_contrast"] = float(cross / within) if within else float("nan")

    # Domain size distribution
    sizes = np.array([(labels == c).sum() for c in uniq])
    metrics["domain_size_min"] = int(sizes.min())
    metrics["domain_size_max"] = int(sizes.max())
    metrics["domain_size_cv"] = float(sizes.std() / sizes.mean())

    if verbose:
        print("\nUnsupervised metrics (no ground truth):")
        for k, v in metrics.items():
            print(f"  {k:26s}: {v}")
    return metrics


# ============================================================================
# Output
# ============================================================================

def save_outputs(result, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    gc = result["grid_coords"]

    pd.DataFrame({"x": gc[:, 0], "y": gc[:, 1], "label": result["labels"]}
                 ).to_csv(os.path.join(output_dir, "domain_labels.csv"),
                          index=False)
    print(f"  -> {os.path.join(output_dir, 'domain_labels.csv')}")

    pd.DataFrame({"x": gc[:, 0], "y": gc[:, 1],
                  "boundary_raw": result["boundary_raw"],
                  "boundary_smooth": result["boundary_smooth"]}
                 ).to_csv(os.path.join(output_dir, "boundary_field.csv"),
                          index=False)
    print(f"  -> {os.path.join(output_dir, 'boundary_field.csv')}")

    pd.DataFrame({"metric": list(metrics.keys()),
                  "value": list(metrics.values())}
                 ).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    print(f"  -> {os.path.join(output_dir, 'metrics.csv')}")

    fig = _plot(gc, result["labels"], result["boundary_smooth"], metrics)
    fig.savefig(os.path.join(output_dir, "domain_map.png"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.join(output_dir, 'domain_map.png')}")


def _plot(grid_coords, labels, boundary_smooth, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    ax = axes[0]
    uniq = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 2))
    # y is inverted so anterior/dorsal orientation matches image convention
    for i, c in enumerate(uniq):
        m = labels == c
        ax.scatter(grid_coords[m, 0], grid_coords[m, 1], s=3, alpha=0.7,
                   color=cmap(i))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"Sheaf-inferred domains (k = {len(uniq)})  |  "
                 f"coherence = {metrics['spatial_coherence']:.2f}", fontsize=11)
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")

    ax = axes[1]
    vmax = np.percentile(boundary_smooth, 95)
    sc = ax.scatter(grid_coords[:, 0], grid_coords[:, 1], c=boundary_smooth,
                    s=3, cmap="hot", vmax=vmax if vmax > 0 else 1)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"Smoothed boundary score  |  contrast = "
                 f"{metrics['boundary_contrast']:.2f}", fontsize=11)
    ax.set_xlabel("x (um)")
    plt.colorbar(sc, ax=ax, shrink=0.7)

    fig.suptitle("Sheaf boundary detection - Vizgen Brain Receptor Showcase "
                 "(Slice 1, Replicate 1)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ============================================================================
# Entry point
# ============================================================================

def main():
    args = build_parser().parse_args()
    print("=" * 64)
    print(" SHEAF BOUNDARY - Vizgen Brain Receptor Showcase")
    print("=" * 64)
    t0 = time.time()

    mol = load_vizgen_csv(
        args.input, x_col=args.x_col, y_col=args.y_col, z_col=args.z_col,
        gene_col=args.gene_col, z_plane=args.z_plane,
        top_genes=args.top_genes, subsample=args.subsample,
        chunksize=args.chunksize, seed=args.seed,
    )

    result = run_pipeline(mol, args)
    del mol

    metrics = compute_metrics(result, seed=args.seed)

    print("\nSaving outputs...")
    save_outputs(result, metrics, args.output_dir)

    print("\n" + "=" * 64)
    print(f"  Grid points  : {result['n_points']:,}")
    print(f"  Genes        : {result['n_genes']}")
    print(f"  Clusters     : {len(set(result['labels']))}")
    print(f"  Cleaned      : {result['n_reassigned']}")
    print(f"  Total time   : {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / 60:.1f} min)")
    print("=" * 64)


if __name__ == "__main__":
    main()
