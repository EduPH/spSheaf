#!/usr/bin/env python
"""
3D sheaf boundary detection on the Vizgen Brain Receptor Showcase.

This is the 3D counterpart of ``run_brain_receptor_showcase.py``. Instead of
collapsing the MERSCOPE focal planes to a single z, it keeps the full
``(x, y, z)`` molecule cloud and runs the pipeline over a 3D grid. It exercises
the N-D code path added to the ``spsheaf`` package: ``build_kernel_fibers``
with ``z_col`` builds a 3D grid + anisotropic Gaussian kernel; the spatial
graph, stalks, restriction maps, disagreement score, boundary smoothing, and
clustering are all dimension-agnostic and unchanged.

Note on this dataset: MERSCOPE z is a stack of ~7 focal planes ~1.5 um apart,
so the volume is only ~10 um thick against a ~9 mm field. 3D here is therefore
mainly an engineering testbed for the N-D input path — domain boundaries are
near-vertical columns. The same code applies unchanged to genuinely volumetric
data (e.g. DNA seqFISH+ nuclear zones), which is the intended use.

No ground truth is available; unsupervised metrics are reported (see
``run_brain_receptor_showcase.compute_metrics``).

Usage
-----
    # full run (do this on a normal machine)
    python run_brain_receptor_3d.py \
        --input datasets_..._detected_transcripts_S1R1.csv \
        --grid-spacing 50 --bandwidth 100 --z-step-um 1.5 \
        --z-spacing 1.5 --z-bandwidth 3 --n-clusters 15 \
        --output-dir output_brain_receptor_3d

    # quick test on a crop
    python run_brain_receptor_3d.py --input ... --crop 3000,5000,2000,4000 \
        --subsample 0.3 --top-genes 150 --output-dir /tmp/out3d
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spsheaf import fibers, graph, sheaf, clustering
from run_brain_receptor_showcase import compute_metrics


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default="output_brain_receptor_3d")
    p.add_argument("--x-col", default="global_x")
    p.add_argument("--y-col", default="global_y")
    p.add_argument("--z-col", default="global_z")
    p.add_argument("--gene-col", default="gene")
    p.add_argument("--z-step-um", type=float, default=1.5,
                   help="Physical spacing between MERSCOPE focal planes (um). "
                        "global_z is a plane index; multiply by this for um.")
    p.add_argument("--crop", default=None,
                   help="x0,x1,y0,y1 in um to restrict the field (for testing).")
    p.add_argument("--top-genes", type=int, default=300)
    p.add_argument("--subsample", type=float, default=0.5)
    p.add_argument("--chunksize", type=int, default=6_000_000)
    p.add_argument("--seed", type=int, default=0)
    # spatial / kernel
    p.add_argument("--grid-spacing", type=float, default=50.0)
    p.add_argument("--bandwidth", type=float, default=100.0)
    p.add_argument("--z-spacing", type=float, default=1.5,
                   help="Grid spacing along z (um).")
    p.add_argument("--z-bandwidth", type=float, default=1.0,
                   help="Gaussian bandwidth along z (um). Small (~1) resolves "
                        "depth per focal plane; large (~3) blends the stack and "
                        "yields near-columnar domains. Pair small values with a "
                        "higher --subsample to offset the added noise.")
    p.add_argument("--smooth-radius", type=float, default=200.0)
    p.add_argument("--min-molecules", type=float, default=15.0)
    p.add_argument("--block-size", type=int, default=2000)
    # graph / stalks / clustering
    p.add_argument("--k-graph", type=int, default=6)
    p.add_argument("--k-nbrs", type=int, default=50)
    p.add_argument("--uniform-dim", type=int, default=6)
    p.add_argument("--n-clusters", type=int, default=15)
    p.add_argument("--spatial-weight", type=float, default=0.5)
    p.add_argument("--boundary-alpha", type=float, default=0.5)
    p.add_argument("--overcluster-factor", type=float, default=1.0)
    p.add_argument("--no-cleanup", action="store_true")
    p.add_argument("--min-fragment-frac", type=float, default=0.01)
    return p


def load_vizgen_3d(a, verbose=True):
    """Chunked load keeping ALL z planes -> (gene, x_local, y_local, z_local)."""
    usecols = [a.x_col, a.y_col, a.z_col, a.gene_col]
    dt = {a.x_col: "float32", a.y_col: "float32", a.z_col: "float32",
          a.gene_col: "category"}
    crop = [float(v) for v in a.crop.split(",")] if a.crop else None
    parts, n_raw = [], 0
    for ch in pd.read_csv(a.input, usecols=usecols, dtype=dt,
                          chunksize=a.chunksize):
        n_raw += len(ch)
        if crop:
            x0, x1, y0, y1 = crop
            ch = ch[(ch[a.x_col] >= x0) & (ch[a.x_col] <= x1) &
                    (ch[a.y_col] >= y0) & (ch[a.y_col] <= y1)]
        parts.append(ch)
    mol = pd.concat(parts, ignore_index=True)
    mol[a.gene_col] = mol[a.gene_col].astype("category")
    if verbose:
        print(f"  Molecules  : {n_raw:,} raw -> {len(mol):,} after crop")
        print(f"  z planes   : {sorted(mol[a.z_col].unique().tolist())}")

    if a.top_genes and mol[a.gene_col].nunique() > a.top_genes:
        keep = mol[a.gene_col].value_counts().nlargest(a.top_genes).index
        mol = mol[mol[a.gene_col].isin(keep)].copy()
    if a.subsample and 0 < a.subsample < 1.0:
        mol = mol.sample(frac=a.subsample, random_state=a.seed)

    mol = mol.rename(columns={a.gene_col: "gene", a.x_col: "x", a.y_col: "y"})
    mol["gene"] = mol["gene"].astype(str)
    mol["x_local"] = mol["x"] - mol["x"].min()
    mol["y_local"] = mol["y"] - mol["y"].min()
    # z index -> physical microns
    mol["z_local"] = (mol[a.z_col] - mol[a.z_col].min()) * a.z_step_um
    if verbose:
        print(f"  Extent     : {mol['x_local'].max():.0f} x "
              f"{mol['y_local'].max():.0f} x {mol['z_local'].max():.1f} um "
              f"(x, y, z)")
        print(f"  Genes      : {mol['gene'].nunique()}")
    return mol


def run(mol, a):
    print("\n[1/7] Kernel fibers (3D)")
    grid, raw, genes = fibers.build_kernel_fibers(
        mol, grid_spacing=a.grid_spacing, bandwidth=a.bandwidth,
        min_molecules=a.min_molecules, z_col="z_local",
        z_spacing=a.z_spacing, z_bandwidth=a.z_bandwidth,
        block_size=a.block_size, verbose=True)
    fib = fibers.normalize_fibers(raw); del raw
    print("\n[2/7] Spatial graph (3D)")
    edges = graph.build_spatial_graph(grid, k=a.k_graph, verbose=True)
    print("\n[3/7] Stalks")
    stalks = sheaf.compute_stalks(fib, grid, uniform_dim=a.uniform_dim,
                                  k_nbrs=a.k_nbrs, verbose=True)
    print("\n[4/7] Restriction maps")
    restr = sheaf.compute_restriction_maps(stalks, edges, a.uniform_dim, verbose=True)
    print("\n[5/7] Boundary scores")
    braw, bsm = sheaf.compute_boundary_scores(
        fib, grid, stalks, edges, restr, smooth_radius=a.smooth_radius,
        verbose=True)
    print("\n[6/7] Boundary-aware clustering")
    labels = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=a.n_clusters,
        spatial_weight=a.spatial_weight, boundary_alpha=a.boundary_alpha,
        overcluster_factor=a.overcluster_factor, verbose=True)
    nre = 0
    if not a.no_cleanup:
        print("\n[7/7] Spatial cleanup")
        labels, nre = clustering.spatial_cleanup(
            labels, edges, grid_coords=grid,
            min_fragment_frac=a.min_fragment_frac, verbose=True)
    return dict(grid_coords=grid, fibers_norm=fib, boundary_raw=braw,
                boundary_smooth=bsm, labels=np.asarray(labels), edges=edges,
                n_reassigned=nre, n_points=len(grid), n_genes=len(genes),
                gene_names=genes)


def save_outputs(result, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gc = result["grid_coords"]
    pd.DataFrame({"x": gc[:, 0], "y": gc[:, 1], "z": gc[:, 2],
                  "label": result["labels"]}
                 ).to_csv(os.path.join(out_dir, "domain_labels_3d.csv"),
                          index=False)
    pd.DataFrame({"x": gc[:, 0], "y": gc[:, 1], "z": gc[:, 2],
                  "boundary_raw": result["boundary_raw"],
                  "boundary_smooth": result["boundary_smooth"]}
                 ).to_csv(os.path.join(out_dir, "boundary_field_3d.csv"),
                          index=False)
    pd.DataFrame({"metric": list(metrics.keys()),
                  "value": list(metrics.values())}
                 ).to_csv(os.path.join(out_dir, "metrics_3d.csv"), index=False)
    _plot_zfacets(gc, result["labels"], result["boundary_smooth"], out_dir)
    for f in ("domain_labels_3d.csv", "boundary_field_3d.csv", "metrics_3d.csv",
              "domain_map_3d.png"):
        print(f"  -> {os.path.join(out_dir, f)}")


def _plot_zfacets(gc, labels, bsm, out_dir):
    zvals = np.unique(gc[:, 2])
    ncol = len(zvals) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4 * ncol, 4.2), squeeze=False)
    axes = axes[0]
    uniq = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 2))
    for zi, z in enumerate(zvals):
        ax = axes[zi]
        m = gc[:, 2] == z
        for i, c in enumerate(uniq):
            mm = m & (labels == c)
            if mm.any():
                ax.scatter(gc[mm, 0], gc[mm, 1], s=3, color=cmap(i))
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_title(f"domains @ z={z:.1f}um", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[-1]
    zmid = zvals[len(zvals) // 2]
    m = gc[:, 2] == zmid
    sc = ax.scatter(gc[m, 0], gc[m, 1], c=bsm[m], s=3, cmap="hot",
                    vmax=np.percentile(bsm, 95))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"boundary @ z={zmid:.1f}um", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("3D sheaf boundary detection - Brain Receptor Showcase "
                 "(z-plane facets)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "domain_map_3d.png"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)


def main():
    a = build_parser().parse_args()
    print("=" * 64)
    print(" 3D SHEAF BOUNDARY - Vizgen Brain Receptor Showcase")
    print("=" * 64)
    t0 = time.time()
    mol = load_vizgen_3d(a)
    result = run(mol, a); del mol
    metrics = compute_metrics(result, seed=a.seed, max_pts=5000)
    print("\nSaving outputs...")
    save_outputs(result, metrics, a.output_dir)
    print("\n" + "=" * 64)
    print(f"  Grid points : {result['n_points']:,}  "
          f"({len(np.unique(result['grid_coords'][:,2]))} z levels)")
    print(f"  Clusters    : {len(set(result['labels']))}")
    print(f"  Total time  : {time.time()-t0:.0f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
