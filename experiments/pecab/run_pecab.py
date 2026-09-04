#!/usr/bin/env python
"""
Segmentation-free spatial-domain inference on PECAb (protein) data.

PECAb (Takei et al.) delivers a *cell-level* table: each row is a segmented
cell with a centroid ``(x, y)`` and a dense vector of multiplexed protein
intensities (35 - 220 antibodies). This is exactly the continuous
feature-fiber input contract of the method (cf. the DNAseqFISH+ chromosome
fibers) — no molecule counting, the proteins ARE the fiber. So the same sheaf
boundary-detection pipeline runs unchanged, only the fiber source differs.

We build a regular grid over the tissue, set each grid point's fiber to the
Gaussian-kernel-weighted mean of nearby cells' protein vectors, and run the
sheaf pipeline to carve **protein-state spatial domains**. Per domain we report
the proteins that most define it (domain mean vs global mean, in fiber space),
plus unsupervised quality metrics. No ground truth is assumed.

The headline dataset is the human **carcinosarcoma** tissue section
(6_Carcinosarcoma_33x220plex), whose defining biology is an epithelial
(E-Cadherin / EpCAM) vs mesenchymal (N-Cadherin / VIM / a-SMA / FN1) axis —
a real spatial-domain structure the method should recover directly from the
protein field.

Usage
-----
    python run_pecab.py \
        --input .../6_Carcinosarcoma_33x220plex/Raw/6_raw_SiPECAb_Rep1.csv \
        --n-domains 7 --output-dir output_pecab_carcinosarcoma
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)

from spsheaf import fibers, graph, sheaf, clustering

# columns present in every PECAb raw table that are NOT proteins
META_COLS = ["tileID", "cellID", "y", "x", "orientation", "majoraxis",
             "minoraxis", "cell", "rep", "UMAP1", "UMAP2", "cluster"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="PECAb cell-level CSV")
    p.add_argument("--output-dir", default="output_pecab")
    p.add_argument("--x-col", default="x")
    p.add_argument("--y-col", default="y")
    p.add_argument("--meta-cols", default=None,
                   help="comma list of non-protein columns to drop "
                        "(default: the standard PECAb metadata set)")
    p.add_argument("--proteins", default=None,
                   help="explicit comma list of protein columns to use "
                        "(overrides meta-col exclusion)")
    # grid geometry is in the CSV coordinate units (PECAb = camera pixels).
    p.add_argument("--grid-spacing", type=float, default=150.0)
    p.add_argument("--bandwidth", type=float, default=300.0)
    p.add_argument("--smooth-radius", type=float, default=450.0)
    p.add_argument("--k-graph", type=int, default=6)
    p.add_argument("--k-nbrs", type=int, default=40)
    p.add_argument("--uniform-dim", type=int, default=6)
    p.add_argument("--n-domains", type=int, default=7)
    p.add_argument("--min-cells-per-grid", type=int, default=3)
    p.add_argument("--log1p", action="store_true",
                   help="log1p the intensities before fibre building "
                        "(fibers already z-standardises).")
    p.add_argument("--random-state", type=int, default=42)
    return p


def protein_columns(df, a):
    if a.proteins:
        return [c for c in a.proteins.split(",") if c in df.columns]
    meta = (a.meta_cols.split(",") if a.meta_cols else META_COLS)
    cols = [c for c in df.columns if c not in meta]
    # keep only numeric
    return [c for c in cols if np.issubdtype(df[c].dtype, np.number)]


def main():
    a = build_parser().parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    df = pd.read_csv(a.input)
    prot = protein_columns(df, a)
    df = df.dropna(subset=[a.x_col, a.y_col]).reset_index(drop=True)
    coords = df[[a.x_col, a.y_col]].to_numpy(float)
    feats = df[prot].to_numpy(float)
    feats = np.nan_to_num(feats, nan=0.0)
    if a.log1p:
        feats = np.log1p(feats - np.nanmin(feats, axis=0, keepdims=True))
    print(f"{len(df)} cells x {len(prot)} proteins; "
          f"extent x {coords[:,0].min():.0f}..{coords[:,0].max():.0f} "
          f"y {coords[:,1].min():.0f}..{coords[:,1].max():.0f}")

    # 2D feature fibres (kernel-weighted mean protein vector per grid point)
    grid, fib = fibers.build_feature_fibers(
        coords, feats, grid_spacing=a.grid_spacing, bandwidth=a.bandwidth,
        standardize=True, verbose=True)

    # drop grid points that aggregate too few cells (edges / empty tissue)
    tree = cKDTree(coords)
    n_near = np.array([len(tree.query_ball_point(g, a.bandwidth)) for g in grid])
    keep = n_near >= a.min_cells_per_grid
    grid, fib = grid[keep], fib[keep]
    print(f"grid points: {len(grid)} (>= {a.min_cells_per_grid} cells within bandwidth)")

    ud = min(a.uniform_dim, fib.shape[1] - 1)
    edges = graph.build_spatial_graph(grid, k=a.k_graph, verbose=False)
    stalks = sheaf.compute_stalks(fib, grid, uniform_dim=ud,
                                  k_nbrs=min(a.k_nbrs, len(grid) - 1), verbose=False)
    restr = sheaf.compute_restriction_maps(stalks, edges, ud, verbose=False)
    braw, bsm = sheaf.compute_boundary_scores(fib, grid, stalks, edges, restr,
                                              smooth_radius=a.smooth_radius,
                                              verbose=False)
    labels = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=a.n_domains,
        n_pca=min(10, fib.shape[1]), random_state=a.random_state, verbose=False)
    labels, _ = clustering.spatial_cleanup(labels, edges, grid_coords=grid,
                                           verbose=False)
    labels = np.asarray(labels)
    dom_order = sorted(set(labels))
    print(f"{len(dom_order)} domains: sizes "
          f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    # ---- per-domain marker proteins (domain mean - global mean, fibre space)
    gmean = fib.mean(axis=0)
    rows = []
    for d in dom_order:
        dm = fib[labels == d].mean(axis=0)
        enr = dm - gmean
        rank = np.argsort(-enr)
        for r, gi in enumerate(rank[:12]):
            rows.append(dict(domain=int(d), rank=r + 1, protein=prot[gi],
                             enrichment=round(float(enr[gi]), 3)))
    markers = pd.DataFrame(rows)
    markers.to_csv(os.path.join(a.output_dir, "domain_markers.csv"), index=False)

    # ---- unsupervised metrics
    within = sum(labels[i] == labels[j] for i, j in edges)
    coherence = within / max(len(edges), 1)
    met = {"n_domains": len(dom_order), "n_gridpoints": len(grid),
           "n_cells": len(df), "n_proteins": len(prot),
           "spatial_coherence": round(coherence, 4)}
    if len(dom_order) > 1:
        met["silhouette"] = round(float(silhouette_score(fib, labels)), 4)
        met["calinski_harabasz"] = round(float(calinski_harabasz_score(fib, labels)), 1)
        met["davies_bouldin"] = round(float(davies_bouldin_score(fib, labels)), 4)
    pd.DataFrame([met]).to_csv(os.path.join(a.output_dir, "metrics.csv"), index=False)

    # ---- outputs
    pd.DataFrame({"x": grid[:, 0], "y": grid[:, 1], "domain": labels}).to_csv(
        os.path.join(a.output_dir, "domain_labels.csv"), index=False)
    pd.DataFrame({"x": grid[:, 0], "y": grid[:, 1], "boundary": bsm}).to_csv(
        os.path.join(a.output_dir, "boundary_field.csv"), index=False)
    _plot(grid, labels, bsm, markers, dom_order, a.output_dir)

    print("\nmetrics:", met)
    print("\ntop marker proteins per domain:")
    for d in dom_order:
        top = markers[markers.domain == d].head(6).protein.tolist()
        print(f"  domain {d}: {', '.join(top)}")
    print(f"\n-> {a.output_dir}/ (domain_map.png, domain_markers.csv, metrics.csv)")


def _plot(grid, labels, bsm, markers, dom_order, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(20, 9))
    palette = {d: plt.cm.tab10(i % 10) for i, d in enumerate(dom_order)}
    for d in dom_order:
        m = labels == d
        top = markers[markers.domain == d].head(3).protein.tolist()
        ax[0].scatter(grid[m, 0], grid[m, 1], s=10, color=palette[d],
                      label=f"{d}: {', '.join(top)}")
    ax[0].set_aspect("equal"); ax[0].invert_yaxis()
    ax[0].set_title("Protein-state spatial domains")
    ax[0].legend(markerscale=2, fontsize=8, loc="center left",
                 bbox_to_anchor=(1.0, 0.5))
    sc = ax[1].scatter(grid[:, 0], grid[:, 1], s=10, c=bsm, cmap="magma")
    ax[1].set_aspect("equal"); ax[1].invert_yaxis()
    ax[1].set_title("Sheaf boundary field")
    fig.colorbar(sc, ax=ax[1], shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "domain_map.png"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
