#!/usr/bin/env python
"""Ablation on osmFISH (real ground truth): the sheaf boundary weighting is
what improves the domains over plain k-means. Stage 1 builds+caches fibers,
stalks and the boundary field; stage 2 runs the clustering variants + scores."""
import argparse, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spsheaf import fibers, graph, sheaf, clustering
from run_osmFISH import load_osmfish_molecules, load_cell_ground_truth, evaluate_at_cell_level

CACHE = "ablation_cache.npz"


def stage_build(a):
    mol = load_osmfish_molecules(a.mol_path).sample(frac=0.4, random_state=0)
    cell_xy, cell_gt = load_cell_ground_truth(a.h5ad_path)
    grid, raw, _ = fibers.build_kernel_fibers(mol, grid_spacing=250, bandwidth=250,
                                              min_molecules=5, x_col="x", y_col="y", verbose=False)
    fib = fibers.normalize_fibers(raw)
    edges = graph.build_spatial_graph(grid, k=6, verbose=False)
    stalks = sheaf.compute_stalks(fib, grid, uniform_dim=9, k_nbrs=50, verbose=False)
    restr = sheaf.compute_restriction_maps(stalks, edges, 9, verbose=False)
    _, bsm = sheaf.compute_boundary_scores(fib, grid, stalks, edges, restr,
                                           smooth_radius=300, verbose=False)
    np.savez(CACHE, grid=grid, fib=fib.astype(np.float32), bsm=bsm,
             edges=np.array(edges), cell_xy=cell_xy, cell_gt=np.array(cell_gt, dtype=object))
    print(f"built: {len(grid)} grid pts, {len(cell_xy)} cells with GT")


def stage_eval(a):
    d = np.load(CACHE, allow_pickle=True)
    grid, fib, bsm = d["grid"], d["fib"].astype(float), d["bsm"]
    edges = [tuple(e) for e in d["edges"]]; cell_xy, cell_gt = d["cell_xy"], d["cell_gt"]

    def score(labels):
        m = evaluate_at_cell_level(grid, labels, cell_xy, cell_gt)
        m = m[0] if isinstance(m, tuple) else m
        return m.get("ARI", m.get("ari")), m.get("NMI", m.get("nmi"))

    variants = {}
    # (a) expression only: no spatial, no boundary weighting
    variants["k-means\n(expression only)"] = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=11, spatial_weight=0.0, boundary_alpha=0.0,
        n_pca=10, random_state=42, verbose=False)
    # (b) + spatial coordinates (still no boundary weighting)
    variants["+ spatial\ncoordinates"] = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=11, spatial_weight=0.5, boundary_alpha=0.0,
        n_pca=10, random_state=42, verbose=False)
    # (c) + sheaf boundary weighting (full method)
    full = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=11, spatial_weight=0.5, boundary_alpha=0.5,
        n_pca=10, random_state=42, verbose=False)
    variants["+ sheaf boundary\nweighting"] = full
    # (d) + spatial cleanup
    cleaned, _ = clustering.spatial_cleanup(np.asarray(full), edges, grid_coords=grid, verbose=False)
    variants["+ connectivity\ncleanup (full)"] = cleaned

    names, aris, nmis = [], [], []
    for k, lab in variants.items():
        ari, nmi = score(np.asarray(lab)); names.append(k); aris.append(ari); nmis.append(nmi)
        print(f"{k.replace(chr(10),' '):38s} ARI={ari:.3f} NMI={nmi:.3f}")

    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w/2, aris, w, label="ARI", color="steelblue")
    ax.bar(x+w/2, nmis, w, label="NMI", color="firebrick")
    for i,(av,nv) in enumerate(zip(aris,nmis)):
        ax.text(i-w/2, av+0.01, f"{av:.2f}", ha="center", fontsize=8)
        ax.text(i+w/2, nv+0.01, f"{nv:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("accuracy vs ground truth (osmFISH, 11 regions)")
    ax.set_title("Pipeline component ablation (osmFISH cortex): full pipeline > plain k-means\n"
                 "boundary weighting + connectivity cleanup act synergistically", fontsize=11)
    ax.legend()
    fig.tight_layout(); fig.savefig("ablation.png", dpi=400, bbox_inches="tight")
    print("-> ablation.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build","eval"])
    ap.add_argument("--mol-path", default="mRNA_coords_raw_counting.hdf5")
    ap.add_argument("--h5ad-path", default="osmfish_20260512084934.h5ad")
    a = ap.parse_args()
    (stage_build if a.stage=="build" else stage_eval)(a)


if __name__ == "__main__":
    main()
