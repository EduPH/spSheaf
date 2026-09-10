#!/usr/bin/env python
"""
Ablation on the MERFISH benchmark (135 genes -> strong boundary signal), the
setting where the sheaf boundary weighting matters most.

  --stage cache   stream molecules_full_add.csv.gz, keep one slice + needed
                  columns -> slice parquet (fast: usecols streaming)
  --stage build   build fibers/stalks/boundary + grid-level ground truth -> npz
  --stage eval    plain k-means vs +spatial vs +sheaf boundary weighting
                  vs +cleanup, scored vs the 8 ground-truth domains
"""
import argparse, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from spsheaf import fibers, graph, sheaf, clustering, evaluation, io

CSV = "moffitt2018_export_full/molecules_full_add.csv.gz"
PARQ = "ablation_slice.parquet"
CACHE = "ablation_cache.npz"


def stage_cache(a):
    cols = ["gene", "x", "y", "z", "cell_z", "matched_h5ad_file", "domain"]
    bregma = -float(a.slice.replace("MERFISH_", "").replace(".h5ad", ""))
    parts = []
    for ch in pd.read_csv(CSV, usecols=cols, chunksize=2_000_000, low_memory=False):
        if io.MERFISH_INPUT == "section":       # whole section, all planes
            keep = np.isclose(ch.cell_z.to_numpy(dtype=float), bregma)
        else:
            keep = (ch.matched_h5ad_file == a.slice).to_numpy()
        parts.append(ch[keep][["gene", "x", "y", "z", "domain"]])
    df = pd.concat(parts, ignore_index=True)
    if io.MERFISH_INPUT != "section" and df.z.nunique() > 1:   # legacy: one plane
        df = df[df.z == df.z.value_counts().idxmax()]
    df["x_local"] = df.x - df.x.min(); df["y_local"] = df.y - df.y.min()
    df.to_parquet(PARQ)
    print(f"cached {a.slice}: {len(df):,} molecules, {df.gene.nunique()} genes, "
          f"{df.domain.nunique()} domains -> {PARQ}")


def stage_build(a):
    df = pd.read_parquet(PARQ)
    grid, raw, _ = fibers.build_kernel_fibers(df, grid_spacing=15, bandwidth=25,
                                              x_col="x_local", y_col="y_local", verbose=False)
    fib = fibers.normalize_fibers(raw)
    edges = graph.build_spatial_graph(grid, k=6, verbose=False)
    st = sheaf.compute_stalks(fib, grid, uniform_dim=6, k_nbrs=50, verbose=False)
    rs = sheaf.compute_restriction_maps(st, edges, 6, verbose=False)
    _, bsm = sheaf.compute_boundary_scores(fib, grid, st, edges, rs, smooth_radius=60, verbose=False)
    gt_int, gt_valid, _ = evaluation.map_ground_truth(df, grid)
    np.savez(CACHE, grid=grid, fib=fib.astype(np.float32), bsm=bsm,
             edges=np.array(edges), gt_int=gt_int, gt_valid=gt_valid)
    print(f"built: {len(grid)} grid pts, {int(gt_valid.sum())} with GT domain")


def stage_eval(a):
    d = np.load(CACHE, allow_pickle=True)
    grid, fib, bsm = d["grid"], d["fib"].astype(float), d["bsm"]
    edges = [tuple(e) for e in d["edges"]]; gt, gv = d["gt_int"], d["gt_valid"]
    def score(lab):
        lab = np.asarray(lab)
        return ARI(gt[gv], lab[gv]), NMI(gt[gv], lab[gv])
    V = {}
    V["k-means\n(expression only)"] = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=8, spatial_weight=0.0, boundary_alpha=0.0,
        n_pca=10, random_state=42, verbose=False)
    V["+ spatial\ncoordinates"] = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=8, spatial_weight=0.5, boundary_alpha=0.0,
        n_pca=10, random_state=42, verbose=False)
    full = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=8, spatial_weight=0.5, boundary_alpha=0.5,
        n_pca=10, random_state=42, verbose=False)
    V["+ sheaf boundary\nweighting"] = full
    cl, _ = clustering.spatial_cleanup(np.asarray(full), edges, grid_coords=grid, verbose=False)
    V["+ connectivity\ncleanup (full)"] = cl

    names, aris, nmis = [], [], []
    for k, lab in V.items():
        a_, n_ = score(lab); names.append(k); aris.append(a_); nmis.append(n_)
        print(f"{k.replace(chr(10),' '):36s} ARI={a_:.3f} NMI={n_:.3f}")
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w/2, aris, w, label="ARI", color="steelblue")
    ax.bar(x+w/2, nmis, w, label="NMI", color="firebrick")
    for i,(av,nv) in enumerate(zip(aris,nmis)):
        ax.text(i-w/2, av+0.01, f"{av:.2f}", ha="center", fontsize=8)
        ax.text(i+w/2, nv+0.01, f"{nv:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("accuracy vs ground truth (MERFISH, 8 domains)")
    ax.set_title("Pipeline component ablation on MERFISH (135 genes)")
    ax.legend(); fig.tight_layout(); fig.savefig("ablation.png", dpi=400, bbox_inches="tight")
    print("-> ablation.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cache","build","eval"])
    ap.add_argument("--slice", default="MERFISH_0.19.h5ad")
    a = ap.parse_args()
    {"cache":stage_cache,"build":stage_build,"eval":stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
