#!/usr/bin/env python
"""
Checkpointed driver for the Brain Receptor Showcase run.

The interactive sandbox caps each shell call at ~45s and does not persist
background processes, so the pipeline is split into resumable stages that each
finish under the limit and save progress to <output>/_ckpt/. Re-running a stage
resumes where it left off. Stages:

    load    stream CSV -> reduced molecule parquet   (z-plane, top genes, subsample)
    fibers  build kernel fibers in time-budgeted blocks (resumable)
    sheaf   graph + stalks + restrictions + boundary scores
    cluster boundary-aware clustering + cleanup + metrics + plots -> final outputs

Usage:  python _staged_run.py <stage>
This is a runner only; all algorithm code lives in run_brain_receptor_showcase.py
and the spsheaf package.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import run_brain_receptor_showcase as R
from spsheaf import fibers as F, graph as G, sheaf as S, clustering as C

# ---- fixed run configuration (coarse whole-section) ----
INPUT = "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_detected_transcripts_S1R1.csv"
OUT = "output_brain_receptor"
CK = os.path.join(OUT, "_ckpt")
os.makedirs(CK, exist_ok=True)

CFG = dict(
    z_plane=0.0, top_genes=300, subsample=0.25, chunksize=6_000_000, seed=0,
    grid_spacing=50.0, bandwidth=100.0, smooth_radius=200.0, min_molecules=15.0,
    block_size=1500, k_graph=6, k_nbrs=50, uniform_dim=6,
    n_clusters=15, spatial_weight=0.5, boundary_alpha=0.5,
    overcluster_factor=1.0, min_fragment_frac=0.01,
)
BUDGET = 34.0  # seconds of work before checkpointing


class Args:  # duck-typed args for R.run_pipeline helpers
    pass


def _args():
    a = Args()
    for k, v in CFG.items():
        setattr(a, k, v)
    a.no_cleanup = False
    return a


def stage_load():
    p = os.path.join(CK, "mol.parquet")
    if os.path.exists(p):
        print("load: already done ->", p)
        return
    mol = R.load_vizgen_csv(
        INPUT, x_col="global_x", y_col="global_y", z_col="global_z",
        gene_col="gene", z_plane=CFG["z_plane"], top_genes=CFG["top_genes"],
        subsample=CFG["subsample"], chunksize=CFG["chunksize"], seed=CFG["seed"],
    )
    mol[["gene", "x_local", "y_local"]].to_parquet(p, index=False)
    print(f"load: saved {len(mol):,} molecules -> {p}")


def stage_fibers():
    final = os.path.join(CK, "fibers_final.npz")
    if os.path.exists(final):
        print("fibers: already done ->", final)
        return
    mol = pd.read_parquet(os.path.join(CK, "mol.parquet"))
    gene_names = sorted(mol["gene"].unique())
    g2i = {g: i for i, g in enumerate(gene_names)}
    coords = mol[["x_local", "y_local"]].values.astype(np.float64)
    gidx = mol["gene"].map(g2i).values
    n_genes = len(gene_names)

    ck = os.path.join(CK, "fibers_prog.npz")
    if os.path.exists(ck):
        d = np.load(ck, allow_pickle=True)
        grid = d["grid"]; fib = d["fib"]; tw = d["tw"]; start = int(d["start"])
        print(f"fibers: resume at block start {start:,}/{len(grid):,}")
    else:
        gs = CFG["grid_spacing"]
        x0, x1 = coords[:, 0].min(), coords[:, 0].max()
        y0, y1 = coords[:, 1].min(), coords[:, 1].max()
        xg = np.arange(x0 + gs / 2, x1, gs)
        yg = np.arange(y0 + gs / 2, y1, gs)
        xx, yy = np.meshgrid(xg, yg)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        fib = np.zeros((len(grid), n_genes), dtype=np.float32)
        tw = np.zeros(len(grid), dtype=np.float32)
        start = 0
        print(f"fibers: init {len(grid):,} grid points, {n_genes} genes")

    tree = cKDTree(coords)
    radius = 3.0 * CFG["bandwidth"]
    bs = CFG["block_size"]
    t0 = time.time()
    n_grid = len(grid)
    while start < n_grid and (time.time() - t0) < BUDGET:
        end = min(start + bs, n_grid)
        nbrs = tree.query_ball_point(grid[start:end], r=radius)
        for li, nb in enumerate(nbrs):
            if not nb:
                continue
            gi = start + li
            nb = np.asarray(nb)
            dd = np.linalg.norm(coords[nb] - grid[gi], axis=1)
            w = np.exp(-0.5 * (dd / CFG["bandwidth"]) ** 2)
            np.add.at(fib[gi], gidx[nb], w)
            tw[gi] = w.sum()
        start = end
    if start < n_grid:
        np.savez(ck, grid=grid, fib=fib, tw=tw, start=start)
        print(f"fibers: PROGRESS {start:,}/{n_grid:,} "
              f"({100*start/n_grid:.0f}%) — rerun to continue")
        return
    keep = tw >= CFG["min_molecules"]
    grid_k = grid[keep]
    fib_norm = F.normalize_fibers(fib[keep])
    np.savez(final, grid=grid_k, fib_norm=fib_norm,
             gene_names=np.array(gene_names, dtype=object))
    print(f"fibers: DONE {len(grid_k):,} valid grid points -> {final}")


def stage_sheaf():
    out = os.path.join(CK, "sheaf.npz")
    if os.path.exists(out):
        print("sheaf: already done ->", out)
        return
    d = np.load(os.path.join(CK, "fibers_final.npz"), allow_pickle=True)
    grid = d["grid"]; fib_norm = d["fib_norm"]
    a = _args()
    edges = G.build_spatial_graph(grid, k=a.k_graph, verbose=True)
    stalks = S.compute_stalks(fib_norm, grid, uniform_dim=a.uniform_dim,
                              k_nbrs=a.k_nbrs, verbose=True)
    restr = S.compute_restriction_maps(stalks, edges, a.uniform_dim, verbose=True)
    braw, bsm = S.compute_boundary_scores(
        fib_norm, grid, stalks, edges, restr,
        smooth_radius=a.smooth_radius, verbose=True)
    np.savez(out, edges=np.asarray(edges), braw=braw, bsm=bsm)
    print(f"sheaf: DONE -> {out}")


def stage_cluster():
    d = np.load(os.path.join(CK, "fibers_final.npz"), allow_pickle=True)
    grid = d["grid"]; fib_norm = d["fib_norm"]
    gene_names = list(d["gene_names"])
    s = np.load(os.path.join(CK, "sheaf.npz"))
    edges = [tuple(e) for e in s["edges"]]
    braw = s["braw"]; bsm = s["bsm"]
    a = _args()
    labels = C.boundary_aware_clustering(
        fib_norm, grid, bsm, edges, n_clusters=a.n_clusters,
        spatial_weight=a.spatial_weight, boundary_alpha=a.boundary_alpha,
        overcluster_factor=a.overcluster_factor, verbose=True)
    labels, nre = C.spatial_cleanup(
        labels, edges, grid_coords=grid,
        min_fragment_frac=a.min_fragment_frac, verbose=True)
    result = dict(
        grid_coords=grid, fibers_norm=fib_norm, boundary_raw=braw,
        boundary_smooth=bsm, labels=np.asarray(labels), edges=edges,
        n_reassigned=nre, n_points=len(grid), n_genes=len(gene_names),
        gene_names=gene_names)
    metrics = R.compute_metrics(result, seed=CFG["seed"], max_pts=5000)
    R.save_outputs(result, metrics, OUT)
    print("cluster: DONE")


STAGES = {"load": stage_load, "fibers": stage_fibers,
          "sheaf": stage_sheaf, "cluster": stage_cluster}

if __name__ == "__main__":
    st = sys.argv[1]
    t = time.time()
    STAGES[st]()
    print(f"[{st}] wall {time.time()-t:.0f}s")
