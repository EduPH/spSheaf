"""Cache one SpaceTx section in the format ``experiments/merfish/sheaf_ablation.py``
reads, so the ten-arm edge-score ablation can be run on a section where the
tissue coordinate reference is the selected one.

Fibers, graph, stalks, restriction maps and the boundary field are built at
the section's selected configuration (pass it on the command line); the
annotation is carried to the grid by the nearest registered cell, registration
being the same label-free translation ``run_spacetx.py`` fits.

    SPSHEAF_REFERENCE=tissue python -m experiments.spacetx.ablation_cache \
        MERFISH_Allen_Prototype --bandwidth 40 --smooth-radius 120 --uniform-dim 6
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spsheaf import fibers, graph, sheaf, paths, io
from experiments.spacetx.run_spacetx import build_parser, load, register


def main():
    p = build_parser()
    p.add_argument("--k-nbrs", type=int, default=50)
    a = p.parse_args()
    if a.reference is not None:
        sheaf.REFERENCE = a.reference
    mol, cxy, cid, gt, cbg = load(a)
    valid_cell = pd.notna(gt)
    names = sorted(set(gt[valid_cell]))
    mol_s = io.select_slice(mol, f"spacetx_{a.dataset}", verbose=False)
    grid, raw, _ = fibers.build_kernel_fibers(
        mol_s, grid_spacing=a.grid_spacing, bandwidth=a.bandwidth,
        x_col="x_local", y_col="y_local", verbose=False)
    fib = fibers.normalize_fibers(raw)
    edges = graph.build_spatial_graph(grid, k=6, verbose=False)
    stalks = sheaf.compute_stalks(fib, grid, uniform_dim=a.uniform_dim,
                                  k_nbrs=a.k_nbrs, verbose=False)
    rs = sheaf.compute_restriction_maps(stalks, edges, a.uniform_dim, verbose=False)
    braw, bsm = sheaf.compute_boundary_scores(
        fib, grid, stalks, edges, rs, smooth_radius=a.smooth_radius, verbose=False)

    t, r, r0 = register(mol, cxy, cid, cbg, a.register_radius, grid)
    cells_al = cxy + t
    tree = cKDTree(cells_al[valid_cell])
    d, idx = tree.query(grid, k=1)
    gti_cells = np.array([names.index(g) for g in gt[valid_cell]])
    gt_int = np.where(d <= 2 * a.grid_spacing, gti_cells[idx], -1)
    gt_valid = gt_int >= 0

    fib_global = fib - fib.mean(axis=0)
    fib_local = fib - sheaf.local_means(fib, grid, a.k_nbrs)
    out = paths.cache("spacetx") / f"ablation_{a.dataset}.npz"
    np.savez_compressed(
        out, grid=grid, fib=fib.astype(np.float32),
        fib_global=fib_global.astype(np.float32), fib_local=fib_local.astype(np.float32),
        stalks=np.asarray(stalks, dtype=np.float32), edges=np.array(edges),
        boundary_raw=braw, boundary_smooth=bsm, gt_int=gt_int, gt_valid=gt_valid,
        params=np.array([a.grid_spacing, a.bandwidth, 6, a.k_nbrs, a.uniform_dim,
                         a.smooth_radius]))
    print(f"{a.dataset} ({sheaf.REFERENCE} reference): {len(grid)} grid points, "
          f"{len(edges)} edges, {int(gt_valid.sum())} annotated, registration r "
          f"{r0:.3f} -> {r:.3f}; -> {out}")


if __name__ == "__main__":
    main()
