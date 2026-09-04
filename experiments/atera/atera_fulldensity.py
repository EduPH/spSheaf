#!/usr/bin/env python
"""
Full-density confirmatory rasteriser for the marker-aware Atera run.

Instead of subsampling molecules, this streams the ENTIRE section and
accumulates, per gene, a 2D histogram at grid resolution (no subsampling of any
gene). Memory is bounded by the raster stack (n_genes x ny x nx), not by the
molecule count, so all ~740M molecules contribute. The gene set and grid
geometry are taken from an existing 1%-run fibers.npz so the two are directly
comparable. Resumable per row group (budget-limited per call).

When done, emits fibers.npz in the SAME format as atera_markeraware --phase
fibers, so `--phase cluster` / `--phase report` run on it unchanged.
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--ref-npz", required=True, help="fibers.npz from the 1% run (gene set + grid)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bandwidth", type=float, default=100.0)
    p.add_argument("--min-density", type=float, default=0.5)
    p.add_argument("--min-qv", type=float, default=20.0)
    p.add_argument("--budget", type=float, default=34.0)
    return p


def main():
    a = build_parser().parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ref = np.load(a.ref_npz, allow_pickle=True)
    genes = list(ref["genes"]); gidx = {g: i for i, g in enumerate(genes)}
    x0, y0, gs = float(ref["x0"]), float(ref["y0"]), float(ref["gridspacing"])
    ny, nx = ref["mask"].shape
    ras_path = os.path.join(a.out_dir, "rasters_partial.npy")
    prog_path = os.path.join(a.out_dir, "fd_progress.json")
    if os.path.exists(ras_path):
        rasters = np.load(ras_path)
        prog = json.load(open(prog_path))
    else:
        rasters = np.zeros((len(genes), ny, nx), np.float64)
        prog = {"next_rg": 0}

    pf = pq.ParquetFile(a.input); n_rg = pf.num_row_groups
    cols = ["feature_name", "x_location", "y_location", "qv", "is_gene"]
    t0 = time.time(); rg = prog["next_rg"]
    while rg < n_rg and time.time() - t0 < a.budget:
        d = pf.read_row_group(rg, columns=cols).to_pandas()
        d = d[(d.qv >= a.min_qv) & (d.is_gene)]
        gi = d.feature_name.astype(str).map(gidx)
        d = d[gi.notna()]; gi = gi[gi.notna()].astype(int).to_numpy()
        ix = np.clip(((d.x_location.values - x0) / gs).astype(int), 0, nx - 1)
        iy = np.clip(((d.y_location.values - y0) / gs).astype(int), 0, ny - 1)
        np.add.at(rasters, (gi, iy, ix), 1.0)
        rg += 1
    np.save(ras_path, rasters)
    prog["next_rg"] = rg; prog["done"] = rg >= n_rg
    json.dump(prog, open(prog_path, "w"))
    print(f"row groups {rg}/{n_rg} ({100*rg/n_rg:.1f}%); done={prog['done']}")
    if not prog["done"]:
        return

    # finalize: smooth + per-gene z-standardise -> fibers.npz (markeraware format)
    sig = a.bandwidth / gs
    fields = np.stack([gaussian_filter(rasters[i].astype(np.float32), sig) for i in range(len(genes))])
    total = fields.sum(0); mask = total >= a.min_density
    yy, xx = np.where(mask)
    grid = np.column_stack([x0 + xx * gs, y0 + yy * gs]).astype(float)
    F = fields[:, yy, xx].T
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    np.savez(os.path.join(a.out_dir, "fibers.npz"),
             grid=grid, fib=Z.astype(np.float32), genes=np.array(genes),
             fields=fields.astype(np.float32), mask=mask, x0=x0, y0=y0, gridspacing=gs)
    print(f"DONE full-density: {len(grid)} grid points, {len(genes)} genes -> fibers.npz")


if __name__ == "__main__":
    main()
