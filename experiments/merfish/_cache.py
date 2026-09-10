"""Shared cache for the MERFISH revision experiments.

boundary_eval.py, sheaf_ablation.py and k_sensitivity.py all need the same
intermediates. Building them once costs a few minutes; rebuilding them three
times does not.

Extends the cache that ablation.py already writes with the stalks and the
locally-centered fibers, which the sheaf-specific ablation needs.

    python -m experiments.merfish._cache --stage slice --slice MERFISH_0.19.h5ad
    python -m experiments.merfish._cache --stage build --slice MERFISH_0.19.h5ad
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from spsheaf import fibers, graph, sheaf, evaluation, paths, io

RAW_CSV = "moffitt2018_export_full/molecules_full_add.csv.gz"
GRID_SPACING, BANDWIDTH = 15.0, 25.0
K_GRAPH, K_NBRS, STALK_DIM, SMOOTH_RADIUS = 6, 50, 6, 60.0


def _slice_tag(slice_name):
    return slice_name.replace(".h5ad", "").replace("MERFISH_", "s")


def parquet_path(slice_name):
    return paths.cache("merfish") / f"revision_{_slice_tag(slice_name)}.parquet"


def cache_path(slice_name):
    return paths.cache("merfish") / f"revision_{_slice_tag(slice_name)}.npz"


def _require_parquet_engine():
    """Check before the expensive read, not after.

    stage_slice streams an 848 MB gzipped CSV before it writes anything, so a
    missing parquet engine would otherwise surface several minutes in, with
    the work already thrown away.
    """
    try:
        import pyarrow            # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    try:
        import fastparquet        # noqa: F401
        return
    except ModuleNotFoundError:
        raise SystemExit(
            "No parquet engine available, and the slice cache is written as "
            "parquet.\n"
            "  pip install pyarrow\n"
            "(or reinstall the package, which now declares it: pip install -e .)")


def stage_slice(a):
    _require_parquet_engine()
    src = paths.raw("merfish") / RAW_CSV
    if not src.exists():
        raise SystemExit(
            f"{src} not found.\n"
            f"Expected the Moffitt molecule table under data/raw/merfish/. "
            f"Check paths.toml or SPSHEAF_DATA.")
    cols = ["gene", "x", "y", "z", "cell_z", "matched_h5ad_file", "domain"]
    bregma = -float(a.slice.replace("MERFISH_", "").replace(".h5ad", ""))
    parts = []
    for ch in pd.read_csv(src, usecols=cols, chunksize=2_000_000, low_memory=False):
        if io.MERFISH_INPUT == "section":
            keep = np.isclose(ch.cell_z.to_numpy(dtype=float), bregma)
        else:
            keep = (ch.matched_h5ad_file == a.slice).to_numpy()
        parts.append(ch[keep][["gene", "x", "y", "z", "domain"]])
    df = pd.concat(parts, ignore_index=True)
    if io.MERFISH_INPUT != "section" and df.z.nunique() > 1:
        df = df[df.z == df.z.value_counts().idxmax()]     # legacy: one plane
    df["x_local"] = df.x - df.x.min()
    df["y_local"] = df.y - df.y.min()
    print(f"  input policy: {io.MERFISH_INPUT}")
    out = parquet_path(a.slice)
    df.to_parquet(out)
    print(f"{a.slice}: {len(df):,} molecules, {df.gene.nunique()} genes, "
          f"{df.domain.nunique()} domains -> {out}")


def stage_build(a):
    df = pd.read_parquet(parquet_path(a.slice))
    grid, raw, _ = fibers.build_kernel_fibers(
        df, grid_spacing=GRID_SPACING, bandwidth=BANDWIDTH,
        x_col="x_local", y_col="y_local", verbose=False)
    fib = fibers.normalize_fibers(raw)
    edges = graph.build_spatial_graph(grid, k=K_GRAPH, verbose=False)
    stalks = sheaf.compute_stalks(fib, grid, uniform_dim=STALK_DIM,
                                  k_nbrs=K_NBRS, verbose=False)
    rs = sheaf.compute_restriction_maps(stalks, edges, STALK_DIM, verbose=False)
    braw, bsm = sheaf.compute_boundary_scores(
        fib, grid, stalks, edges, rs, smooth_radius=SMOOTH_RADIUS, verbose=False)
    gt_int, gt_valid, _ = evaluation.map_ground_truth(df, grid)

    # global- and locally-centered fibers: the centering ablation needs both
    fib_global = fib - fib.mean(axis=0)
    fib_local = fib - sheaf.local_means(fib, grid, K_NBRS)   # deterministic ties

    out = cache_path(a.slice)
    np.savez_compressed(
        out,
        grid=grid, fib=fib.astype(np.float32),
        fib_global=fib_global.astype(np.float32),
        fib_local=fib_local.astype(np.float32),
        stalks=np.asarray(stalks, dtype=np.float32),
        edges=np.array(edges), boundary_raw=braw, boundary_smooth=bsm,
        gt_int=gt_int, gt_valid=gt_valid,
        params=np.array([GRID_SPACING, BANDWIDTH, K_GRAPH, K_NBRS,
                         STALK_DIM, SMOOTH_RADIUS]))
    print(f"built {len(grid)} grid points, {len(edges)} edges, "
          f"{int(gt_valid.sum())} with a ground-truth domain -> {out}")


def load(slice_name):
    p = cache_path(slice_name)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Run:\n"
            f"  python -m experiments.merfish._cache --stage slice --slice {slice_name}\n"
            f"  python -m experiments.merfish._cache --stage build --slice {slice_name}")
    d = np.load(p, allow_pickle=True)
    return {
        "grid": d["grid"], "fib": d["fib"].astype(float),
        "fib_global": d["fib_global"].astype(float),
        "fib_local": d["fib_local"].astype(float),
        "stalks": [s for s in d["stalks"].astype(float)],
        "edges": [tuple(e) for e in d["edges"]],
        "boundary_raw": d["boundary_raw"], "boundary_smooth": d["boundary_smooth"],
        "gt_int": d["gt_int"], "gt_valid": d["gt_valid"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["slice", "build"])
    ap.add_argument("--slice", default="MERFISH_0.19.h5ad")
    a = ap.parse_args()
    {"slice": stage_slice, "build": stage_build}[a.stage](a)


if __name__ == "__main__":
    main()
