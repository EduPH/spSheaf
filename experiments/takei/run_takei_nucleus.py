#!/usr/bin/env python
"""
Phase-1 skeleton: 3D sheaf nuclear-zone detection from Takei DNA seqFISH+.

Treats each nucleus's locus point cloud (x, y, z, locus_id) exactly like the
tissue pipeline treats (x, y, gene): locus identity plays the role of "gene",
and a kernel-smoothed *locus-composition* field is the fiber. The N-D pipeline
in ``spsheaf`` (3D grid + anisotropic kernel, graph, stalks, restriction
maps, disagreement, boundary smoothing, clustering) is used unchanged.

IMPORTANT — schema: the exact column names in the Zenodo ``DNAseqFISH+`` files
are confirmed by running with ``--inspect`` first. The defaults below are the
common seqFISH+ convention; override with --cell-col / --x-col / --y-col /
--z-col / --locus-col once you've looked.

Usage
-----
    # 0) look at the file structure first
    python run_takei_nucleus.py --input dna_seqfish/<file>.csv --inspect

    # 1) run one (or a few) nuclei
    python run_takei_nucleus.py --input dna_seqfish/<file>.csv \
        --cell-col cellID --x-col x --y-col y --z-col z --locus-col geneID \
        --max-cells 3 --output-dir output_takei
"""
import argparse
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spsheaf import fibers, graph, sheaf, clustering


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="A DNAseqFISH+ CSV, or a folder of them.")
    p.add_argument("--inspect", action="store_true",
                   help="Print columns / head / cell + locus counts and exit.")
    p.add_argument("--output-dir", default="output_takei")
    # --- schema (Takei DNAseqFISH+ 1Mb/25kb files) ---
    p.add_argument("--fov-col", default="fov",
                   help="Field-of-view column; a nucleus is (fov, cellID).")
    p.add_argument("--cell-col", default="cellID")
    p.add_argument("--x-col", default="x")
    p.add_argument("--y-col", default="y")
    p.add_argument("--z-col", default="z")
    p.add_argument("--locus-col", default="geneID",
                   help="Locus identity, e.g. 'chr1-#1'.")
    p.add_argument("--chrom-level", action="store_true",
                   help="Aggregate loci to chromosome (chr1-#1 -> chr1): a "
                        "dense, low-dim (~20) fiber. Recommended first pass; "
                        "the per-locus fiber is ~2,460-dim and near "
                        "single-copy (sparse/noisy).")
    # --- voxel -> nm ---
    p.add_argument("--xy-nm", type=float, default=103.0)
    p.add_argument("--z-nm", type=float, default=250.0)
    # --- subnuclear grid / kernel (nm) ---
    p.add_argument("--grid-nm", type=float, default=500.0)
    p.add_argument("--bandwidth-nm", type=float, default=700.0)
    p.add_argument("--z-grid-nm", type=float, default=400.0)
    p.add_argument("--z-bandwidth-nm", type=float, default=600.0)
    p.add_argument("--min-molecules", type=float, default=3.0)
    p.add_argument("--block-size", type=int, default=2000)
    # --- graph / stalks / clustering ---
    p.add_argument("--k-graph", type=int, default=6)
    p.add_argument("--k-nbrs", type=int, default=40)
    p.add_argument("--uniform-dim", type=int, default=6)
    p.add_argument("--n-zones", type=int, default=6)
    p.add_argument("--spatial-weight", type=float, default=0.5)
    p.add_argument("--boundary-alpha", type=float, default=0.5)
    p.add_argument("--smooth-radius-nm", type=float, default=500.0)
    # --- cell selection ---
    p.add_argument("--min-cell-loci", type=int, default=500)
    p.add_argument("--max-cells", type=int, default=3)
    p.add_argument("--cells", default=None,
                   help="Comma-separated cell ids to run (overrides max-cells).")
    return p


def load_table(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise SystemExit(f"No CSVs found in {path}")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return pd.read_csv(path)


def inspect(df, a):
    print("Columns:", list(df.columns))
    print("\nHead:\n", df.head(8).to_string())
    print(f"\nRows: {len(df):,}")
    for col in (a.cell_col, a.locus_col):
        if col in df.columns:
            print(f"  {col}: {df[col].nunique()} unique")
        else:
            print(f"  [!] expected column '{col}' NOT found — set the right "
                  f"--{col.replace('_','-')} name")
    for col in (a.x_col, a.y_col, a.z_col):
        if col in df.columns:
            print(f"  {col}: range {df[col].min():.1f}..{df[col].max():.1f}")


def _locus_labels(cell_df, a):
    g = cell_df[a.locus_col].astype(str)
    if a.chrom_level:
        # The 1Mb file mixes chromosome loci ('chr1-#1') with named-gene loci
        # ('0610010F05Rik'). Map the former to their chromosome (chr1..chrX)
        # and bucket the rest into 'other' -> ~21 dense fiber dimensions.
        chrom = g.str.extract(r'^(chr[0-9XYM]+)')[0]
        g = chrom.fillna("other")
    return g.values


def run_one_nucleus(cell_df, a, cell_id):
    mol = pd.DataFrame({
        "gene": _locus_labels(cell_df, a),
        "x_local": (cell_df[a.x_col].values - cell_df[a.x_col].min()) * a.xy_nm,
        "y_local": (cell_df[a.y_col].values - cell_df[a.y_col].min()) * a.xy_nm,
        "z_local": (cell_df[a.z_col].values - cell_df[a.z_col].min()) * a.z_nm,
    })
    grid, raw, genes = fibers.build_kernel_fibers(
        mol, grid_spacing=a.grid_nm, bandwidth=a.bandwidth_nm,
        min_molecules=a.min_molecules, z_col="z_local",
        z_spacing=a.z_grid_nm, z_bandwidth=a.z_bandwidth_nm,
        block_size=a.block_size, verbose=False)
    if len(grid) < a.n_zones * 5:
        print(f"  cell {cell_id}: too few grid points ({len(grid)}) — skipped")
        return None
    fib = fibers.normalize_fibers(raw)
    edges = graph.build_spatial_graph(grid, k=a.k_graph, verbose=False)
    stalks = sheaf.compute_stalks(fib, grid, uniform_dim=a.uniform_dim,
                                  k_nbrs=min(a.k_nbrs, len(grid) - 1), verbose=False)
    restr = sheaf.compute_restriction_maps(stalks, edges, a.uniform_dim, verbose=False)
    braw, bsm = sheaf.compute_boundary_scores(
        fib, grid, stalks, edges, restr, smooth_radius=a.smooth_radius_nm,
        verbose=False)
    labels = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=a.n_zones,
        spatial_weight=a.spatial_weight, boundary_alpha=a.boundary_alpha,
        verbose=False)
    labels, _ = clustering.spatial_cleanup(labels, edges, grid_coords=grid,
                                           verbose=False)
    return dict(grid=grid, labels=np.asarray(labels), boundary=bsm,
                n_loci=len(cell_df), n_grid=len(grid), n_zones=len(set(labels)))


def plot_nucleus(res, cell_id, out_dir):
    g, lab = res["grid"], res["labels"]
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    uniq = sorted(set(lab))
    cmap = plt.cm.get_cmap("tab10", max(len(uniq), 2))
    for i, c in enumerate(uniq):
        m = lab == c
        ax.scatter(g[m, 0], g[m, 1], g[m, 2], s=6, color=cmap(i))
    ax.set_title(f"nuclear zones (k={len(uniq)})")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)"); ax.set_zlabel("z (nm)")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    p = ax2.scatter(g[:, 0], g[:, 1], g[:, 2], c=res["boundary"], s=6, cmap="hot")
    ax2.set_title("sheaf boundary score")
    fig.colorbar(p, ax=ax2, shrink=0.6)
    fig.suptitle(f"Takei DNA seqFISH+ — cell {cell_id} "
                 f"({res['n_loci']} loci, {res['n_grid']} grid pts)",
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, f"nucleus_{cell_id}_zones.png")
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    a = build_parser().parse_args()
    df = load_table(a.input)
    if a.inspect:
        inspect(df, a)
        return
    need = [a.cell_col, a.x_col, a.y_col, a.z_col, a.locus_col]
    if a.fov_col in df.columns:
        need = [a.fov_col] + need
    for col in need:
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not in file. Run --inspect and "
                             "set the correct --*-col names.")
    os.makedirs(a.output_dir, exist_ok=True)

    # a nucleus is (fov, cellID) together
    if a.fov_col in df.columns:
        df["_cell"] = df[a.fov_col].astype(str) + "_" + df[a.cell_col].astype(str)
    else:
        df["_cell"] = df[a.cell_col].astype(str)

    counts = df.groupby("_cell").size()
    good = counts[counts >= a.min_cell_loci].index.tolist()
    if a.cells:
        cells = [c for c in a.cells.split(",")]
    else:
        cells = good[:a.max_cells]
    print(f"{len(good)} nuclei with >= {a.min_cell_loci} dots; "
          f"running {len(cells)}")

    rows = []
    for cid in cells:
        sub = df[df["_cell"] == str(cid)]
        t0 = time.time()
        res = run_one_nucleus(sub, a, cid)
        if res is None:
            continue
        p = plot_nucleus(res, cid, a.output_dir)
        pd.DataFrame({"x": res["grid"][:, 0], "y": res["grid"][:, 1],
                      "z": res["grid"][:, 2], "zone": res["labels"]}
                     ).to_csv(os.path.join(a.output_dir,
                              f"nucleus_{cid}_zones.csv"), index=False)
        rows.append(dict(cell=cid, n_loci=res["n_loci"], n_grid=res["n_grid"],
                         n_zones=res["n_zones"], sec=round(time.time() - t0, 1)))
        print(f"  cell {cid}: {res['n_grid']} grid pts, {res['n_zones']} zones "
              f"-> {p}")
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.output_dir, "summary.csv"),
                                  index=False)
        print(f"\nSummary -> {os.path.join(a.output_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
