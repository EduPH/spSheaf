#!/usr/bin/env python
"""
Phase-2 (dense-feature fiber) nuclear-zone detection.

Instead of the sparse one-hot locus fiber, this uses a *dense continuous*
per-point feature vector as the fiber, via ``fibers.build_feature_fibers``.
Two feature sources are supported from the DNAseqFISH+ 25kb file:

  --features chrom   the 20 per-dot chromosome intensities (chr1..chrX) —
                     available now; yields chromosome-territory zones.
  --features cols:A,B,...   arbitrary intensity columns (e.g. once you merge
                     the chromatin-mark IF intensities in — the TRUE zone
                     signal per the paper).

The same feature-fiber machinery works for the chromatin-mark IF field; only
the input columns change.

Certification
-------------
For each nucleus, each locus dot is assigned the zone of its nearest grid
point, and we report:
  * per-zone mean chromosome-intensity signature (which chromosomes dominate
    each zone) — an interpretable identity for the zone;
  * ARI/NMI vs the file's ``labelID`` column IF requested (--score-labelid),
    with the caveat that labelID is ~binary and does NOT encode the paper's
    chromatin-mark zones, so this is a weak reference, not ground truth.

The proper certification (marker identity of each zone) requires the IF
chromatin-mark intensities; see README Phase 2.

Usage
-----
    python run_takei_features.py \
        --input dna_seqfish/DNAseqFISH+/DNAseqFISH+25kbloci-E14-replicate1.csv \
        --features chrom --n-zones 5 --max-cells 5 --output-dir output_takei_feat
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from spsheaf import fibers, graph, sheaf, clustering
from plot_nucleus import slices_png


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default="output_takei_feat")
    p.add_argument("--features", default="chrom",
                   help="'chrom' (chr1..chrX intensities) or 'cols:c1,c2,...'")
    p.add_argument("--fov-col", default="fov")
    p.add_argument("--cell-col", default="cellID")
    p.add_argument("--x-col", default="x"); p.add_argument("--y-col", default="y")
    p.add_argument("--z-col", default="z")
    p.add_argument("--xy-nm", type=float, default=103.0)
    p.add_argument("--z-nm", type=float, default=250.0)
    p.add_argument("--grid-nm", type=float, default=550.0)
    p.add_argument("--bandwidth-nm", type=float, default=800.0)
    p.add_argument("--z-grid-nm", type=float, default=450.0)
    p.add_argument("--z-bandwidth-nm", type=float, default=700.0)
    p.add_argument("--k-graph", type=int, default=6)
    p.add_argument("--k-nbrs", type=int, default=40)
    p.add_argument("--uniform-dim", type=int, default=6)
    p.add_argument("--n-zones", type=int, default=5)
    p.add_argument("--smooth-radius-nm", type=float, default=900.0)
    p.add_argument("--min-cell-dots", type=int, default=500)
    p.add_argument("--max-cells", type=int, default=5)
    p.add_argument("--cells", default=None)
    p.add_argument("--score-labelid", action="store_true",
                   help="Also report ARI/NMI vs labelID (weak reference).")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the per-nucleus slices PNG (much faster batches).")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute nuclei whose zones CSV already exists.")
    return p


def feature_cols(df, spec):
    if spec == "chrom":
        return [c for c in df.columns
                if c.endswith("_intensity") and c != "dot_intensity"]
    if spec.startswith("cols:"):
        return spec.split(":", 1)[1].split(",")
    raise SystemExit("--features must be 'chrom' or 'cols:...'")


def run_nucleus(sub, a, cols):
    coords = np.column_stack([
        (sub[a.x_col].values - sub[a.x_col].min()) * a.xy_nm,
        (sub[a.y_col].values - sub[a.y_col].min()) * a.xy_nm,
        (sub[a.z_col].values - sub[a.z_col].min()) * a.z_nm])
    feats = np.log1p(sub[cols].values.astype(float))
    grid, fib = fibers.build_feature_fibers(
        coords, feats, grid_spacing=a.grid_nm, bandwidth=a.bandwidth_nm,
        z_spacing=a.z_grid_nm, z_bandwidth=a.z_bandwidth_nm, verbose=False)
    if len(grid) < a.n_zones * 5:
        return None
    edges = graph.build_spatial_graph(grid, k=a.k_graph, verbose=False)
    stalks = sheaf.compute_stalks(fib, grid, uniform_dim=min(a.uniform_dim, fib.shape[1] - 1),
                                  k_nbrs=min(a.k_nbrs, len(grid) - 1), verbose=False)
    restr = sheaf.compute_restriction_maps(stalks, edges, min(a.uniform_dim, fib.shape[1] - 1), verbose=False)
    braw, bsm = sheaf.compute_boundary_scores(fib, grid, stalks, edges, restr,
                                              smooth_radius=a.smooth_radius_nm, verbose=False)
    labels = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=a.n_zones, n_pca=min(10, fib.shape[1]),
        verbose=False)
    labels, _ = clustering.spatial_cleanup(labels, edges, grid_coords=grid, verbose=False)
    return grid, np.asarray(labels), coords


def zone_chrom_signature(sub, dot_zone, cols):
    """Mean chromosome intensity per inferred zone -> top chromosome each."""
    m = pd.DataFrame(sub[cols].values, columns=cols)
    m["zone"] = dot_zone
    prof = m.groupby("zone").mean()
    top = prof.idxmax(axis=1).to_dict()
    return {int(z): c.replace("_intensity", "") for z, c in top.items()}


def main():
    a = build_parser().parse_args()
    df = pd.read_csv(a.input)
    cols = feature_cols(df, a.features)
    df["_cell"] = df[a.fov_col].astype(str) + "_" + df[a.cell_col].astype(str)
    counts = df.groupby("_cell").size()
    good = counts[counts >= a.min_cell_dots].index.tolist()
    cells = a.cells.split(",") if a.cells else good[:a.max_cells]
    os.makedirs(a.output_dir, exist_ok=True)
    print(f"{len(good)} nuclei; running {len(cells)}; fiber = {len(cols)} features "
          f"({a.features})")

    rows = []
    for cid in cells:
        zpath = os.path.join(a.output_dir, f"nucleus_{cid}_zones.csv")
        if os.path.exists(zpath) and not a.overwrite:
            continue  # resume: already done
        sub = df[df["_cell"] == str(cid)].reset_index(drop=True)
        res = run_nucleus(sub, a, cols)
        if res is None:
            print(f"  {cid}: skipped (too few grid pts)"); continue
        grid, labels, coords = res
        # assign each dot the zone of its nearest grid point
        _, j = cKDTree(grid).query(coords, k=1)
        dot_zone = labels[j]
        sig = zone_chrom_signature(sub, dot_zone, cols)
        row = dict(cell=cid, dots=len(sub), grid=len(grid),
                   zones=len(set(labels)))
        if a.score_labelid and "labelID" in sub.columns:
            m = sub["labelID"].values != -1
            if m.sum() > 10 and len(set(sub["labelID"][m])) > 1:
                row["ARI_vs_labelID"] = round(adjusted_rand_score(
                    sub["labelID"][m], dot_zone[m]), 3)
                row["NMI_vs_labelID"] = round(normalized_mutual_info_score(
                    sub["labelID"][m], dot_zone[m]), 3)
        pd.DataFrame({"x": grid[:, 0], "y": grid[:, 1], "z": grid[:, 2],
                      "zone": labels}).to_csv(
            os.path.join(a.output_dir, f"nucleus_{cid}_zones.csv"), index=False)
        slices_png(pd.DataFrame({"x": grid[:, 0], "y": grid[:, 1],
                    "z": grid[:, 2], "zone": labels}), "zone",
                   os.path.join(a.output_dir, f"nucleus_{cid}_slices.png"))
        rows.append(row)
        print(f"  {cid}: {len(grid)} grid pts, {len(set(labels))} zones, "
              f"zone->top-chrom {sig}"
              + (f"  ARI(labelID)={row.get('ARI_vs_labelID')}" if a.score_labelid else ""))
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.output_dir, "summary.csv"), index=False)
        print(f"\nSummary -> {os.path.join(a.output_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
