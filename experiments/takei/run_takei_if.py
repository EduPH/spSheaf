#!/usr/bin/env python
"""
WS2a - TRUE chromatin-mark nuclear zones from Takei sequential-IF data.

This is the Phase-2 signal the paper actually uses to define nuclear zones:
per-voxel **immunofluorescence intensities of ~21 chromatin marks / nuclear
bodies** (H3K9me3, H3K27me3, H3K4/H3K9/H3K27-ac, H4K20me1/2/3, Lamin B1,
Fibrillarin, SF3a66, mH2A1, LINE1/SINEB1, MinSat, Telomere, …). Each mark is a
(Channel, hybID) pair given by the supplementary Excel; the binned data store,
per cell and channel, a long table of ``hybID, avgint, xIdx, yIdx, zIdx`` (one
row per voxel per hyb round).

Pipeline (per nucleus):
  1. read the Ch1/Ch2 CSVs, keep the labelled (Channel, hyb) rows, and pivot
     (xIdx, yIdx, zIdx) voxels x marks -> a dense per-voxel mark-intensity fiber;
  2. voxel coords -> nm (xy x 103, z x 250), the feature fiber machinery
     (fibers.build_feature_fibers, 3D, anisotropic) smooths marks onto a grid;
  3. the sheaf pipeline carves boundary-aware **nuclear zones**;
  4. certification: per-zone mean mark z-score -> the chromatin identity of each
     zone (e.g. a Lamin B1 / H3K9me3 peripheral heterochromatin zone vs an
     RNAPII / H3K27ac active interior zone).

The data zip's inner folder is labelled 'replicate2' (a Zenodo naming quirk;
md5 matches the replicate1-Pos0 record), so marks are read from the
'E14 replicate 2' Excel sheet by default.

Usage
-----
    python run_takei_if.py \
        --zip IF-binned1x1x1-E14-replicate1-Pos0.zip \
        --excel IF_supp_table_hyb_targets.xlsx \
        --max-cells 3 --n-zones 5 --output-dir output_takei_if
    # sandbox-fast validation on one cell:
    python run_takei_if.py --zip ... --excel ... --cells 1 \
        --subsample-voxels 0.15 --no-plot --output-dir output_takei_if_test
"""
import argparse
import os
import re
import zipfile
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spsheaf import fibers, graph, sheaf, clustering
try:
    from plot_nucleus import slices_png
except Exception:
    slices_png = None

# heterochromatin / active groupings, only for a readable certification summary
HETERO = {"H3K9me3", "H3K9me2", "H3K27me3", "H3K27me2", "H4K20me3", "H4K20me2",
          "mH2A1", "LINE1", "MinSat", "Lamin B1"}
ACTIVE = {"H3K9ac", "H3K27ac", "H4K16ac", "RNAPII Ser5-P", "SF3a66", "H4K20me1"}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True, help="IF-binned ...Pos0.zip")
    p.add_argument("--excel", required=True, help="IF_supp_table_hyb_targets.xlsx")
    p.add_argument("--sheet", default="E14 replicate 2")
    p.add_argument("--output-dir", default="output_takei_if")
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
    p.add_argument("--subsample-voxels", type=float, default=1.0,
                   help="fraction of voxels to keep (speed; 1.0 = all).")
    p.add_argument("--normalize-voxel", action="store_true",
                   help="divide each voxel's mark vector by its mean before "
                        "log1p, so zones reflect mark COMPOSITION not overall "
                        "brightness (removes staining-depth gradients).")
    p.add_argument("--min-voxels", type=int, default=2000)
    p.add_argument("--max-cells", type=int, default=3)
    p.add_argument("--cells", default=None, help="comma list of cell numbers")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def mark_map(excel, sheet):
    """(Channel, hybID) -> mark name, dropping unlabelled/repeated hybs."""
    d = pd.read_excel(excel, sheet_name=sheet)[["hybID", "Channel", "Target"]]
    d = d.dropna(subset=["Target"])
    d["hybID"] = d.hybID.astype(int)
    d["Channel"] = d.Channel.astype(int)
    return {(int(r.Channel), int(r.hybID)): str(r.Target).strip()
            for r in d.itertuples()}


def list_cells(zf):
    cells = set()
    for n in zf.namelist():
        m = re.search(r"-Cell(\d+)-", n)
        if m and "outputData" in n:
            cells.add(int(m.group(1)))
    return sorted(cells)


def load_cell_fiber(zf, cell, marks, a):
    """Return (voxel_coords_nm, feature_matrix, mark_names) for one nucleus."""
    # channel number embedded in file name -> Excel 'Channel'
    ch_files = {}
    for n in zf.namelist():
        m = re.search(rf"-Ch(\d)-Pos0-Cell{cell}-", n)
        if m and n.endswith(".csv"):
            ch_files[int(m.group(1))] = n
    frames = {}
    for ch, fn in ch_files.items():
        want_hybs = {h for (c, h) in marks if c == ch}
        if not want_hybs:
            continue  # e.g. Ch4 = DAPI, not a labelled mark
        with zf.open(fn) as fh:
            df = pd.read_csv(fh, dtype={"hybID": "int16", "avgint": "float32",
                                        "xIdx": "int32", "yIdx": "int32",
                                        "zIdx": "int16"})
        df = df[df.hybID.isin(want_hybs)]
        for h, g in df.groupby("hybID"):
            name = marks[(ch, int(h))]
            frames[name] = g.set_index(["xIdx", "yIdx", "zIdx"])["avgint"]
    if not frames:
        return None
    wide = pd.DataFrame(frames)                     # voxels x marks
    wide = wide.dropna()                            # keep voxels imaged in all marks
    if a.subsample_voxels < 1.0 and len(wide) > a.min_voxels:
        wide = wide.sample(frac=a.subsample_voxels, random_state=a.seed)
    if len(wide) < a.min_voxels:
        return None
    idx = np.array(wide.index.tolist(), dtype=float)   # (x,y,z) voxel idx
    coords = np.column_stack([(idx[:, 0] - idx[:, 0].min()) * a.xy_nm,
                              (idx[:, 1] - idx[:, 1].min()) * a.xy_nm,
                              (idx[:, 2] - idx[:, 2].min()) * a.z_nm])
    vals = wide.values.astype(float)
    if a.normalize_voxel:
        vals = vals / (vals.mean(axis=1, keepdims=True) + 1e-9)
    feats = np.log1p(vals)
    return coords, feats, list(wide.columns)


def run_nucleus(coords, feats, a):
    grid, fib = fibers.build_feature_fibers(
        coords, feats, grid_spacing=a.grid_nm, bandwidth=a.bandwidth_nm,
        z_spacing=a.z_grid_nm, z_bandwidth=a.z_bandwidth_nm, verbose=False)
    if len(grid) < a.n_zones * 5:
        return None
    ud = min(a.uniform_dim, fib.shape[1] - 1)
    edges = graph.build_spatial_graph(grid, k=a.k_graph, verbose=False)
    stalks = sheaf.compute_stalks(fib, grid, uniform_dim=ud,
                                  k_nbrs=min(a.k_nbrs, len(grid) - 1), verbose=False)
    restr = sheaf.compute_restriction_maps(stalks, edges, ud, verbose=False)
    _, bsm = sheaf.compute_boundary_scores(fib, grid, stalks, edges, restr,
                                           smooth_radius=a.smooth_radius_nm,
                                           verbose=False)
    labels = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=a.n_zones,
        n_pca=min(10, fib.shape[1]), random_state=a.seed, verbose=False)
    labels, _ = clustering.spatial_cleanup(labels, edges, grid_coords=grid,
                                           verbose=False)
    return grid, np.asarray(labels), fib


def zone_mark_signature(fib, labels, mark_names):
    """Per-zone mean z-scored mark profile -> top marks + hetero/active balance."""
    Z = (fib - fib.mean(0)) / (fib.std(0) + 1e-9)
    rows, tops = [], {}
    for z in sorted(set(labels)):
        prof = Z[labels == z].mean(0)
        order = np.argsort(-prof)
        top = [mark_names[i] for i in order[:4]]
        tops[int(z)] = top
        het = np.mean([prof[i] for i, m in enumerate(mark_names) if m in HETERO])
        act = np.mean([prof[i] for i, m in enumerate(mark_names) if m in ACTIVE])
        rows.append(dict(zone=int(z), top_marks=", ".join(top),
                         hetero_z=round(float(het), 2),
                         active_z=round(float(act), 2),
                         **{m: round(float(prof[i]), 2)
                            for i, m in enumerate(mark_names)}))
    return pd.DataFrame(rows), tops


def main():
    a = build_parser().parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    marks = mark_map(a.excel, a.sheet)
    zf = zipfile.ZipFile(a.zip)
    all_cells = list_cells(zf)
    cells = ([int(c) for c in a.cells.split(",")] if a.cells
             else all_cells[:a.max_cells])
    print(f"{len(all_cells)} nuclei in zip; running {cells}; "
          f"{len(marks)} labelled marks ({sorted(set(marks.values()))[:6]}...)")

    summ = []
    for cid in cells:
        zpath = os.path.join(a.output_dir, f"nucleus_{cid}_zones.csv")
        if os.path.exists(zpath) and not a.overwrite:
            print(f"  cell {cid}: exists, skip"); continue
        loaded = load_cell_fiber(zf, cid, marks, a)
        if loaded is None:
            print(f"  cell {cid}: too few voxels, skip"); continue
        coords, feats, mark_names = loaded
        res = run_nucleus(coords, feats, a)
        if res is None:
            print(f"  cell {cid}: too few grid pts, skip"); continue
        grid, labels, fib = res
        sig, tops = zone_mark_signature(fib, labels, mark_names)
        pd.DataFrame({"x": grid[:, 0], "y": grid[:, 1], "z": grid[:, 2],
                      "zone": labels}).to_csv(zpath, index=False)
        sig.to_csv(os.path.join(a.output_dir,
                                f"nucleus_{cid}_zone_marks.csv"), index=False)
        if not a.no_plot and slices_png is not None:
            slices_png(pd.DataFrame({"x": grid[:, 0], "y": grid[:, 1],
                        "z": grid[:, 2], "zone": labels}), "zone",
                       os.path.join(a.output_dir, f"nucleus_{cid}_slices.png"))
        summ.append(dict(cell=cid, voxels=len(coords), grid=len(grid),
                         zones=len(set(labels))))
        print(f"  cell {cid}: {len(coords)} voxels, {len(grid)} grid pts, "
              f"{len(set(labels))} zones")
        for z in sorted(tops):
            r = sig[sig.zone == z].iloc[0]
            print(f"      zone {z}: {tops[z][0]}, {tops[z][1]}, {tops[z][2]} "
                  f"(hetero {r.hetero_z:+.2f} / active {r.active_z:+.2f})")
    if summ:
        pd.DataFrame(summ).to_csv(os.path.join(a.output_dir, "summary.csv"),
                                  index=False)
        print(f"\n-> {a.output_dir}/")


if __name__ == "__main__":
    main()
