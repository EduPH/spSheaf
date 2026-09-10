"""spSheaf on one SpaceTx mouse-VISp dataset, scored on a per-cell annotation.

Mirrors ``experiments/osmfish/run_osmFISH.py``: the pipeline runs blind on
the spot table, grid labels are mapped to annotated cells by nearest grid
point, and the annotation is registered onto the molecule grid WITHOUT
labels (composition correlation of nearby spots against the cell's own
counts, as ``experiments/osmfish/register.py``). Every sheaf parameter is the
MERFISH default; only the spatial scale, stalk dimension and k are arguments.

Column names differ between the five SpaceTx tables; pass them explicitly
after reading ``inspect.py``'s output.

    python -m experiments.spacetx.run_spacetx MERFISH_Allen_Prototype \\
        --spot-x xc --spot-y yc --spot-gene gene \\
        --cell-id sample_name --cell-x rotated_x --cell-y rotated_y --label layer \\
        --unit-um 1.0 --grid-spacing 10 --bandwidth 25 --uniform-dim 6

(MERFISH_Allen_Prototype: spots ``xc``/``yc`` and cells ``rotated_x``/``rotated_y``
are the same micrometre frame; ``xc_adjusted`` is ``rotated`` + 5 um; the
cell-by-gene table is transposed with a ``gene_name`` row.)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spsheaf import evaluation, paths, pipeline
from experiments.common.panel_export import export_panels


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--spot-x", default="xc"); p.add_argument("--spot-y", default="yc")
    p.add_argument("--spot-gene", default="gene")
    p.add_argument("--cell-id", default="sample_name"); p.add_argument("--cell-x", default="rotated_x")
    p.add_argument("--cell-y", default="rotated_y"); p.add_argument("--label", default="layer",
                   help="annotation column in the mapped cell table (or in --label-csv)")
    p.add_argument("--label-csv", default=None,
                   help="optional CSV with columns <cell-id>,<label> (e.g. exported from the SDMBench h5ad)")
    p.add_argument("--unit-um", type=float, default=1.0,
                   help="micrometres per coordinate unit of the spot AND cell tables")
    p.add_argument("--grid-spacing", type=float, default=10.0, help="um")
    p.add_argument("--bandwidth", type=float, default=25.0, help="um")
    p.add_argument("--smooth-radius", type=float, default=60.0, help="um")
    p.add_argument("--uniform-dim", type=int, default=6)
    p.add_argument("--n-clusters", type=int, default=None, help="default: number of annotation classes")
    p.add_argument("--register-radius", type=float, default=10.0, help="um, cell radius for the composition score")
    p.add_argument("--no-register", action="store_true", help="centroid alignment only (not recommended)")
    p.add_argument("--seed", type=int, default=42, help="k-means random_state")
    p.add_argument("--output-dir", default=None)
    return p


def load(a):
    root = paths.raw("spacetx") / a.dataset
    spots = pd.read_csv(root / "s3_spot_table.csv", usecols=[a.spot_x, a.spot_y, a.spot_gene])
    cells = pd.read_csv(root / "s3_mapped_cell_table.csv")
    cbg = pd.read_csv(root / "s3_cell_by_gene.csv", index_col=0)
    if "gene_name" in cbg.index:
        # SpaceTx layout: header = column positions, row 'gene_name' = gene names,
        # remaining rows = cells (index = cell id as in the mapped cell table)
        cbg.columns = cbg.loc["gene_name"].astype(str).values
        cbg = cbg.drop(index="gene_name")
        cbg.index = cbg.index.astype(str)
        cbg = cbg.apply(pd.to_numeric, errors="coerce").fillna(0)
    mol = pd.DataFrame({"gene": spots[a.spot_gene].astype(str).values,
                        "x": spots[a.spot_x].values * a.unit_um,
                        "y": spots[a.spot_y].values * a.unit_um,
                        "matched_h5ad_file": f"spacetx_{a.dataset}"})
    cxy = cells[[a.cell_x, a.cell_y]].values * a.unit_um
    cid = cells[a.cell_id].astype(str).values
    if a.label_csv:
        lab = pd.read_csv(a.label_csv)
        lab = lab.set_index(lab.columns[0])[lab.columns[1]]
        gt = np.array([lab.get(c, np.nan) for c in cid], dtype=object)
    else:
        gt = cells[a.label].values.astype(object)
    return mol, cxy, cid, gt, cbg


def composition_r(cxy, mol_xy, gcode, counts, n_genes, radius, seed=0, n=1500):
    tree = cKDTree(mol_xy)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cxy), min(n, len(cxy)), replace=False)
    lists = tree.query_ball_point(cxy[pick], radius)
    lens = np.array([len(l) for l in lists]); keep = lens >= 10
    if keep.sum() == 0:
        return float("nan")
    ids = np.concatenate([lists[i] for i in np.where(keep)[0]])
    rows = np.repeat(np.arange(keep.sum()), lens[keep]); g = gcode[ids]; ok = g >= 0
    C = np.zeros((keep.sum(), n_genes)); np.add.at(C, (rows[ok], g[ok]), 1.0)
    T = counts[pick[keep]].astype(float)
    good = (C.sum(1) > 0) & (T.sum(1) > 0); C, T = C[good], T[good]
    C = C / C.sum(1, keepdims=True); T = T / T.sum(1, keepdims=True)
    C -= C.mean(1, keepdims=True); T -= T.mean(1, keepdims=True)
    r = (C * T).sum(1) / (np.linalg.norm(C, axis=1) * np.linalg.norm(T, axis=1) + 1e-12)
    return float(np.nanmean(r))


def register(mol, cxy, cid, cbg, radius, grid):
    """Label-free translation from the cell frame to the molecule grid frame."""
    xy = mol[["x", "y"]].values; xy = xy - xy.min(0)
    genes = [g for g in cbg.columns if g in set(mol.gene)]
    gi = {g: i for i, g in enumerate(genes)}
    gcode = mol.gene.map(lambda g: gi.get(g, -1)).values
    counts = cbg.reindex(cid)[genes].fillna(0).values
    centroid_t = xy.mean(0) - cxy.mean(0)
    c0 = cxy + centroid_t
    span = 0.15 * (xy.max(0) - xy.min(0))
    step = max(radius, span.max() / 12)
    def s(dx, dy, n): return composition_r(c0 + [dx, dy], xy, gcode, counts, len(genes), radius, n=n)
    r0 = s(0, 0, 1500)
    cand = [(s(dx, dy, 400), dx, dy) for dx in np.arange(-span[0], span[0] + 1, step)
            for dy in np.arange(-span[1], span[1] + 1, step)]
    _, dx0, dy0 = max(cand)
    fine = step / 4
    cand = [(s(dx, dy, 1500), dx, dy) for dx in np.arange(dx0 - step, dx0 + step + 1, fine)
            for dy in np.arange(dy0 - step, dy0 + step + 1, fine)]
    r, dx, dy = max(cand)
    print(f"  registration: composition r {r0:.3f} (centroid) -> {r:.3f} at shift ({dx:.1f}, {dy:.1f}) um")
    return centroid_t + np.array([dx, dy]), r, r0


def main():
    a = build_parser().parse_args()
    mol, cxy, cid, gt, cbg = load(a)
    valid = pd.notna(gt)
    names = sorted(set(gt[valid]))
    k = a.n_clusters or len(names)
    print(f"{a.dataset}: {len(mol):,} spots, {mol.gene.nunique()} genes, {len(cxy):,} cells, "
          f"{valid.sum():,} annotated in {len(names)} classes -> k = {k}")
    res = pipeline.run_single_slice(
        mol=mol, slice_name=f"spacetx_{a.dataset}", n_clusters=k,
        grid_spacing=a.grid_spacing, bandwidth=a.bandwidth, smooth_radius=a.smooth_radius,
        k_graph=6, k_nbrs=50, uniform_dim=a.uniform_dim, spatial_weight=0.5,
        boundary_alpha=0.5, cleanup=True, eval_at_cell_level=False,
        random_state=a.seed, verbose=False)
    grid, labels = res["grid_coords"], res["labels"]
    if a.no_register:
        t = grid.mean(0) - cxy.mean(0); r = r0 = float("nan")
    else:
        t, r, r0 = register(mol, cxy, cid, cbg, a.register_radius, grid)
    cells_al = cxy + t
    _, idx = cKDTree(grid).query(cells_al, k=1)
    pred = labels[idx]
    gti = np.array([names.index(g) if isinstance(g, str) else -1 for g in gt])
    m = evaluation.evaluate_clustering_full(pred, cells_al, gt_int=gti, gt_valid=valid,
                                            name=f"spSheaf {a.dataset}", verbose=False)
    matched = evaluation.hungarian_match_labels(pred, gti, valid)
    rec = {n: float(np.mean(matched[valid & (gti == i)] == i)) for i, n in enumerate(names)}
    print(f"  ARI {m['ari']:.3f}  NMI {m['nmi']:.3f}  HOM {m['hom']:.3f}  COM {m['com']:.3f}")
    for n, v in rec.items():
        print(f"    {n:30s} recall {v:.2f}")
    out = a.output_dir or str(paths.results("spacetx") / a.dataset)
    export_panels(out, f"spsheaf_seed{a.seed}" if a.seed != 42 else "spsheaf", arrays=dict(grid_coords=grid, grid_labels=labels,
                  boundary_smooth=res["boundary_smooth"], cell_coords=cells_al,
                  cell_gt_int=gti, cell_predicted=pred, cell_pred_matched=matched),
                  meta=dict(dataset=a.dataset, gt_names=names, metrics=m, recall=rec,
                            registration=dict(translation=[float(x) for x in t], r=r, r_centroid=r0),
                            params=vars(a)))
    row = dict(seed=a.seed, **{k: m[k] for k in ("ari", "nmi", "hom", "com", "chaos", "pas", "asw")})
    f = f"{out}/metrics.csv"
    import os
    prev = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    if len(prev) and "seed" in prev.columns:
        prev = prev[prev["seed"] != a.seed]
    elif len(prev):
        prev = pd.DataFrame()          # old single-row format without a seed column
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(f, index=False)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
