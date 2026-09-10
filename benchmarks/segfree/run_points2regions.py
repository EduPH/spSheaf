"""Points2Regions (Andersson et al. 2024, Biol Imaging) on the Moffitt MERFISH slices.

Run as: python run_p2r_merfish.py [slice ...]

Points2Regions takes exactly the same input as the sheaf method — molecule
(x, y) plus gene label — and returns a region label per molecule, so this is a
like-for-like comparison with no format concessions on either side.

Hyperparameters. The authors expose three: pixel_width (rasterisation bin, in
data units), pixel_smoothing (neighbourhood radius in pixels) and
min_num_pts_per_pixel. Their own tuning notebook searches pixel_width in
[0.1, 4] and pixel_smoothing in [1, 20] on data whose units are ~1 px per
molecule spacing, so the absolute numbers do not transfer to micron-scale
MERFISH coordinates. We therefore run a grid over pixel_width (micron) x
pixel_smoothing and report both the full grid and the best-ARI setting, which
is a bound in Points2Regions' favour (its parameters are tuned against the
ground truth; the sheaf paper's are cross-validated).

num_clusters is set to 8, the number of annotated anatomical domains, matching
the k given to every method in the SDMBench comparison.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from points2regions import Points2Regions

from benchmarks.segfree import common as bc

from spsheaf import paths as _paths
OUT = str(_paths.results("segfree/p2r"))
os.makedirs(OUT, exist_ok=True)

K = 8
# The grid is deliberately extended until the optimum is interior rather than on
# a boundary, so the reported Points2Regions number is not an artefact of an
# undertuned search. pixel_width x pixel_smoothing is the effective spatial
# support in micron (10..600 um here; tissue domains in this data are ~100-500 um).
PIXEL_WIDTHS = [5.0, 10.0, 15.0, 25.0, 40.0]   # micron per rasterised pixel
PIXEL_SMOOTHINGS = [1.0, 2.0, 4.0, 6.0, 9.0, 15.0, 22.0, 30.0, 45.0]
SEEDS = [42, 0, 1, 2, 3]                    # seed sensitivity at the best setting


def run_slice(slice_id, molecules):
    adata, mols = bc.load_merfish_slice(slice_id, molecules=molecules)
    xy = np.c_[mols.x.values, mols.y.values]
    labels = mols.gene.values
    cell_xy = adata.obsm["spatial"]
    gt = adata.obs["domain"].astype(str).values

    from scipy.spatial import cKDTree
    tree = cKDTree(xy)

    def molecule_to_cell(mol_labels):
        """Majority region label of the molecules nearest each cell centroid.

        Points2Regions labels molecules, not cells; we transfer to cells by
        majority vote over the 30 molecules closest to the centroid, which is
        the molecule-level analogue of the nearest-grid-point transfer used for
        grid-based methods.
        """
        _, idx = tree.query(cell_xy, k=30)
        lab = np.asarray(mol_labels)[idx]
        out = np.empty(len(cell_xy), dtype=object)
        for i in range(len(cell_xy)):
            v, c = np.unique(lab[i][lab[i] >= 0], return_counts=True)
            out[i] = str(v[np.argmax(c)]) if len(v) else "-1"
        return out

    rows, out = [], {"slice": slice_id, "n_molecules": int(len(xy)), "grid": []}
    for pw in PIXEL_WIDTHS:
        for ps in PIXEL_SMOOTHINGS:
            t = time.time()
            p2r = Points2Regions(xy, labels, pixel_width=pw, pixel_smoothing=ps)
            p2r.fit(num_clusters=K, seed=42)
            mol_lab = p2r.predict(output="marker")
            dt = time.time() - t
            pred = molecule_to_cell(mol_lab)
            pred = bc.fill_background(pred, cell_xy, bg="-1")
            m = bc.all_metrics(gt, pred, cell_xy)
            m.update(pixel_width=pw, pixel_smoothing=ps, seed=42, total_s=dt,
                     frac_bg_molecules=float(np.mean(np.asarray(mol_lab) < 0)))
            rows.append(m)
            print(f"[p2r {slice_id}] pw={pw} ps={ps} ARI={m['ARI']:.3f} "
                  f"NMI={m['NMI']:.3f} k={m['n_pred_domains']} {dt:.0f}s", flush=True)
    out["grid"] = rows

    best = max(rows, key=lambda r: r["ARI"])
    out["best"] = best
    # re-run best setting across seeds and persist labels
    seed_rows = []
    for sd in SEEDS:
        p2r = Points2Regions(xy, labels, pixel_width=best["pixel_width"],
                             pixel_smoothing=best["pixel_smoothing"])
        p2r.fit(num_clusters=K, seed=sd)
        mol_lab = p2r.predict(output="marker")
        pred = bc.fill_background(molecule_to_cell(mol_lab), cell_xy, bg="-1")
        m = bc.all_metrics(gt, pred, cell_xy)
        m["seed"] = sd
        seed_rows.append(m)
        if sd == SEEDS[0]:
            pd.DataFrame({"cell": adata.obs_names, "gt": gt, "pred": pred}).to_csv(
                f"{OUT}/labels_{slice_id}_best.csv", index=False)
    out["seeds"] = seed_rows
    print(f"[p2r {slice_id} best] pw={best['pixel_width']} ps={best['pixel_smoothing']} "
          f"ARI={best['ARI']:.3f} NMI={best['NMI']:.3f} | seed ARI "
          f"{np.mean([r['ARI'] for r in seed_rows]):.3f}"
          f"+-{np.std([r['ARI'] for r in seed_rows]):.3f}", flush=True)

    with open(f"{OUT}/metrics_{slice_id}.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    slices = sys.argv[1:] or bc.MER_SLICES
    molecules = bc.load_merfish_export()
    print(f"loaded {len(molecules):,} molecules", flush=True)
    for s in slices:
        run_slice(s, molecules)
