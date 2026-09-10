"""Does the MERFISH input asymmetry matter? spSheaf and Points2Regions on
matched and unmatched inputs.

Two differences between what spSheaf and the comparators read on the Moffitt
slices, both found on 5-8 Sep 2026:

  cells   spSheaf's loaders keep molecules whose cell is in the SDMBench
          annotation (``matched_h5ad_file == slice``); the comparators'
          loader keeps every molecule in the slice's bounding box. On slice
          0.19 that is 5,803 annotated + 5,684 unannotated cells, spatially
          interleaved, all classes (1,315 "Ambiguous").
  planes  spSheaf keeps one focal plane (z = 0); the comparators get all
          seven.

This script runs both methods on every combination, per slice, and scores
each on the annotated cells only, as always:

  spSheaf         matched/z0 (= Table 1)  matched/all-z  all-cells/z0  all-cells/all-z
  Points2Regions  all-cells/all-z (= stored benchmark; a reproduction check)
                  all-cells/z0    matched/z0 (= spSheaf's input), at the
                  configuration leave-one-slice-out selected for that slice.

    SPSHEAF_DATA=... python -m benchmarks.segfree.input_check [slice ...]

Writes data/results/segfree/input_check/<slice>.json and input_check.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spsheaf import paths, pipeline
from benchmarks.segfree import common as bc

RAW = "moffitt2018_export_full/molecules_full_add.csv.gz"
SLICES = ["0.04", "0.09", "0.14", "0.19", "0.24"]
COLS = ["gene", "x", "y", "z", "cell_id", "cell_x", "cell_y", "cell_z",
        "matched_h5ad_file", "domain"]
#: leave-one-slice-out selected Points2Regions configuration per held-out
#: slice, from benchmarks/segfree/results/metrics (matched_selection.py)
P2R_HELDOUT = {"0.04": (10.0, 30.0), "0.09": (10.0, 45.0), "0.14": (10.0, 30.0),
               "0.19": (10.0, 45.0), "0.24": (25.0, 22.0)}
PAD = 60.0


def load_all():
    src = paths.raw("merfish") / RAW
    parts = [ch for ch in pd.read_csv(src, usecols=COLS, chunksize=3_000_000, low_memory=False)]
    return pd.concat(parts, ignore_index=True)


def slice_bbox(mol, s):
    name = f"MERFISH_{s}.h5ad"
    ann = mol[mol.matched_h5ad_file == name]
    cx, cy = ann.cell_x, ann.cell_y
    box = ((mol.x >= cx.min() - PAD) & (mol.x <= cx.max() + PAD)
           & (mol.y >= cy.min() - PAD) & (mol.y <= cy.max() + PAD))
    return mol[box].copy(), name


def variant(df, name, cells, planes):
    if cells == "all":
        d = df
    elif cells == "section":
        # the physical section only: the export lays the 12 bregma sections of
        # animal 1 out in overlapping xy blocks, so a bounding box also captures
        # a second section (bregma 0.11 lies on top of -0.19, etc.)
        bregma = df.loc[df.matched_h5ad_file == name, "cell_z"].iloc[0]
        d = df[df.cell_z == bregma]
    else:
        d = df[df.matched_h5ad_file == name]
    d = d.copy()
    d["matched_h5ad_file"] = name           # so select_slice keeps every row
    if planes == "z0":
        d = d[d.z == 0]
    else:
        d = d.drop(columns=["z"])           # select_slice then keeps all planes
    return d


def run_spsheaf(d, name):
    t = time.time()
    res = pipeline.run_single_slice(d, name, n_clusters=8, verbose=False)
    m = res["metrics"]
    return dict(ari=float(m["ari"]), nmi=float(m["nmi"]), n_mol=int(len(d)),
                n_cells_scored=int(res["n_cells"]), time_s=time.time() - t)


def run_p2r(d, name, cfg):
    from points2regions import Points2Regions
    ann = d[d.matched_h5ad_file == name] if "matched_h5ad_file" in d else d
    cells = (d.dropna(subset=["domain"]).drop_duplicates("cell_id")
             .set_index("cell_id")[["cell_x", "cell_y", "domain"]])
    xy = np.c_[d.x.values, d.y.values]
    cell_xy = cells[["cell_x", "cell_y"]].values
    gt = cells.domain.astype(str).values
    t = time.time()
    p2r = Points2Regions(xy, d.gene.values, pixel_width=cfg[0], pixel_smoothing=cfg[1])
    p2r.fit(num_clusters=8, seed=42)
    mol_lab = np.asarray(p2r.predict(output="marker"))
    _, idx = cKDTree(xy).query(cell_xy, k=30)
    lab = mol_lab[idx]
    pred = np.empty(len(cell_xy), dtype=object)
    for i in range(len(cell_xy)):
        v, c = np.unique(lab[i][lab[i] >= 0], return_counts=True)
        pred[i] = str(v[np.argmax(c)]) if len(v) else "-1"
    pred = bc.fill_background(pred, cell_xy, bg="-1")
    m = bc.all_metrics(gt, pred, cell_xy)
    return dict(ari=float(m["ARI"]), nmi=float(m["NMI"]), n_mol=int(len(d)),
                n_cells_scored=int(len(gt)), time_s=time.time() - t)


def main():
    slices = sys.argv[1:] or SLICES
    out_dir = paths.results("segfree") / "input_check"
    os.makedirs(out_dir, exist_ok=True)
    mol = load_all()
    print(f"loaded {len(mol):,} molecules", flush=True)
    rows = []
    for s in slices:
        box, name = slice_bbox(mol, s)
        rep = {}
        for cells, planes in [("matched", "z0"), ("section", "z0"), ("section", "allz"),
                              ("all", "z0"), ("all", "allz")]:
            d = variant(box, name, cells, planes)
            # the annotation columns must stay NaN for unannotated cells
            r = run_spsheaf(d, name)
            rep[f"spSheaf/{cells}/{planes}"] = r
            print(f"[{s}] spSheaf {cells:8s} {planes:5s} ARI {r['ari']:.3f} NMI {r['nmi']:.3f} "
                  f"n_mol {r['n_mol']:,} cells {r['n_cells_scored']} ({r['time_s']:.0f}s)", flush=True)
        for cells, planes in [("all", "allz"), ("all", "z0"), ("section", "allz"),
                              ("section", "z0"), ("matched", "z0")]:
            d = variant(box, name, cells, planes)
            r = run_p2r(d, name, P2R_HELDOUT[s])
            rep[f"Points2Regions/{cells}/{planes}"] = r
            print(f"[{s}] P2R     {cells:8s} {planes:5s} ARI {r['ari']:.3f} NMI {r['nmi']:.3f} "
                  f"n_mol {r['n_mol']:,} ({r['time_s']:.0f}s)", flush=True)
        (out_dir / f"{s}.json").write_text(json.dumps(rep, indent=1))
        for k, v in rep.items():
            meth, cells, planes = k.split("/")
            rows.append(dict(slice=s, method=meth, cells=cells, planes=planes, **v))
        pd.DataFrame(rows).to_csv(out_dir / "input_check.csv", index=False)
    df = pd.DataFrame(rows)
    print("\nmean over slices:")
    print(df.groupby(["method", "cells", "planes"])[["ari", "nmi"]].mean().round(3))


if __name__ == "__main__":
    main()
