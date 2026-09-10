"""Points2Regions on the Moffitt slices with the input restricted to the
physical section (cell_z == bregma of the annotated slice), all focal planes.

The original benchmark loader selected molecules by the annotated slice's xy
bounding box; the export lays the twelve bregma sections of animal 1 out in
overlapping blocks, so each box also contained a second, unrelated section
(0.11 on top of -0.19, 0.16 on -0.14, 0.21 on -0.09, 0.26 on -0.04, 0.01 on
-0.24). Every competitor MERFISH number in the manuscript before 8 Sep 2026
was computed on that contaminated input. This reruns the full
Points2Regions grid (same PIXEL_WIDTHS x PIXEL_SMOOTHINGS, same seeds, same
cell mapping) on the clean section.

    SPSHEAF_DATA=... python -m benchmarks.segfree.run_p2r_clean [slice ...]
Writes data/results/segfree/p2r_clean/metrics_<slice>.json (+ labels).
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from points2regions import Points2Regions
from spsheaf import paths
from benchmarks.segfree import common as bc
from benchmarks.segfree.input_check import load_all, slice_bbox, variant, SLICES

OUT = paths.results("segfree") / "p2r_clean"; os.makedirs(OUT, exist_ok=True)
K = 8
PIXEL_WIDTHS = [5.0, 10.0, 15.0, 25.0, 40.0]
PIXEL_SMOOTHINGS = [1.0, 2.0, 4.0, 6.0, 9.0, 15.0, 22.0, 30.0, 45.0]
SEEDS = [42, 0, 1, 2, 3]

def to_cells(mol_lab, tree, cell_xy):
    _, idx = tree.query(cell_xy, k=30); lab = np.asarray(mol_lab)[idx]
    out = np.empty(len(cell_xy), dtype=object)
    for i in range(len(cell_xy)):
        v, c = np.unique(lab[i][lab[i] >= 0], return_counts=True)
        out[i] = str(v[np.argmax(c)]) if len(v) else "-1"
    return bc.fill_background(out, cell_xy, bg="-1")

def run_slice(s, mol):
    box, name = slice_bbox(mol, s)
    d = variant(box, name, "section", "allz")
    cells = (d.dropna(subset=["domain"]).drop_duplicates("cell_id")
             .set_index("cell_id")[["cell_x", "cell_y", "domain"]])
    xy = np.c_[d.x.values, d.y.values]; genes = d.gene.values
    cell_xy = cells[["cell_x", "cell_y"]].values; gt = cells.domain.astype(str).values
    tree = cKDTree(xy)
    rows = []
    for pw in PIXEL_WIDTHS:
        for ps in PIXEL_SMOOTHINGS:
            t = time.time()
            p = Points2Regions(xy, genes, pixel_width=pw, pixel_smoothing=ps); p.fit(num_clusters=K, seed=42)
            pred = to_cells(p.predict(output="marker"), tree, cell_xy)
            m = bc.all_metrics(gt, pred, cell_xy); m.update(pixel_width=pw, pixel_smoothing=ps, seed=42, total_s=time.time() - t)
            rows.append(m); print(f"[p2r-clean {s}] pw={pw} ps={ps} ARI={m['ARI']:.3f} NMI={m['NMI']:.3f}", flush=True)
    best = max(rows, key=lambda r: r["ARI"]); seed_rows = []
    for sd in SEEDS:
        p = Points2Regions(xy, genes, pixel_width=best["pixel_width"], pixel_smoothing=best["pixel_smoothing"]); p.fit(num_clusters=K, seed=sd)
        pred = to_cells(p.predict(output="marker"), tree, cell_xy)
        m = bc.all_metrics(gt, pred, cell_xy); m["seed"] = sd; seed_rows.append(m)
        if sd == SEEDS[0]:
            pd.DataFrame({"cell": cells.index, "gt": gt, "pred": pred}).to_csv(OUT / f"labels_{s}_best.csv", index=False)
    out = dict(slice=s, n_molecules=int(len(xy)), n_cells=int(len(gt)), grid=rows, best=best, seeds=seed_rows)
    (OUT / f"metrics_{s}.json").write_text(json.dumps(out, indent=1))
    print(f"[p2r-clean {s} best] pw={best['pixel_width']} ps={best['pixel_smoothing']} ARI={best['ARI']:.3f} "
          f"NMI={best['NMI']:.3f} seeds {np.mean([r['ARI'] for r in seed_rows]):.3f}+-{np.std([r['ARI'] for r in seed_rows]):.3f}", flush=True)

if __name__ == "__main__":
    mol = load_all(); print(f"loaded {len(mol):,}", flush=True)
    for s in (sys.argv[1:] or SLICES): run_slice(s, mol)
