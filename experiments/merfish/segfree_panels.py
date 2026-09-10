"""Dump what Fig. 2's spatial row draws: one MERFISH slice, per cell, the
annotation and the domain label of every method in the segmentation-free
comparison, Hungarian-matched to the annotation.

spSheaf labels are recomputed from the cached fibers and boundary field of
``experiments.merfish._cache`` (the same cache the ablation uses), with the
package defaults and ``random_state=42``; the competitors' labels are the
tracked per-cell tables under ``benchmarks/segfree/results/labels/``, whose
row order is the slice's cell order in ``cell_metadata_add.csv.gz`` (checked
here by comparing the annotation column).

    python -m experiments.merfish.segfree_panels --slice MERFISH_0.19.h5ad

Writes ``data/results/merfish/panels/segfree_maps_<slice>.npz`` + ``.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spsheaf import clustering, evaluation, paths
from experiments.common.panel_export import export_panels
from experiments.merfish import _cache

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "benchmarks" / "segfree" / "results" / "labels"
RAW_CSV = "moffitt2018_export_full/molecules_full_add.csv.gz"

#: Competitor labels on the clean section (benchmarks/segfree/common.py,
#: bregma selection; 8 Sep 2026). The earlier ``*_merfish__labels_*`` files were
#: scored on two superimposed sections and are kept only as a record.
COMPETITORS = {
    "FICTURE": "ficture_merfish_clean__labels_{tag}_best.csv",
    "Points2Regions": "points2regions_merfish_clean__labels_{tag}_best.csv",
    # SSAM 1.1.3 under-resolves at the published Louvain resolution; the panel
    # uses the sctransform / resolution-4 clustering, which is the per-slice
    # argmax of its full configuration grid (Methods, SSAM).
    "SSAM": "ssam_merfish_clean__labels_{tag}_sct_r4_best_sweep.csv",
}


def _cells(slice_name):
    """Per-cell centroid and annotation for the slice, in the cache's local frame."""
    src = paths.raw("merfish") / RAW_CSV
    cols = ["cell_id", "cell_x", "cell_y", "matched_h5ad_file", "domain"]
    parts = []
    for ch in pd.read_csv(src, usecols=cols, chunksize=2_000_000, low_memory=False):
        parts.append(ch[ch.matched_h5ad_file == slice_name])
    mol = pd.concat(parts, ignore_index=True)
    cells = (mol.drop_duplicates("cell_id")
                .set_index("cell_id")[["cell_x", "cell_y", "domain"]])
    # same cell order as the tracked label tables
    meta = pd.read_csv(paths.raw("merfish") / "moffitt2018_export_full/cell_metadata_add.csv.gz")
    order = meta.loc[meta.matched_h5ad_file == slice_name, "cell_id"]
    cells = cells.reindex(order)
    par = pd.read_parquet(_cache.parquet_path(slice_name))
    x0 = float((par.x - par.x_local).iloc[0]); y0 = float((par.y - par.y_local).iloc[0])
    cells["x_local"] = cells.cell_x - x0
    cells["y_local"] = cells.cell_y - y0
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slice", default="MERFISH_0.19.h5ad")
    a = p.parse_args()
    tag = _cache._slice_tag(a.slice)

    c = _cache.load(a.slice)
    grid, fib, bsm, edges = c["grid"], c["fib"], c["boundary_smooth"], c["edges"]
    labels = clustering.boundary_aware_clustering(
        fib, grid, bsm, edges, n_clusters=8, random_state=42, verbose=False)
    labels, _ = clustering.spatial_cleanup(labels, edges, grid_coords=grid, verbose=False)

    cells = _cells(a.slice)
    gt_names = sorted(cells.domain.dropna().unique())
    gt = np.array([gt_names.index(d) if isinstance(d, str) else -1 for d in cells.domain])
    valid = gt >= 0
    _, idx = cKDTree(grid).query(cells[["x_local", "y_local"]].values, k=1)
    preds = {"spSheaf": labels[idx]}
    for name, pat in COMPETITORS.items():
        t = pd.read_csv(LABELS / pat.format(tag=tag.lstrip("s")))
        if "cell" in t.columns:                  # align on cell id, not on row order
            t = t.set_index("cell").reindex(cells.index[valid]).reset_index()
            full = pd.DataFrame({"gt": cells.domain.values, "pred": -1}, index=cells.index)
            full.loc[cells.index[valid], "pred"] = t["pred"].values
            t = full.reset_index(drop=True)
        assert len(t) == len(cells), (name, len(t), len(cells))
        assert (t["gt"].values[valid] == cells.domain.values[valid]).all(), name
        pr = pd.factorize(t.pred)[0]
        preds[name] = pr

    ari = {}
    matched = {}
    for name, pr in preds.items():
        ari[name] = float(evaluation.compute_ari(pr, gt, valid))
        matched[name] = evaluation.hungarian_match_labels(pr, gt, valid)

    out = paths.results("merfish")
    export_panels(
        str(out), f"segfree_maps_{tag}",
        arrays=dict(x=cells.x_local.values, y=cells.y_local.values, gt=gt, valid=valid,
                    **{f"pred_{k}": v for k, v in matched.items()}),
        meta=dict(slice=a.slice, tag=tag, gt_names=gt_names, ari=ari,
                  methods=list(preds), n_cells=int(len(cells))),
    )
    print({k: round(v, 3) for k, v in ari.items()}, "->", out / "panels" / f"segfree_maps_{tag}.npz")


if __name__ == "__main__":
    main()
