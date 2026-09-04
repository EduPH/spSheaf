"""SSAM and Points2Regions on the osmFISH mouse somatosensory cortex.

Run as: python run_osmfish.py [ssam|p2r|both]

Input is the raw molecule table (1.98M molecules, 35 gene labels) registered
onto the benchmark cell frame by osm_register.py. Ground truth is the 11
annotated cortical regions in the SDMBench AnnData (4,839 cells).

SSAM hyperparameters follow the authors' own osmFISH notebook
(HiDiHlabs/ssam_example/osmFISH_SSp.ipynb): bandwidth 2.5, sampling_distance 1,
expression threshold 0.027, norm threshold 0.04, pca_dims 33, resolution 0.15,
bin_celltypemaps(step=10, radius=100), find_domains(n_clusters=15,
merge_remote=False, merge_thres=0.6, norm_thres=4500).

The osmFISH molecule frame is in camera pixels at 15.3846 px/um, so coordinates
are converted to micron before rasterisation — otherwise every length-scale
hyperparameter (kernel bandwidth, smoothing radius, window radius) would be off
by 15x and both methods would be evaluated far outside their intended regime.
"""
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from benchmarks.segfree import common as bc
from benchmarks.segfree import osm_register as osr

PIXEL_PER_UM = 15.3846      # osmFISH imaging scale (Codeluppi et al. 2018)
from spsheaf import paths as _paths
OUT = str(_paths.results("segfree/osmfish"))
os.makedirs(OUT, exist_ok=True)
os.makedirs("work", exist_ok=True)

K_GT = 11                   # annotated cortical regions


def load():
    o, om, _, cell_xy_px, _, _, _ = osr.load_cells()
    mol_um = np.c_[om.x.values, om.y.values] / PIXEL_PER_UM
    cell_um = cell_xy_px / PIXEL_PER_UM
    gt = o.obs["Region"].astype(str).values
    return o, om.gene.values, mol_um, cell_um, gt


def run_ssam(genes, mol_um, cell_um, gt):
    import ssam
    x0, y0 = mol_um[:, 0].min() - 10.0, mol_um[:, 1].min() - 10.0
    loc = pd.DataFrame({"gene": genes, "x": mol_um[:, 0] - x0, "y": mol_um[:, 1] - y0})
    width = float(np.ceil(loc.x.max() + 10))
    height = float(np.ceil(loc.y.max() + 10))
    ds = ssam.SSAMDataset(store="work/ssam_osmfish.zarr")
    an = ssam.SSAMAnalysis(ds, ncores=20, verbose=False)
    err = io.StringIO()
    timings = {}
    with contextlib.redirect_stderr(err):
        t = time.time()
        an.run_kde(locations=loc, width=width, height=height, depth=1,
                   bandwidth=2.5, sampling_distance=1.0)
        timings["kde"] = time.time() - t
        an.set_thresholds(expression_threshold=0.027, norm_threshold=0.04)
        t = time.time()
        an.find_localmax(search_size=3)
        an.normalize_vectors()
        an.scale_vectors()
        an.cluster_vectors(method="louvain", pca_dims=33, resolution=0.15,
                           max_correlation=1.0, metric="correlation",
                           centroid_correction_threshold=0.6, random_state=0)
        timings["cluster"] = time.time() - t
        t = time.time()
        an.map_celltypes()
        an.filter_celltypemaps(min_norm="local",
                               filter_params={"block_size": 151, "method": "mean",
                                              "mode": "constant", "offset": 0.2},
                               min_r=0.6, min_blob_area=50, fill_blobs=True)
        an.bin_celltypemaps(step=10, radius=100)
        timings["celltype_map"] = time.time() - t

    out = {"n_localmax": int(np.asarray(ds.local_maxs).shape[1]),
           "n_celltypes": int(len(np.unique(np.asarray(ds.cluster_labels)[np.asarray(ds.cluster_labels) >= 0]))),
           "timings": timings, "runs": []}

    def evaluate(kw, name, save=False):
        with contextlib.redirect_stderr(err):
            t = time.time()
            an.find_domains(**kw)
            dt = time.time() - t
        dom = np.asarray(ds.inferred_domains)
        pred = bc.nearest_pixel_labels(dom[..., dom.shape[2] // 2], cell_um, x0, y0)
        pred = bc.fill_background(pred, cell_um, bg=-1)
        m = bc.all_metrics(gt, pred, cell_um)
        m.update(find_domains_s=dt, params=kw)
        if save:
            pd.DataFrame({"gt": gt, "pred": pred}).to_csv(f"{OUT}/labels_ssam_{name}.csv", index=False)
        return m

    pub = dict(n_clusters=15, merge_remote=False, merge_thres=0.6, norm_thres=4500)
    out["published"] = evaluate(pub, "published", save=True)
    print(f"[ssam osmfish published] ARI={out['published']['ARI']:.3f} "
          f"NMI={out['published']['NMI']:.3f} k={out['published']['n_pred_domains']}", flush=True)
    sweep = [dict(n_clusters=n, merge_remote=rm, merge_thres=t, norm_thres=nt)
             for n in (11, 15, 20) for t in (0.6, 0.7, 0.8) for rm in (False, True)
             for nt in (1500, 4500)]
    out["sweep"] = [evaluate(kw, "sweep") for kw in sweep]
    best = max(out["sweep"], key=lambda r: r["ARI"])
    out["best_sweep"] = evaluate(best["params"], "best_sweep", save=True)
    print(f"[ssam osmfish best] ARI={best['ARI']:.3f} NMI={best['NMI']:.3f} "
          f"k={best['n_pred_domains']} params={best['params']}", flush=True)
    with open(f"{OUT}/metrics_ssam.json", "w") as f:
        json.dump(out, f, indent=1)


def run_p2r(genes, mol_um, cell_um, gt):
    from points2regions import Points2Regions
    from scipy.spatial import cKDTree
    tree = cKDTree(mol_um)

    def to_cells(mol_lab):
        _, idx = tree.query(cell_um, k=30)
        lab = np.asarray(mol_lab)[idx]
        out = np.empty(len(cell_um), dtype=object)
        for i in range(len(cell_um)):
            v, c = np.unique(lab[i][lab[i] >= 0], return_counts=True)
            out[i] = str(v[np.argmax(c)]) if len(v) else "-1"
        return out

    rows = []
    for pw in [10.0, 20.0, 40.0, 60.0]:
        for ps in [1.0, 2.0, 4.0, 6.0, 9.0, 15.0, 22.0, 30.0]:
            t = time.time()
            p2r = Points2Regions(mol_um, genes, pixel_width=pw, pixel_smoothing=ps)
            p2r.fit(num_clusters=K_GT, seed=42)
            pred = bc.fill_background(to_cells(p2r.predict(output="marker")), cell_um, bg="-1")
            m = bc.all_metrics(gt, pred, cell_um)
            m.update(pixel_width=pw, pixel_smoothing=ps, total_s=time.time() - t)
            rows.append(m)
            print(f"[p2r osmfish] pw={pw} ps={ps} ARI={m['ARI']:.3f} NMI={m['NMI']:.3f} "
                  f"k={m['n_pred_domains']}", flush=True)
    best = max(rows, key=lambda r: r["ARI"])
    seeds = []
    for sd in (42, 0, 1, 2, 3):
        p2r = Points2Regions(mol_um, genes, pixel_width=best["pixel_width"],
                             pixel_smoothing=best["pixel_smoothing"])
        p2r.fit(num_clusters=K_GT, seed=sd)
        pred = bc.fill_background(to_cells(p2r.predict(output="marker")), cell_um, bg="-1")
        m = bc.all_metrics(gt, pred, cell_um)
        m["seed"] = sd
        seeds.append(m)
        if sd == 42:
            pd.DataFrame({"gt": gt, "pred": pred}).to_csv(f"{OUT}/labels_p2r_best.csv", index=False)
    print(f"[p2r osmfish best] pw={best['pixel_width']} ps={best['pixel_smoothing']} "
          f"ARI={best['ARI']:.3f} NMI={best['NMI']:.3f} | seed ARI "
          f"{np.mean([r['ARI'] for r in seeds]):.3f}+-{np.std([r['ARI'] for r in seeds]):.3f}", flush=True)
    with open(f"{OUT}/metrics_p2r.json", "w") as f:
        json.dump({"grid": rows, "best": best, "seeds": seeds}, f, indent=1)


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    o, genes, mol_um, cell_um, gt = load()
    print(f"osmFISH: {len(genes):,} molecules, {len(cell_um)} cells, "
          f"{len(set(gt))} regions, extent "
          f"{(mol_um.max(0) - mol_um.min(0)).round(0)} um", flush=True)
    if which in ("p2r", "both"):
        run_p2r(genes, mol_um, cell_um, gt)
    if which in ("ssam", "both"):
        run_ssam(genes, mol_um, cell_um, gt)
