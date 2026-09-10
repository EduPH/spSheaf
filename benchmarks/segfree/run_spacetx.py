"""SSAM and Points2Regions on a SpaceTx mouse-VISp dataset, matched protocol.

Run as: python -m benchmarks.segfree.run_spacetx <dataset> [ssam|p2r|both]
        (dataset: MERFISH_Allen_Prototype, BARISTASeq, ...)

Input is the spot table spSheaf was run on (experiments/spacetx/run_spacetx.py),
in micrometres; cells are placed in the molecule frame with the label-free
translation that run stored (panels/spsheaf.json), so both sides of the
comparison are scored against the same registration. Ground truth is the
per-cell layer annotation of the mapped cell table.

SSAM uses its authors' MERFISH-notebook settings (the panel is the Zhuang MOp
panel), then the same find_domains sweep as the MERFISH benchmark; the best
in-sample setting is reported, as for osmFISH (a single section admits no
held-out rule). Points2Regions is grid-searched over pixel_width x
pixel_smoothing with the cluster count fixed to the annotated number.

Column names default to the MERFISH_Allen_Prototype layout; override with the
--spot-*/--cell-* flags for the other datasets.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from benchmarks.segfree import common as bc
from spsheaf import paths as _paths

SSAM_EXP_THRES, SSAM_NORM_THRES = 0.0055, 0.0035
SSAM_BANDWIDTH, SSAM_SAMPLING = 2.5, 1.0
SSAM_PCA, SSAM_RES = 60, 0.15
SSAM_FILTER = {"block_size": 151, "method": "mean", "mode": "constant", "offset": 0}
#: Cell-type clustering variants swept as part of SSAM's grid, as on MERFISH
#: (Methods, SSAM): the published normalization is sctransform, and SSAM 1.1.3
#: under-resolves at the published Louvain resolution, so the resolution is a
#: hyperparameter. Set with SSAM_RESOLUTIONS="1 4 8"; SSAM_NORMALIZATION picks
#: "sctransform" (the paper's, needs R) or "builtin".
SSAM_RESOLUTIONS = [float(v) for v in os.environ.get("SSAM_RESOLUTIONS", "1.0").split()]
SSAM_NORMALIZATION = os.environ.get("SSAM_NORMALIZATION", "builtin")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset"); p.add_argument("which", nargs="?", default="both")
    p.add_argument("--spot-x", default="xc"); p.add_argument("--spot-y", default="yc")
    p.add_argument("--spot-gene", default="gene")
    p.add_argument("--cell-id", default="sample_name"); p.add_argument("--cell-x", default="rotated_x")
    p.add_argument("--cell-y", default="rotated_y"); p.add_argument("--label", default="layer")
    p.add_argument("--unit-um", type=float, default=1.0)
    return p.parse_args()


def load(a):
    root = _paths.raw("spacetx") / a.dataset
    spots = pd.read_csv(root / "s3_spot_table.csv", usecols=[a.spot_x, a.spot_y, a.spot_gene])
    cells = pd.read_csv(root / "s3_mapped_cell_table.csv")
    genes = spots[a.spot_gene].astype(str).values
    mol_um = np.c_[spots[a.spot_x].values, spots[a.spot_y].values] * a.unit_um
    cxy = cells[[a.cell_x, a.cell_y]].values * a.unit_um
    gt = cells[a.label].astype(str).values
    meta = _paths.results("spacetx") / a.dataset / "panels" / "spsheaf.json"
    if meta.exists():
        t = np.asarray(json.loads(meta.read_text())["registration"]["translation"], dtype=float)
        cell_um = cxy + t + mol_um.min(0)      # spSheaf's grid frame is molecules minus their minimum
        print(f"cells placed with spSheaf's label-free registration (translation {t.round(1)} um "
              f"relative to the molecule-local frame)")
    else:
        cell_um = cxy
        print("WARNING: no spSheaf registration found for this dataset; using the tables' frames as is")
    return genes, mol_um, cell_um, gt


def run_ssam(a, genes, mol_um, cell_um, gt, out):
    import ssam
    k_gt = len(set(gt))
    x0, y0 = mol_um[:, 0].min() - 10.0, mol_um[:, 1].min() - 10.0
    loc = pd.DataFrame({"gene": genes, "x": mol_um[:, 0] - x0, "y": mol_um[:, 1] - y0})
    width = float(np.ceil(loc.x.max() + 10)); height = float(np.ceil(loc.y.max() + 10))
    os.makedirs("work", exist_ok=True)
    ds = ssam.SSAMDataset(store=f"work/ssam_spacetx_{a.dataset}.zarr")
    an = ssam.SSAMAnalysis(ds, ncores=8, verbose=False)
    err = io.StringIO(); timings = {}
    with contextlib.redirect_stderr(err):
        t = time.time()
        an.run_kde(locations=loc, width=width, height=height, depth=1,
                   bandwidth=SSAM_BANDWIDTH, sampling_distance=SSAM_SAMPLING)
        timings["kde"] = time.time() - t
        an.set_thresholds(expression_threshold=SSAM_EXP_THRES, norm_threshold=SSAM_NORM_THRES)
        t = time.time()
        an.find_localmax(search_size=3)
        if SSAM_NORMALIZATION == "sctransform":
            from benchmarks.segfree.sctransform_csv import ssam_use_csv_sctransform
            ssam_use_csv_sctransform()
            an.normalize_vectors_sctransform()
        else:
            an.normalize_vectors()
        an.scale_vectors()
        # SSAM >= 1.1 adds a medoid-correlation outlier filter that, on this
        # sparser dataset (2,067 local maxima against tens of thousands on
        # Moffitt MERFISH), discards three quarters of the vectors and leaves 0-5
        # cell types at the MERFISH resolution 0.15; it is switched off and the
        # resolution raised to 1.0, which gives 13 cell types. Recorded in
        # metrics_ssam.json.
        pass
    n_localmax = int(np.asarray(ds.local_maxs).shape[1])

    def cluster_at(resolution):
        """Cell types at one Louvain resolution, carried through to binned maps."""
        t0 = time.time()
        with contextlib.redirect_stderr(err):
            an.cluster_vectors(method="louvain", pca_dims=min(SSAM_PCA, len(set(genes)) - 1),
                               resolution=resolution, max_correlation=1.0, metric="correlation",
                               outlier_detection_method=None, random_state=0)
            an.map_celltypes()
            an.filter_celltypemaps(min_norm="local", filter_params=SSAM_FILTER,
                                   min_r=0.6, min_blob_area=7, fill_blobs=True)
            an.bin_celltypemaps(step=10, radius=100)
        lab = np.asarray(ds.cluster_labels)
        return int(len(np.unique(lab[lab >= 0]))), time.time() - t0

    res = {"ssam_settings": dict(expression_threshold=SSAM_EXP_THRES, norm_threshold=SSAM_NORM_THRES,
                                 pca_dims=SSAM_PCA, outlier_detection=None,
                                 normalization=SSAM_NORMALIZATION),
           "n_localmax": n_localmax, "timings": timings}

    def evaluate(kw, name, save=False):
        try:
            with contextlib.redirect_stderr(err):
                t = time.time(); an.find_domains(**kw); dt = time.time() - t
        except ValueError as e:          # no bins survive norm_thres at this density
            print(f"[ssam {a.dataset}] {kw}: no domains ({str(e)[:60]})", flush=True)
            return dict(ARI=float("nan"), NMI=float("nan"), n_pred_domains=0, params=kw, failed=True)
        dom = np.asarray(ds.inferred_domains)
        pred = bc.nearest_pixel_labels(dom[..., dom.shape[2] // 2], cell_um, x0, y0, scale=SSAM_SAMPLING)
        pred = bc.fill_background(pred, cell_um, bg=-1)
        m = bc.all_metrics(gt, pred, cell_um); m.update(find_domains_s=dt, params=kw)
        if save:
            pd.DataFrame({"gt": gt, "pred": pred}).to_csv(f"{out}/labels_ssam_{name}.csv", index=False)
        return m

    print(f"[ssam {a.dataset}] {n_localmax} local maxima", flush=True)
    pub = dict(n_clusters=20, merge_remote=True, merge_thres=0.8, norm_thres=4000)
    # norm_thres is an absolute L1 norm of the binned cell-type map and scales
    # with molecule density; the MERFISH values (1000, 4000) can leave no bins
    # on a sparser dataset, so the sweep goes down two decades.
    # binned-map L1 norms here have median ~830 and 99th percentile ~2,240, so
    # the MERFISH thresholds (1000, 4000) leave few or no bins; 300-1000 spans
    # the useful range and thresholds below 300 make find_domains very slow.
    sweep = [dict(n_clusters=n, merge_remote=rm, merge_thres=t, norm_thres=nt)
             for n in (k_gt, 12, 20) for t in (0.6, 0.7, 0.8, 0.9) for rm in (False, True)
             for nt in (300, 600, 1000)]
    for resolution in SSAM_RESOLUTIONS:
        n_ct, dt = cluster_at(resolution)
        rtag = f"_r{resolution:g}" if len(SSAM_RESOLUTIONS) > 1 or resolution != 1.0 else ""
        r = dict(res)
        r["ssam_settings"] = dict(res["ssam_settings"], resolution=resolution)
        r["n_celltypes"] = n_ct
        r["timings"] = dict(res["timings"], cluster=dt)
        print(f"[ssam {a.dataset}{rtag}] resolution {resolution:g}: {n_ct} cell types ({dt:.0f} s)", flush=True)
        r["published"] = evaluate(pub, f"published{rtag}", save=True)
        print(f"[ssam {a.dataset}{rtag} published] ARI={r['published']['ARI']:.3f} "
              f"NMI={r['published']['NMI']:.3f} k={r['published']['n_pred_domains']}", flush=True)
        r["sweep"] = [evaluate(kw, "sweep") for kw in sweep]
        ok = [x for x in r["sweep"] if not x.get("failed")]
        if not ok:
            print(f"[ssam {a.dataset}{rtag}] no find_domains setting produced domains", flush=True)
            continue
        best = max(ok, key=lambda x: x["ARI"])
        r["best_sweep"] = evaluate(best["params"], f"best_sweep{rtag}", save=True)
        print(f"[ssam {a.dataset}{rtag} best] ARI={best['ARI']:.3f} NMI={best['NMI']:.3f} "
              f"k={best['n_pred_domains']} params={best['params']}", flush=True)
        json.dump(r, open(f"{out}/metrics_ssam{rtag}.json", "w"), indent=1)


def run_p2r(a, genes, mol_um, cell_um, gt, out):
    from points2regions import Points2Regions
    from scipy.spatial import cKDTree
    k_gt = len(set(gt))
    tree = cKDTree(mol_um)

    def to_cells(mol_lab):
        _, idx = tree.query(cell_um, k=30)
        lab = np.asarray(mol_lab)[idx]
        o = np.empty(len(cell_um), dtype=object)
        for i in range(len(cell_um)):
            v, c = np.unique(lab[i][lab[i] >= 0], return_counts=True)
            o[i] = str(v[np.argmax(c)]) if len(v) else "-1"
        return o

    rows = []
    for pw in [5.0, 10.0, 20.0, 40.0]:
        for ps in [1.0, 2.0, 4.0, 6.0, 9.0, 15.0]:
            t = time.time()
            p2r = Points2Regions(mol_um, genes, pixel_width=pw, pixel_smoothing=ps)
            p2r.fit(num_clusters=k_gt, seed=42)
            pred = bc.fill_background(to_cells(p2r.predict(output="marker")), cell_um, bg="-1")
            m = bc.all_metrics(gt, pred, cell_um)
            m.update(pixel_width=pw, pixel_smoothing=ps, total_s=time.time() - t); rows.append(m)
            print(f"[p2r {a.dataset}] pw={pw} ps={ps} ARI={m['ARI']:.3f} NMI={m['NMI']:.3f} "
                  f"k={m['n_pred_domains']}", flush=True)
    best = max(rows, key=lambda r: r["ARI"])
    seeds = []
    for sd in (42, 0, 1, 2, 3):
        p2r = Points2Regions(mol_um, genes, pixel_width=best["pixel_width"], pixel_smoothing=best["pixel_smoothing"])
        p2r.fit(num_clusters=k_gt, seed=sd)
        pred = bc.fill_background(to_cells(p2r.predict(output="marker")), cell_um, bg="-1")
        m = bc.all_metrics(gt, pred, cell_um); m["seed"] = sd; seeds.append(m)
        if sd == 42:
            pd.DataFrame({"gt": gt, "pred": pred}).to_csv(f"{out}/labels_p2r_best.csv", index=False)
    print(f"[p2r {a.dataset} best] pw={best['pixel_width']} ps={best['pixel_smoothing']} ARI={best['ARI']:.3f} "
          f"NMI={best['NMI']:.3f} | seeds ARI {np.mean([r['ARI'] for r in seeds]):.3f}+-{np.std([r['ARI'] for r in seeds]):.3f}",
          flush=True)
    json.dump({"grid": rows, "best": best, "seeds": seeds}, open(f"{out}/metrics_p2r.json", "w"), indent=1)


if __name__ == "__main__":
    a = parse()
    out = str(_paths.results(f"segfree/spacetx_{a.dataset}")); os.makedirs(out, exist_ok=True)
    genes, mol_um, cell_um, gt = load(a)
    print(f"{a.dataset}: {len(genes):,} spots, {len(set(genes))} genes, {len(cell_um)} cells, "
          f"{len(set(gt))} classes, extent {(mol_um.max(0) - mol_um.min(0)).round(0)} um", flush=True)
    if a.which in ("p2r", "both"):
        run_p2r(a, genes, mol_um, cell_um, gt, out)
    if a.which in ("ssam", "both"):
        run_ssam(a, genes, mol_um, cell_um, gt, out)
