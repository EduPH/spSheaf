"""SSAM in *guided* mode on the Moffitt MERFISH slices.

Run as: python run_ssam_guided.py [slice ...]

Rationale. In de-novo mode SSAM discovers cell types by clustering local maxima,
and the published MERFISH analysis then curates those clusters by hand
(`exclude_and_merge_clusters` with manually chosen indices). An automated
benchmark cannot reproduce that curation, so de-novo mode understates what SSAM
achieves in the paper. Guided mode is SSAM's own answer to this: cell-type
signatures are supplied and the vector field is mapped onto them by correlation.

Here the signatures are the mean expression profiles of the 15 annotated
`cell_class` labels shipped with the benchmark AnnData. That gives SSAM
ground-truth-derived cell-type information that no other method in the
comparison receives, so the resulting numbers are an upper bound in SSAM's
favour rather than a like-for-like score. Reported as such.

Reuses the KDE cached by run_ssam_merfish.py in work/ssam_<slice>.zarr.
"""
import contextlib
import io
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import ssam

from benchmarks.segfree import common as bc

from spsheaf import paths as _paths
OUT = str(_paths.results("segfree/ssam_guided"))
os.makedirs(OUT, exist_ok=True)

EXP_THRES, NORM_THRES = 0.0055, 0.0035
FILTER_PARAMS = {"block_size": 151, "method": "mean", "mode": "constant", "offset": 0}
SWEEP = [dict(n_clusters=n, merge_remote=True, merge_thres=t, norm_thres=nt)
         for n in (8, 12, 20) for t in (0.6, 0.7, 0.8, 0.9) for nt in (1000, 4000)]


def celltype_centroids(slice_id, genes):
    """Mean normalised profile per annotated cell_class, in SSAM's units.

    SSAM correlates its scaled vector field against centroids that were built
    the same way as its own normalised vectors: per-vector L1 normalisation to a
    fixed size followed by log1p. We mirror normalize_vectors() defaults
    (normalize_vector=True, size_after_normalization=10, log_transform=True) on
    the raw count matrix, then z-score across genes as scale_vectors() does.
    """
    a = ad.read_h5ad(f"{bc.MER_DIR}MERFISH_{slice_id}_counts.h5ad")
    import scipy.sparse as sp
    a = a[:, [g for g in genes]]
    X = (a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype(float)
    s = X.sum(1, keepdims=True)
    s[s == 0] = 1
    X = np.log1p(X / s * 10.0)
    lbl = a.obs["cell_class"].astype(str).values
    classes = sorted(set(lbl))
    C = np.vstack([X[lbl == c].mean(0) for c in classes])
    mu, sd = C.mean(0), C.std(0)
    sd[sd == 0] = 1
    return (C - mu) / sd, classes


def run_slice(slice_id, molecules):
    adata, mols = bc.load_merfish_slice(slice_id, molecules=molecules)
    x0, y0 = mols.x.min() - 10.0, mols.y.min() - 10.0
    store = f"work/ssam_{slice_id}.zarr"
    if not os.path.exists(store):
        raise SystemExit(f"{store} missing - run run_ssam_merfish.py {slice_id} first")
    ds = ssam.SSAMDataset(store=store)
    an = ssam.SSAMAnalysis(ds, ncores=10, verbose=False)
    an.set_thresholds(expression_threshold=EXP_THRES, norm_threshold=NORM_THRES)

    cents, classes = celltype_centroids(slice_id, list(ds.genes))
    err = io.StringIO()
    # map_celltypes(centroids=...) uses the supplied centroids but does not
    # register them on the dataset, so bin_celltypemaps would still size its
    # count vectors from whatever de-novo centroids the zarr store holds.
    ds.centroids = cents
    with contextlib.redirect_stderr(err):
        t = time.time()
        an.map_celltypes(centroids=cents)
        an.filter_celltypemaps(min_norm="local", filter_params=FILTER_PARAMS,
                               min_r=0.6, min_blob_area=7, fill_blobs=True)
        an.bin_celltypemaps(step=10, radius=100)
        t_map = time.time() - t

    cell_xy = adata.obsm["spatial"]
    gt = adata.obs["domain"].astype(str).values
    out = {"slice": slice_id, "n_celltypes_given": len(classes),
           "celltype_classes": classes, "map_s": t_map, "sweep": []}

    def evaluate(kw, name, save=False):
        with contextlib.redirect_stderr(err):
            t = time.time()
            an.find_domains(**kw)
            dt = time.time() - t
        dom = np.asarray(ds.inferred_domains)
        pred = bc.nearest_pixel_labels(dom[..., dom.shape[2] // 2], cell_xy, x0, y0)
        pred = bc.fill_background(pred, cell_xy, bg=-1)
        m = bc.all_metrics(gt, pred, cell_xy)
        m.update(find_domains_s=dt, params=kw)
        if save:
            pd.DataFrame({"cell": adata.obs_names, "gt": gt, "pred": pred}).to_csv(
                f"{OUT}/labels_{slice_id}_{name}.csv", index=False)
        return m

    out["published"] = evaluate(dict(n_clusters=20, merge_remote=True, merge_thres=0.8,
                                     norm_thres=4000), "published", save=True)
    out["sweep"] = [evaluate(kw, "sweep") for kw in SWEEP]
    best = max(out["sweep"], key=lambda r: r["ARI"])
    out["best_sweep"] = evaluate(best["params"], "best_sweep", save=True)
    print(f"[ssam-guided {slice_id}] published ARI={out['published']['ARI']:.3f} "
          f"NMI={out['published']['NMI']:.3f} | best ARI={best['ARI']:.3f} "
          f"NMI={best['NMI']:.3f} k={best['n_pred_domains']}", flush=True)
    with open(f"{OUT}/metrics_{slice_id}.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    slices = sys.argv[1:] or bc.MER_SLICES
    molecules = pd.read_csv(bc.MER_MOL)
    for s in slices:
        run_slice(s, molecules)
