"""SSAM cell-type resolution sweep on a finished KDE store (phase 1b).

SSAM's published MERFISH analysis found 68 de novo clusters at Louvain
resolution 0.15 with version 1.0; version 1.1.3 builds the SNN graph
differently and adds a medoid-correlation outlier filter that leaves ~70% of
local maxima unassigned, so the same setting yields 6 clusters here. The
domain step then has almost nothing to work with. To give SSAM the same
in-sample latitude as the other comparators, the resolution is swept (with
the outlier filter off, as for the SpaceTx sections) and each cell-type map
is carried through bin_celltypemaps to its own store; the domain sweep
(ssam_sweep_offline.py) then runs on each, and the best is reported.

    SSAM_TAG=_sct python -m benchmarks.segfree.ssam_resolution 0.19 --resolutions 1 2 4 8
writes data/results/segfree/ssam/store_<slice>_sct_r<res>.npz
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, sys, time
import numpy as np, pandas as pd, ssam
from spsheaf import paths
from benchmarks.segfree import common as bc
from benchmarks.segfree.run_ssam import OUT, FILTER_PARAMS, PCA_DIMS, SAMPLING_DISTANCE
from benchmarks.segfree import ssam_domains as sd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slice_id"); ap.add_argument("--resolutions", type=float, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()
    tag = os.environ.get("SSAM_TAG", "")
    store_path = f"work/ssam_{a.slice_id}{tag}.zarr"
    base = np.load(f"{OUT}/store_{a.slice_id}{tag}.npz", allow_pickle=True)
    ds = ssam.SSAMDataset(store=store_path); an = ssam.SSAMAnalysis(ds, ncores=int(os.environ.get("SSAM_NCORES", "4")), verbose=False)
    err = io.StringIO()
    for res in a.resolutions:
        t = time.time()
        with contextlib.redirect_stderr(err):
            an.cluster_vectors(method="louvain", pca_dims=PCA_DIMS, resolution=res, max_correlation=1.0, metric="correlation",
                               centroid_correction_threshold=0.6, random_state=0, outlier_detection_method=None)
            an.map_celltypes()
            an.filter_celltypemaps(min_norm="local", filter_params=FILTER_PARAMS, min_r=0.6, min_blob_area=7, fill_blobs=True)
            an.bin_celltypemaps(step=10, radius=100)
        lab = np.asarray(ds.cluster_labels); n_ct = int(len(np.unique(lab[lab >= 0])))
        store = sd.load_store(store_path)
        rtag = f"{tag}_r{res:g}"
        np.savez_compressed(f"{OUT}/store_{a.slice_id}{rtag}.npz", binned=store["binned"],
                            vf_norm_shape=np.array(store["vf_norm_shape"]), filtered=store["filtered"],
                            cell_xy=base["cell_xy"], gt=base["gt"], cell_names=base["cell_names"], x0=base["x0"], y0=base["y0"],
                            n_molecules=base["n_molecules"], n_celltypes=n_ct, n_localmax=base["n_localmax"],
                            timings=json.dumps(dict(resolution=res, seconds=time.time() - t)))
        print(f"[ssam {a.slice_id}{rtag}] resolution {res:g}: {n_ct} cell types ({time.time()-t:.0f} s) -> store_{a.slice_id}{rtag}.npz", flush=True)


if __name__ == "__main__":
    main()
