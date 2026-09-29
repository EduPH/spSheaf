"""Which annotated domain splits when k is raised above the annotation?

The five MERFISH sweeps put the silhouette's peak at k = 8-10 against an
annotated 8. One reading is that an annotated domain contains substructure
the annotation does not resolve. This runs k = 8, 9, 10 on every slice at the
manuscript's configuration and stores what is needed to ask whether the same
annotated domain splits, in the same place, on every slice -- grid coordinates,
labels at each k, the annotation on the grid, and the normalised fibers, from
which a segmentation-free differential-expression test between the two halves
can be run (``k_split_analysis.py``).

    python -m experiments.merfish.k_split --slice MERFISH_0.19.h5ad
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from spsheaf import paths, pipeline
from experiments.merfish import _cache
from experiments.merfish.k_selection import CFG, SLICE_COL


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--slice", default="MERFISH_0.19.h5ad")
    a.add_argument("--k", type=int, nargs="+", default=[8, 9, 10])
    a.add_argument("--seed", type=int, default=42)
    a = a.parse_args()

    df = pd.read_parquet(_cache.parquet_path(a.slice))
    if SLICE_COL not in df.columns:
        df[SLICE_COL] = a.slice
    out = paths.results("merfish") / f"k_split_{a.slice.replace('.h5ad', '')}.npz"
    store = {}
    for k in a.k:
        t0 = time.time()
        res = pipeline.run_single_slice(
            mol=df, slice_name=a.slice, n_clusters=k, cleanup=True,
            eval_at_cell_level=False, random_state=a.seed, verbose=False, **CFG)
        if "grid_coords" not in store:
            store["grid_coords"] = np.asarray(res["grid_coords"])
            store["fibers_norm"] = np.asarray(res["fibers_norm"], dtype=np.float32)
            store["gt_int"] = np.asarray(res["gt_int"])
            store["gt_names"] = np.asarray(res["gt_names"], dtype=str)
            store["gene_names"] = np.asarray(sorted(df.gene.unique()), dtype=str)
        store[f"labels_k{k}"] = np.asarray(res["labels"])
        print(f"{a.slice} k={k}: ARI {res['metrics']['ari']:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **store)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
