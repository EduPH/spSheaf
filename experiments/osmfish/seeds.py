"""osmFISH seed sensitivity at the defaults (Methods, "Statistics").

20 k-means initializations of the transferred configuration, scored at cell
level in the registered frame; also the alpha_s = 0.25 comparison the text
quotes. Writes data/results/osmfish/seed_sensitivity.csv (alpha_s, seed, ari, nmi).

    python -m experiments.osmfish.seeds
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from spsheaf import pipeline, paths
from experiments.osmfish.run_osmFISH import (
    load_osmfish_molecules, load_cell_ground_truth, evaluate_at_cell_level)
from experiments.osmfish.elongation import BASE, SLICE_NAME


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.25])
    ap.add_argument("--n-clusters", type=int, default=11)
    a = ap.parse_args()
    root = paths.raw("osmfish")
    mol = load_osmfish_molecules(str(root / "mRNA_coords_raw_counting.hdf5"))
    cell_coords, cell_gt = load_cell_ground_truth(str(root / "osmfish_20260512084934.h5ad"))
    rows = []
    for alpha in a.alphas:
        for sd in range(a.seeds):
            res = pipeline.run_single_slice(
                mol=mol, slice_name=SLICE_NAME, n_clusters=a.n_clusters, k_graph=6, k_nbrs=50,
                spatial_weight=alpha, boundary_alpha=0.5, cleanup=True, random_state=sd,
                verbose=False, **BASE)
            m, *_ = evaluate_at_cell_level(res["grid_coords"], res["labels"], cell_coords, cell_gt,
                                           verbose=False)
            rows.append(dict(alpha_s=alpha, seed=sd, ari=m["ari"], nmi=m["nmi"]))
            print(f"  alpha_s={alpha:.2f} seed={sd:2d}  ARI {m['ari']:.4f}  NMI {m['nmi']:.4f}", flush=True)
        df = pd.DataFrame(rows); d = df[df.alpha_s == alpha]
        print(f"alpha_s={alpha:.2f}: ARI {d.ari.mean():.3f} +- {d.ari.std():.3f} (median {d.ari.median():.3f})")
    pd.DataFrame(rows).to_csv(paths.results("osmfish") / "seed_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
