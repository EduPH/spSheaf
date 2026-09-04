"""Sensitivity to k, and whether k can be chosen without the annotation.

Reviewer memo point 5. The manuscript uses k = 8 on MERFISH and k = 11 on
osmFISH, matching the annotated domain counts, which weakens the claim of
unsupervised inference. This sweeps k and reports, side by side:

  * label-dependent scores (ARI, NMI) -- "benchmark mode, known k"
  * annotation-free criteria (silhouette, spatial coherence, CHAOS, PAS)
    -- "unsupervised mode, estimated k"

The question the output answers is whether an annotation-free criterion would
have selected k = 8 on its own. If it would, the benchmark-mode number is
defensible. If it would not, say so and report both.

Usage:
    python -m experiments.merfish.k_sensitivity --slice MERFISH_0.19.h5ad --k 4 16
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

from spsheaf import clustering, evaluation, paths

from . import _cache


def spatial_coherence(labels, edges):
    """Fraction of graph edges whose endpoints share a label."""
    lab = np.asarray(labels)
    same = sum(1 for u, v in edges if lab[u] == lab[v])
    return same / max(len(edges), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice", default="MERFISH_0.19.h5ad")
    ap.add_argument("--k", type=int, nargs=2, default=[4, 16], metavar=("MIN", "MAX"))
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    c = _cache.load(a.slice)
    grid, edges, fib = c["grid"], c["edges"], c["fib"]
    bsm, gt, valid = c["boundary_smooth"], c["gt_int"], c["gt_valid"]

    rows = []
    for k in range(a.k[0], a.k[1] + 1):
        per_seed = []
        for seed in range(a.seeds):
            lab = np.asarray(clustering.boundary_aware_clustering(
                fib, grid, bsm, edges, n_clusters=k, spatial_weight=0.5,
                boundary_alpha=0.5, n_pca=10, random_state=seed, verbose=False))
            lab, _ = clustering.spatial_cleanup(lab, edges, grid_coords=grid,
                                                verbose=False)
            lab = np.asarray(lab)
            rec = dict(
                k=k, seed=seed,
                ARI=adjusted_rand_score(gt[valid], lab[valid]),
                NMI=normalized_mutual_info_score(gt[valid], lab[valid]),
                coherence=spatial_coherence(lab, edges),
                CHAOS=evaluation.compute_chaos(lab, grid),
                PAS=evaluation.compute_pas(lab, grid),
            )
            rec["silhouette"] = (
                silhouette_score(fib, lab) if len(np.unique(lab)) > 1 else np.nan)
            per_seed.append(rec)
        df_k = pd.DataFrame(per_seed)
        m = df_k.mean(numeric_only=True)
        rows.append(df_k)
        print(f"  k={k:2d}  ARI={m.ARI:.3f}  NMI={m.NMI:.3f}  "
              f"sil={m.silhouette:.3f}  coh={m.coherence:.3f}  "
              f"CHAOS={m.CHAOS:.4f}  PAS={m.PAS:.4f}")

    df = pd.concat(rows, ignore_index=True)
    out = paths.results("merfish") / f"k_sensitivity_{a.slice.replace('.h5ad','')}.csv"
    df.to_csv(out, index=False)

    mean = df.groupby("k").mean(numeric_only=True)
    print(f"\n-> {out}")
    print(f"\n  best k by ARI (uses labels)      : {int(mean.ARI.idxmax())}")
    print(f"  best k by silhouette (label-free): {int(mean.silhouette.idxmax())}")
    print(f"  best k by coherence  (label-free): {int(mean.coherence.idxmax())}")
    print(f"  best k by CHAOS      (label-free): {int(mean.CHAOS.idxmin())}")
    print("\n  Note: coherence and CHAOS both improve monotonically as k falls,\n"
          "  so neither is a usable selector on its own -- report the silhouette\n"
          "  or an elbow criterion, and say plainly which mode each number is in.")


if __name__ == "__main__":
    main()
