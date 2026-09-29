"""Isolate what the sheaf construction contributes, against simpler controls.

Reviewer memo point 2. Extended Data Fig. 3 shows the full pipeline beats
plain k-means, +spatial and no-cleanup variants -- that boundary-aware
weighting helps. It does not show that a *cellular sheaf* provides information
a simpler local-subspace method could not.

Here only the edge score changes. Grid, graph, stalks, smoothing, weighting,
clustering and cleanup are identical across arms, so any difference is
attributable to the score itself:

    sheaf        the method: residual coordinate disagreement after alignment
    sheaf_local  the same, with xi referenced to the LOCAL neighbourhood mean
                 instead of the global mean -- the centering ablation the memo
                 asks for. Note the stalk basis is already locally centered
                 (sklearn PCA re-centers), so this isolates xi alone.
    grassmann    principal angles only: subspace rotation with no coordinates
    projection   subspace / covariance distance, a non-sheaf local-geometry
                 baseline
    procrustes   orthogonal Procrustes residual, no sheaf interpretation
    identity     identity restriction maps: does the alignment step matter?
    neighbour    || f_u - f_v ||^2, the plain kernel baseline

Each arm is reported both as a boundary detector (memo point 3) and through
the downstream partition (ARI/NMI), because the two can disagree and the
disagreement is itself informative.

Usage:
    python -m experiments.merfish.sheaf_ablation --slice MERFISH_0.19.h5ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from spsheaf import clustering, sheaf, paths
from spsheaf.boundary_metrics import ground_truth_boundary, evaluate_field
from spsheaf.edge_scores import EDGE_SCORES, compute_edge_score

from . import _cache
from .boundary_eval import per_vertex

ARMS = ["sheaf", "sheaf_local", "grassmann", "projection",
        "procrustes", "identity", "neighbour"]

# Null controls. Without these the ablation cannot distinguish "the boundary
# field carries signal" from "attenuating some features regularises k-means".
# On slice 0.19 a *random* field already recovers a third of the gain over no
# weighting at all, so the controls are not optional.
CONTROLS = ["none", "random", "shuffled"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice", default="MERFISH_0.19.h5ad")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=list(EDGE_SCORES))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smooth-radius", type=float, default=60.0)
    ap.add_argument("--tolerance", type=float, default=15.0)
    ap.add_argument("--no-controls", action="store_true",
                    help="skip the null-field controls (not recommended)")
    ap.add_argument("--spatial-weight", type=float, default=0.5)
    ap.add_argument("--cache-path", default=None,
                    help="an npz in the MERFISH cache format (e.g. from "
                         "experiments/spacetx/ablation_cache.py) instead of --slice")
    ap.add_argument("--out", default=None, help="output CSV (default: results/merfish/...)")
    a = ap.parse_args()

    if a.cache_path:
        d = np.load(a.cache_path, allow_pickle=True)
        c = {"grid": d["grid"], "fib": d["fib"].astype(float),
             "fib_global": d["fib_global"].astype(float), "fib_local": d["fib_local"].astype(float),
             "stalks": [s for s in d["stalks"].astype(float)],
             "edges": [tuple(e) for e in d["edges"]],
             "boundary_raw": d["boundary_raw"], "boundary_smooth": d["boundary_smooth"],
             "gt_int": d["gt_int"], "gt_valid": d["gt_valid"]}
    else:
        c = _cache.load(a.slice)
    grid, edges, stalks = c["grid"], c["edges"], c["stalks"]
    gt, valid = c["gt_int"], c["gt_valid"]
    n = len(grid)
    gt_bnd = ground_truth_boundary(gt, edges, valid)

    print(f"{a.slice}: {n} grid points, {int(gt_bnd.sum())} boundary vertices, "
          f"k={a.k}, {a.seeds} seeds\n")

    rows = []
    for kind in a.arms:
        es = compute_edge_score(kind, stalks, edges, c["fib_global"], c["fib_local"])
        field = sheaf._smooth_boundary_field(
            per_vertex(es, edges, n), grid, a.smooth_radius)

        bd = evaluate_field(field, grid, gt_bnd, valid, tols=(a.tolerance,))

        aris, nmis = [], []
        for seed in range(a.seeds):
            lab = np.asarray(clustering.boundary_aware_clustering(
                c["fib"], grid, field, edges, n_clusters=a.k, spatial_weight=a.spatial_weight,
                boundary_alpha=0.5, n_pca=10, random_state=seed, verbose=False))
            lab, _ = clustering.spatial_cleanup(lab, edges, grid_coords=grid,
                                                verbose=False)
            lab = np.asarray(lab)
            aris.append(adjusted_rand_score(gt[valid], lab[valid]))
            nmis.append(normalized_mutual_info_score(gt[valid], lab[valid]))

        row = dict(arm=kind,
                   AP=bd[f"AP@{a.tolerance:g}"], F1=bd[f"F1@{a.tolerance:g}"],
                   meanD=bd["mean_pred_to_gt"], H95=bd["hausdorff_95"],
                   ARI=float(np.mean(aris)), ARI_sd=float(np.std(aris)),
                   NMI=float(np.mean(nmis)), NMI_sd=float(np.std(nmis)))
        rows.append(row)
        print(f"  {kind:12s} AP={row['AP']:.3f} F1={row['F1']:.3f} "
              f"meanD={row['meanD']:5.1f}um | "
              f"ARI={row['ARI']:.3f}+-{row['ARI_sd']:.3f} "
              f"NMI={row['NMI']:.3f}+-{row['NMI_sd']:.3f}")

    if not a.no_controls:
        print()
        rng = np.random.default_rng(0)
        for ctrl in CONTROLS:
            if ctrl == "none":
                field, alpha = np.zeros(n), 0.0
            elif ctrl == "random":
                field, alpha = rng.random(n), 0.5
            else:
                field, alpha = c["boundary_smooth"].copy(), 0.5
                rng.shuffle(field)
            aris, nmis = [], []
            for seed in range(a.seeds):
                lab = np.asarray(clustering.boundary_aware_clustering(
                    c["fib"], grid, field, edges, n_clusters=a.k,
                    spatial_weight=a.spatial_weight, boundary_alpha=alpha, n_pca=10,
                    random_state=seed, verbose=False))
                lab, _ = clustering.spatial_cleanup(lab, edges,
                                                    grid_coords=grid, verbose=False)
                lab = np.asarray(lab)
                aris.append(adjusted_rand_score(gt[valid], lab[valid]))
                nmis.append(normalized_mutual_info_score(gt[valid], lab[valid]))
            rows.append(dict(arm=f"CONTROL:{ctrl}", AP=np.nan, F1=np.nan,
                             meanD=np.nan, H95=np.nan,
                             ARI=float(np.mean(aris)), ARI_sd=float(np.std(aris)),
                             NMI=float(np.mean(nmis)), NMI_sd=float(np.std(nmis))))
            print(f"  {'CONTROL:'+ctrl:12s} {'':32s}"
                  f"ARI={np.mean(aris):.3f} NMI={np.mean(nmis):.3f}")

    df = pd.DataFrame(rows).set_index("arm")
    out = Path(a.out) if a.out else paths.results("merfish") / f"sheaf_ablation_{a.slice.replace('.h5ad','')}.csv"
    df.to_csv(out)
    print(f"\n-> {out}")

    if "sheaf" in df.index:
        others = df.drop(index=[i for i in ("sheaf", "sheaf_local") if i in df.index])
        if len(others):
            best = others.ARI.idxmax()
            gap = df.loc["sheaf", "ARI"] - others.loc[best, "ARI"]
            print(f"\n  sheaf minus best non-sheaf control ({best}): "
                  f"ARI {gap:+.3f}")
            if gap <= 0.01:
                print("  -> The sheaf score does NOT separate from a simpler\n"
                      "     local-subspace baseline on this slice. That is the\n"
                      "     reviewer's exact objection; it needs an answer in the\n"
                      "     text rather than a stronger claim.")
            else:
                print("  -> The sheaf score leads the simpler controls. Report this\n"
                      "     table; it is the direct answer to memo point 2.")


if __name__ == "__main__":
    main()
