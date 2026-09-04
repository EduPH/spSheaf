"""Evaluate the boundary FIELD directly, not through downstream clustering.

Reviewer memo point 3. The manuscript is framed around a boundary-detection
principle -- domain boundaries occur where neighbouring local expression
subspaces become incompatible -- but almost every quantitative claim is an ARI
or NMI on the final partition, which tests the whole pipeline rather than the
signal. This script tests the signal.

The annotated domains are converted to ground-truth boundaries on the grid,
and each candidate field is scored as a soft boundary detector: average
precision, best boundary-F1 at a spatial tolerance, and symmetric distances
between the predicted and true boundary sets.

Candidates, all aggregated per vertex and smoothed identically so the
comparison is of the signal and not of the post-processing:

    sheaf        the method
    grassmann    principal angles only
    projection   subspace / covariance distance
    procrustes   alignment residual, no sheaf interpretation
    identity     identity restriction maps (control)
    neighbour    || f_u - f_v ||^2, the plain kernel baseline

Usage:
    python -m experiments.merfish.boundary_eval --slice MERFISH_0.19.h5ad
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from spsheaf import sheaf, paths
from spsheaf.boundary_metrics import ground_truth_boundary, evaluate_field
from spsheaf.edge_scores import EDGE_SCORES, compute_edge_score

from . import _cache

FIELDS = ["sheaf", "grassmann", "projection", "procrustes", "identity", "neighbour"]


def per_vertex(edge_score, edges, n):
    """Mean over incident edges, matching spsheaf.sheaf.compute_boundary_scores."""
    tot = np.zeros(n)
    deg = np.zeros(n)
    for ei, (u, v) in enumerate(edges):
        tot[u] += edge_score[ei]
        tot[v] += edge_score[ei]
        deg[u] += 1
        deg[v] += 1
    return tot / np.maximum(deg, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice", default="MERFISH_0.19.h5ad")
    ap.add_argument("--smooth-radius", type=float, default=60.0)
    ap.add_argument("--tolerances", type=float, nargs="+", default=[0.0, 15.0, 30.0],
                    help="spatial tolerance in microns; 15 = one grid step")
    ap.add_argument("--fields", nargs="+", default=FIELDS, choices=list(EDGE_SCORES))
    a = ap.parse_args()

    c = _cache.load(a.slice)
    grid, edges, stalks = c["grid"], c["edges"], c["stalks"]
    gt, valid = c["gt_int"], c["gt_valid"]
    n = len(grid)

    gt_bnd = ground_truth_boundary(gt, edges, valid)
    print(f"{a.slice}: {n} grid points, {int(valid.sum())} annotated, "
          f"{int(gt_bnd.sum())} ground-truth boundary vertices "
          f"({100*gt_bnd[valid].mean():.1f}% of annotated)\n")

    rows = []
    for kind in a.fields:
        es = compute_edge_score(kind, stalks, edges, c["fib_global"], c["fib_local"])
        field = per_vertex(es, edges, n)
        field = sheaf._smooth_boundary_field(field, grid, a.smooth_radius)
        r = evaluate_field(field, grid, gt_bnd, valid, tols=tuple(a.tolerances))
        r["field"] = kind
        rows.append(r)
        tol = a.tolerances[min(1, len(a.tolerances) - 1)]
        print(f"  {kind:12s} AP@{tol:g}={r[f'AP@{tol:g}']:.3f}  "
              f"F1@{tol:g}={r[f'F1@{tol:g}']:.3f}  "
              f"meanD(pred->gt)={r['mean_pred_to_gt']:.1f}um  "
              f"H95={r['hausdorff_95']:.1f}um")

    df = pd.DataFrame(rows).set_index("field")
    out = paths.results("merfish") / f"boundary_eval_{a.slice.replace('.h5ad','')}.csv"
    df.to_csv(out)
    print(f"\n-> {out}")
    print("\nIf 'sheaf' does not separate from 'grassmann' and 'projection' here,\n"
          "the sheaf framing is not carrying the boundary signal and the\n"
          "manuscript's central claim needs rewording rather than defending.")


if __name__ == "__main__":
    main()
