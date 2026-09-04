"""Boundary-detection metrics against a ground-truth partition on a graph/grid.

Answers reviewer memo point 3: the manuscript is framed around a
boundary-detection principle but evaluates almost entirely through downstream
clustering ARI/NMI, which does not test that principle directly.

Ground-truth boundary points are grid vertices with at least one graph
neighbour in a different annotated domain. A candidate scalar field is then
scored as a soft boundary detector against that set, BSDS-style: a spatial
tolerance is allowed on both sides of the match, because a boundary predicted
one grid step away from the annotation is not a false positive in any
meaningful sense.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def ground_truth_boundary(gt_int, edges, valid=None):
    """Vertices with a graph neighbour in a different domain."""
    gt_int = np.asarray(gt_int)
    n = len(gt_int)
    if valid is None:
        valid = np.ones(n, bool)
    valid = np.asarray(valid, bool)
    b = np.zeros(n, bool)
    for u, v in edges:
        if valid[u] and valid[v] and gt_int[u] != gt_int[v]:
            b[u] = b[v] = True
    return b


def _match(pred_xy, gt_xy, tol):
    """(precision_hits, recall_hits) under a symmetric distance tolerance."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return 0, 0
    d_p = cKDTree(gt_xy).query(pred_xy)[0]      # each prediction -> nearest GT
    d_g = cKDTree(pred_xy).query(gt_xy)[0]      # each GT -> nearest prediction
    return int((d_p <= tol).sum()), int((d_g <= tol).sum())


def pr_curve(score, coords, gt_bnd, valid=None, tol=0.0, n_thresh=60):
    """Precision/recall of `score` as a boundary detector, at tolerance `tol`.

    The sweep is over the *fraction of vertices selected*, not over score
    quantiles. This matters: the candidate fields being compared (sheaf
    disagreement, expression gradient, neighbour distance, principal angle)
    have very differently shaped score distributions, and a fixed grid of
    score quantiles would sample the precision-recall trade-off unevenly
    between them -- penalising the more skewed field for its shape rather
    than its accuracy.
    """
    score = np.asarray(score, float)
    coords = np.asarray(coords, float)
    if valid is None:
        valid = np.ones(len(score), bool)
    valid = np.asarray(valid, bool)

    s, xy, g = score[valid], coords[valid], np.asarray(gt_bnd, bool)[valid]
    gt_xy = xy[g]
    if g.sum() == 0:
        raise ValueError("no ground-truth boundary vertices")

    order = np.argsort(-s)
    n = len(s)
    fracs = np.unique(np.geomspace(max(1.0 / n, 1e-4), 1.0, n_thresh))
    prec, rec = [], []
    for f in fracs:
        k = max(int(round(f * n)), 1)
        sel = order[:k]
        tp_p, tp_g = _match(xy[sel], gt_xy, tol)
        prec.append(tp_p / k)
        rec.append(tp_g / len(gt_xy))
    return np.array(prec), np.array(rec)


def average_precision(prec, rec):
    """Average precision: sum of precision weighted by the recall increment.

    Standard AP, not a trapezoid normalised by the observed recall range --
    that normalisation degenerates for smooth fields, whose recall barely
    varies across thresholds.
    """
    o = np.argsort(rec)
    r, p = rec[o], prec[o]
    if len(r) == 0:
        return float("nan")
    r = np.concatenate([[0.0], r])
    p = np.concatenate([[p[0]], p])
    return float(np.sum(np.diff(r) * p[1:]))


def best_f1(prec, rec):
    f = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f))
    return float(f[i]), float(prec[i]), float(rec[i])


def boundary_distances(score, coords, gt_bnd, valid=None, frac=None):
    """Symmetric distances between the predicted and true boundary sets.

    The predicted set is the top `frac` of vertices by score; `frac` defaults
    to the true boundary prevalence, so every field is compared at matched
    budget rather than at a threshold that happens to favour one of them.
    """
    score = np.asarray(score, float)
    coords = np.asarray(coords, float)
    if valid is None:
        valid = np.ones(len(score), bool)
    valid = np.asarray(valid, bool)
    s, xy, g = score[valid], coords[valid], np.asarray(gt_bnd, bool)[valid]
    if frac is None:
        frac = g.mean()
    k = max(int(round(frac * len(s))), 1)
    sel = np.zeros(len(s), bool)
    sel[np.argsort(-s)[:k]] = True

    gt_xy, pr_xy = xy[g], xy[sel]
    d_p = cKDTree(gt_xy).query(pr_xy)[0]
    d_g = cKDTree(pr_xy).query(gt_xy)[0]
    return {
        "mean_pred_to_gt": float(d_p.mean()),
        "mean_gt_to_pred": float(d_g.mean()),
        "hausdorff_95": float(max(np.percentile(d_p, 95), np.percentile(d_g, 95))),
        "hausdorff": float(max(d_p.max(), d_g.max())),
        "frac_used": float(frac),
    }


def evaluate_field(score, coords, gt_bnd, valid=None, tols=(0.0, 15.0, 30.0)):
    """Full report for one candidate boundary field."""
    out = {}
    for tol in tols:
        p, r = pr_curve(score, coords, gt_bnd, valid, tol=tol)
        f1, pi, ri = best_f1(p, r)
        out[f"AP@{tol:g}"] = average_precision(p, r)
        out[f"F1@{tol:g}"] = f1
        out[f"P@{tol:g}"] = pi
        out[f"R@{tol:g}"] = ri
    out.update(boundary_distances(score, coords, gt_bnd, valid))
    return out
