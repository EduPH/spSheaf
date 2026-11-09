"""How different are adjacent stalks? Principal angles on real edges.

With k_nbrs = 50 on a 15 um grid, adjacent vertices share most of their
stalk neighbourhoods, so a referee may ask whether the SVD alignment of
Section 2.1 is ever non-trivial. For every edge of every slice this records
the principal angles between the two stalk subspaces (arccos of the singular
values of U_u^T U_v), the Frobenius distance of the restriction-map product
F_u^T F_v from the identity, and the same angles for a null in which each
vertex's neighbourhood is replaced by a random set of k_nbrs grid points.
Reads the per-slice cache of _cache.py (grid, normalised fibers, stalks,
edges at the reference configuration), so no raw data is needed. Writes
data/results/merfish/stalk_angles.csv (per edge, per slice) and prints the
quantiles per slice.

    python -m experiments.merfish.stalk_angles [--cache-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import numpy as np, pandas as pd

import argparse
from pathlib import Path

SLICES = ["0.04", "0.09", "0.14", "0.19", "0.24"]


def angles(stalks, edges):
    out = np.empty((len(edges), stalks[0].shape[1])); dist = np.empty(len(edges))
    for i, (u, v) in enumerate(edges):
        L, S, Rt = np.linalg.svd(stalks[u].T @ stalks[v], full_matrices=False)
        out[i] = np.degrees(np.arccos(np.clip(S, -1, 1)))
        dist[i] = np.linalg.norm(L @ Rt - np.eye(len(S)))
    return out, dist


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--cache-dir", default=None, help="folder of revision_s*.npz (default: SPSHEAF_DATA/cache/merfish)")
    a.add_argument("--out-dir", default=None, help="default: SPSHEAF_DATA/results/merfish")
    a = a.parse_args()
    if a.cache_dir is None or a.out_dir is None:
        from spsheaf import paths
    cache = Path(a.cache_dir) if a.cache_dir else paths.cache("merfish")
    out = Path(a.out_dir) if a.out_dir else paths.results("merfish")
    rows = []; summ = []
    for s in SLICES:
        d = np.load(cache / f"revision_s{s}.npz")
        g, fn, edges = d["grid"], d["fib"], [tuple(e) for e in d["edges"]]
        stalks = list(d["stalks"])                       # (n, genes, d)
        k_nbrs, dim = 50, stalks[0].shape[1]
        A, D = angles(stalks, edges)
        rng = np.random.default_rng(0)
        null = []
        for v in range(len(g)):
            nb = rng.choice(len(g), k_nbrs, replace=False)
            X = (fn[nb] - fn[nb].mean(0)).T
            U, _, _ = np.linalg.svd(X, full_matrices=False)
            null.append(U[:, :dim])
        A0, D0 = angles(null, edges)
        rows.append(pd.DataFrame(dict(slice=s, edge=np.arange(len(edges)),
                                      max_angle=A.max(1), mean_angle=A.mean(1), min_angle=A.min(1),
                                      restriction_dist=D,
                                      null_max_angle=A0.max(1), null_mean_angle=A0.mean(1))))
        q = lambda x: np.percentile(x, [50, 90, 99])
        summ.append(dict(slice=s, n_edges=len(edges),
                         max_angle_q50=q(A.max(1))[0], max_angle_q90=q(A.max(1))[1], max_angle_q99=q(A.max(1))[2],
                         mean_angle_q50=q(A.mean(1))[0], mean_angle_q90=q(A.mean(1))[1],
                         frac_max_angle_gt10=(A.max(1) > 10).mean(), frac_max_angle_gt30=(A.max(1) > 30).mean(),
                         restriction_dist_q50=q(D)[0], restriction_dist_q90=q(D)[1],
                         null_max_angle_q50=q(A0.max(1))[0], null_mean_angle_q50=q(A0.mean(1))[0]))
        print(pd.DataFrame([summ[-1]]).round(3).T.to_string(header=False), "\n", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out / "stalk_angles.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "stalk_angles_summary.csv", index=False)
    print(pd.DataFrame(summ).round(2).to_string())


if __name__ == "__main__":
    main()
